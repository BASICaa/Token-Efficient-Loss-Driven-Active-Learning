"""Offline, full-answer, length-matched selection and replay diagnostics.

This is a fresh development experiment, not the historical 100-example recipe.
Preparation reads only local inputs/tokenizer; run explicitly starts training.
No teacher or provider client is created. Final-test scoring is not supported.
"""

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import random
import time

from TEFLD.src.candidate_pool import LeakageGuard, candidate_id
from TEFLD.src.dataschema import Training_Sample
from TEFLD.src.encoding import encode_sample, supervised_tokens
from TEFLD.src.experiment_runner import require_adapter, save_adapter
from TEFLD.src.experiment_settings import TrainingSettings
from TEFLD.src.helper import read_json_file, stable_json_hash, write_json_file
from TEFLD.src.runtime import cleanup, file_fingerprint, load_runtime, set_seed
from TEFLD.src.training_safety import require_finite
from TEFLD.src.transactions import section_lock
from TEFLD.src.validation import validate_sample
from TEFLD.src.validation_data import SplitData, score_samples

ROOT = Path(__file__).resolve().parent
SELECTIONS = ("random", "loss", "validation")
REPLAYS = ("off", "random", "guided")


def code_identity():
    files = [Path(__file__), *sorted((ROOT / "TEFLD/src").glob("*.py"))]
    return stable_json_hash(
        {str(p.relative_to(ROOT)): file_fingerprint(p) for p in files}
    )


def model_identity(path):
    path = Path(path)
    names = {
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "chat_template.jinja",
        "generation_config.json",
        "model.safetensors.index.json",
    }
    files = sorted(
        p for p in path.iterdir() if p.suffix == ".safetensors" or p.name in names
    )
    if not any(p.suffix == ".safetensors" for p in files):
        raise ValueError("A local model with safetensors weights is required.")
    return stable_json_hash({p.name: file_fingerprint(p) for p in files})


def make_schedule(records, seed, updates, microbatches, warmup_updates):
    if updates <= warmup_updates or warmup_updates < 1 or microbatches < 1:
        raise ValueError(
            "Require positive microbatches and 0 < warmup_updates < updates."
        )
    count = updates * microbatches
    if len(records) < count:
        raise ValueError(
            f"Need at least {count} unique fitting candidates; bank has {len(records)}."
        )
    rng = random.Random(seed)
    ids = sorted(records)
    rng.shuffle(ids)
    warmup = ids[: warmup_updates * microbatches]
    remaining = Counter(records[i]["answer_tokens"] for i in ids[len(warmup) :])
    anchors = {records[i]["answer_tokens"] for i in warmup}
    schedule = []
    for update in range(updates):
        slots = []
        for slot in range(microbatches):
            if update < warmup_updates:
                identity = warmup[update * microbatches + slot]
                slots.append(
                    {
                        "length": records[identity]["answer_tokens"],
                        "warmup_id": identity,
                        "replay_slot": False,
                    }
                )
                continue
            # A predeclared first-slot opportunity, anchored in common warmup,
            # guarantees that every replay arm has a matching exposed candidate.
            choices = sorted(k for k, v in remaining.items() if v > 0)
            preferred = [k for k in choices if k in anchors] if slot == 0 else []
            length = rng.choice(preferred or choices)
            remaining[length] -= 1
            slots.append(
                {
                    "length": length,
                    "warmup_id": None,
                    "replay_slot": bool(slot == 0 and length in anchors),
                }
            )
        schedule.append(slots)
    return schedule


def choose(options, selection, weakness, rng, losses=None):
    options = sorted(options)
    if not options:
        raise RuntimeError(
            "Length-matched candidate quota exhausted; no substitution or generation."
        )
    if selection == "random":
        return rng.choice(options)
    if selection == "guided":
        if losses is None or any(i not in losses for i in options):
            raise ValueError("Guided replay requires fresh training-only scores.")
        return min(options, key=lambda i: (-losses[i], i))
    grouped = defaultdict(list)
    for identity, category in options:
        grouped[category].append(identity)
    categories = sorted(grouped)
    fallback = sum(weakness.values()) / len(weakness) if weakness else 1.0
    category = rng.choices(
        categories,
        weights=[max(1e-5, weakness.get(c, fallback)) for c in categories],
        k=1,
    )[0]
    return rng.choice(grouped[category])


def select_group(
    records, slots, seen, selection, replay, weakness, replay_losses, seed
):
    rng = random.Random(seed)
    selected, trace = [], []
    for slot in slots:
        is_replay = slot["replay_slot"] and replay != "off"
        if slot["warmup_id"]:
            options = [slot["warmup_id"]]
            identity = options[0]
        else:
            options = [
                i
                for i, row in records.items()
                if row["answer_tokens"] == slot["length"]
                and ((i in seen) if is_replay else (i not in seen))
                and i not in selected
            ]
            if is_replay:
                identity = choose(options, replay, {}, rng, replay_losses)
            elif selection == "random":
                identity = choose(options, "random", {}, rng)
            else:
                identity = choose(
                    [(i, records[i]["category"]) for i in options],
                    selection,
                    weakness,
                    rng,
                )
        trace.append(
            {
                "candidate_id": identity,
                "category": records[identity]["category"],
                "answer_tokens": slot["length"],
                "replay": is_replay,
                "eligible_candidates": len(options),
                "eligible_categories": len({records[i]["category"] for i in options}),
                "selection_weight": (
                    None
                    if selection == "random" or is_replay
                    else weakness.get(records[identity]["category"])
                ),
                "output_origin": records[identity]["origin"],
                "replay_training_loss": (
                    replay_losses.get(identity) if is_replay else None
                ),
            }
        )
        selected.append(identity)
    return selected, trace


def optimizer_step(model, optimizer, encoded, max_grad_norm=1.0):
    import torch

    total = sum(supervised_tokens(x) for x in encoded)
    if not encoded or any(supervised_tokens(x) < 1 for x in encoded):
        raise ValueError("Every example requires complete positive answer supervision.")
    optimizer.zero_grad(set_to_none=True)
    losses = []
    try:
        device = next(model.parameters()).device
        for item in encoded:
            loss = require_finite(
                model(**{k: v.unsqueeze(0).to(device) for k, v in item.items()}).loss
            )
            losses.append(float(loss.detach()))
            (loss * (supervised_tokens(item) / total)).backward()
        params = [p for p in model.parameters() if p.requires_grad]
        for parameter in params:
            if parameter.grad is not None:
                require_finite(parameter.grad, "gradient")
        norm = torch.nn.utils.clip_grad_norm_(
            params, max_grad_norm, error_if_nonfinite=True
        )
        require_finite(norm, "gradient norm")
        lr = optimizer.param_groups[0]["lr"]
        optimizer.step()
    except BaseException:
        optimizer.zero_grad(set_to_none=True)
        raise
    return {
        "answer_tokens": total,
        "example_losses": losses,
        "token_weighted_training_loss": sum(
            loss * supervised_tokens(item) for loss, item in zip(losses, encoded)
        )
        / total,
        "gradient_norm_before_clipping": float(norm),
        "clipped": bool(norm > max_grad_norm),
        "learning_rate": lr,
    }


def prepare(
    output,
    dataset,
    model,
    bank_path=None,
    *,
    seeds=(42, 43, 44),
    updates=25,
    microbatches=4,
    warmup_updates=2,
    round_updates=2,
    replay_ablation=False,
    optimizer_lifecycle="reset-per-round",
    per_category=8,
):
    from transformers import AutoTokenizer

    output = Path(output).resolve()
    if output.exists():
        raise ValueError(
            "Choose a fresh output directory; use run to resume a prepared study."
        )
    if (
        round_updates < 1
        or per_category < 1
        or not seeds
        or len(set(seeds)) != len(seeds)
    ):
        raise ValueError("Invalid round/evaluation/seed settings.")
    if optimizer_lifecycle not in {"reset-per-round", "continuous"}:
        raise ValueError("Unknown optimizer lifecycle.")
    protocol = read_json_file(
        ROOT / "experiment_design_2026-10-18/experiment_protocol_hibana.json", {}
    )
    protocol["dataset"]["source"] = str(Path(dataset).resolve())
    protocol["dataset"]["manifest"] = str(
        ROOT / "experiment_design_2026-10-18/split_manifest.json"
    )
    protocol["common_training"]["model_path"] = str(Path(model).resolve())
    protocol["common_training"]["epochs_per_round"] = 1
    protocol["common_training"]["gradient_accumulation_steps"] = microbatches
    # SplitData verifies the frozen source and split fingerprints before use.
    protocol_path = output / "protocol.json"
    write_json_file(protocol_path, protocol)
    data = SplitData(protocol_path)
    tokenizer = AutoTokenizer.from_pretrained(
        model, local_files_only=True, trust_remote_code=False
    )
    guard = LeakageGuard(data)
    records, rejected = {}, Counter()
    if bank_path:
        from run_bank_recovery import verified_bank

        source = [
            (None, Training_Sample.model_validate(x)) for x in verified_bank(bank_path)
        ]
    else:
        source = [
            (i, data.sample(i, "training_pool"))
            for i in sorted(data.ids["training_pool"])
        ]
    for row_id, sample in source:
        try:
            validate_sample(sample).require()
            guard.require(sample)
            encoded = encode_sample(
                sample, tokenizer, protocol["common_training"]["max_length"]
            )
            identity = candidate_id(sample)
            if identity in records:
                rejected["duplicate_prompt"] += 1
                continue
            records[identity] = {
                "sample": sample.model_dump(mode="json"),
                "category": sample.tag,
                "answer_tokens": supervised_tokens(encoded),
                "row_id": row_id,
                "origin": (
                    "verified_synthetic_bank" if bank_path else "original_dataset"
                ),
            }
        except ValueError as exc:
            rejected[str(exc)] += 1
    schedules = {
        str(seed): make_schedule(records, seed, updates, microbatches, warmup_updates)
        for seed in seeds
    }
    if replay_ablation and any(
        not any(slot["replay_slot"] for group in s for slot in group)
        for s in schedules.values()
    ):
        raise ValueError(
            "This bank lacks matching warmup-length replay opportunities; use a larger bank."
        )
    # An immutable, identical training probe for every arm and seed. Exposure
    # membership is reported rather than assuming these rows stay untrained.
    probe = []
    for row_id in sorted(data.ids["training_pool"]):
        sample = data.sample(row_id, "training_pool")
        if sum(x["category"] == sample.tag for x in probe) >= per_category:
            continue
        try:
            guard.require(sample)
            encode_sample(sample, tokenizer, protocol["common_training"]["max_length"])
        except ValueError:
            continue
        probe.append({"row_id": row_id, "category": sample.tag})
    if any(
        sum(x["category"] == c for x in probe) != per_category
        for c in protocol["dataset"]["categories"]
    ):
        raise ValueError("Cannot prepare a balanced fixed training probe.")
    write_json_file(output / "bank.json", records)
    settings = {
        "updates": updates,
        "microbatches": microbatches,
        "warmup_updates": warmup_updates,
        "round_updates": round_updates,
        "optimizer_lifecycle": optimizer_lifecycle,
        "per_category": per_category,
        "seeds": list(seeds),
        "replays": list(REPLAYS) if replay_ablation else ["off"],
    }
    plan = {
        "version": 1,
        "scope": "Development: selection conditional on exact answer length; full answers; no provider calls.",
        "code_sha256": code_identity(),
        "protocol_sha256": file_fingerprint(protocol_path),
        "model_sha256": model_identity(model),
        "bank_sha256": file_fingerprint(output / "bank.json"),
        "bank_source_sha256": (
            file_fingerprint(bank_path) if bank_path else protocol["dataset"]["sha256"]
        ),
        "settings": settings,
        "schedules": schedules,
        "fixed_training_probe": probe,
        "curriculum_rows": [
            i for i, _ in data.samples("curriculum_validation", per_category)
        ],
        "checkpoint_rows": [
            i for i, _ in data.samples("checkpoint_validation", per_category)
        ],
        "tokens_by_seed": {
            seed: sum(slot["length"] for group in schedule for slot in group)
            for seed, schedule in schedules.items()
        },
        "candidate_counts": dict(Counter(x["category"] for x in records.values())),
        "rejected": dict(rejected),
        "final_test_scored": False,
        "new_provider_calls": 0,
        "checkpoint_rule": "Final fixed-budget checkpoint for every arm; curves are diagnostic, no best-checkpoint selection.",
        "replay_policy": "Separate factorial: random vs highest current training NLL among same-length previously trained examples. One predeclared slot per update when warmup length permits; no historical threshold/cooldown policy. Training-only replay scores refreshed each round.",
    }
    write_json_file(output / "plan.json", plan)
    return {
        "status": "prepared",
        "candidates": len(records),
        "updates": updates,
        "tokens_by_seed": plan["tokens_by_seed"],
        "arms_per_seed": 3 * len(settings["replays"]),
        "provider_calls": 0,
        "training_started": False,
    }


def evaluate(model, tokenizer, data, plan, records, seen, step, tokens, *, seed):
    maximum = data.protocol["common_training"]["max_length"]
    train = [
        (x["row_id"], data.sample(x["row_id"], "training_pool"))
        for x in plan["fixed_training_probe"]
    ]
    curriculum = [
        (i, data.sample(i, "curriculum_validation")) for i in plan["curriculum_rows"]
    ]
    checkpoint = [
        (i, data.sample(i, "checkpoint_validation")) for i in plan["checkpoint_rows"]
    ]
    training_feedback = [
        (i, Training_Sample.model_validate(records[identity]["sample"]))
        for i, identity in enumerate(sorted(seen))
    ]
    warmup_ids = [
        slot["warmup_id"]
        for group in plan["schedules"][str(seed)][: plan["settings"]["warmup_updates"]]
        for slot in group
    ]
    warmup = [
        (i, Training_Sample.model_validate(records[identity]["sample"]))
        for i, identity in enumerate(warmup_ids)
    ]
    record = {
        "update": step,
        "tokens": tokens,
        "training_probe": score_samples(model, tokenizer, train, maximum),
        "warmup_probe": score_samples(model, tokenizer, warmup, maximum),
        "curriculum": score_samples(model, tokenizer, curriculum, maximum),
        "checkpoint": score_samples(model, tokenizer, checkpoint, maximum),
        "training_probe_exposed": sum(
            candidate_id(sample) in seen for _, sample in train
        ),
    }
    feedback = (
        score_samples(model, tokenizer, training_feedback, maximum)
        if training_feedback
        else {"categories": {}, "rows": []}
    )
    losses = {
        identity: row["loss"] for identity, row in zip(sorted(seen), feedback["rows"])
    }
    record["seen_training"] = feedback
    return record, losses


def run_arm(output, selection, replay, seed):
    import torch

    output = Path(output).resolve()
    plan = read_json_file(output / "plan.json", {})
    settings = plan["settings"]
    if (
        selection not in SELECTIONS
        or replay not in settings["replays"]
        or seed not in settings["seeds"]
    ):
        raise ValueError("Arm/seed is not in the prepared plan.")
    if (
        plan["code_sha256"] != code_identity()
        or plan["protocol_sha256"] != file_fingerprint(output / "protocol.json")
        or plan["bank_sha256"] != file_fingerprint(output / "bank.json")
    ):
        raise ValueError(
            "Prepared code, protocol or bank changed; prepare a fresh study."
        )
    records = read_json_file(output / "bank.json", {})
    data = SplitData(output / "protocol.json")
    if (
        model_identity(data.protocol["common_training"]["model_path"])
        != plan["model_sha256"]
    ):
        raise ValueError("Local base model changed.")
    directory = output / f"seed_{seed}" / f"{selection}__replay_{replay}"
    with section_lock(output), section_lock(directory):
        state = read_json_file(directory / "state.json", {})
        identity = {
            "plan_sha256": file_fingerprint(output / "plan.json"),
            "selection": selection,
            "replay": replay,
            "seed": seed,
        }
        if state and state["identity"] != identity:
            raise ValueError("Run identity changed.")
        if state.get("status") == "complete":
            return state
        if state.get("adapter"):
            require_adapter(state["adapter"])
        model, tokenizer = load_runtime(
            data.protocol["common_training"]["model_path"],
            seed=seed,
            adapter=state.get("adapter"),
            training_config=TrainingSettings.from_protocol(data.protocol).model_dump(),
        )
        try:
            if not state:
                observation, _ = evaluate(
                    model, tokenizer, data, plan, records, set(), 0, 0, seed=seed
                )
                state = {
                    "identity": identity,
                    "status": "running",
                    "update": 0,
                    "tokens": 0,
                    "seen": [],
                    "updates": [],
                    "evaluations": [observation],
                    "selection_trace": [],
                    "rounds": [],
                    "replay_losses": {},
                }
                write_json_file(directory / "state.json", state)
            seen = set(state["seen"])
            maximum = data.protocol["common_training"]["max_length"]
            encoded_cache = {}
            for i, record in records.items():
                sample = Training_Sample.model_validate(record["sample"])
                validate_sample(sample).require()
                # Every encoded example retains its complete answer.
                encoded_cache[i] = encode_sample(sample, tokenizer, maximum)
                if supervised_tokens(encoded_cache[i]) != record["answer_tokens"]:
                    raise ValueError("Prepared answer length changed.")
            params = [p for p in model.parameters() if p.requires_grad]
            optimizer = None
            scheduler = None
            if settings["optimizer_lifecycle"] == "continuous":
                optimizer = torch.optim.AdamW(
                    params,
                    lr=TrainingSettings.from_protocol(data.protocol).learning_rate,
                    weight_decay=0,
                )
                scheduler = torch.optim.lr_scheduler.LambdaLR(
                    optimizer, lambda step: max(0.0, 1 - step / settings["updates"])
                )
                if state.get("optimizer_file"):
                    if (
                        file_fingerprint(state["optimizer_file"])
                        != state["optimizer_sha256"]
                    ):
                        raise ValueError("Optimizer checkpoint changed.")
                    payload = torch.load(state["optimizer_file"], weights_only=True)
                    optimizer.load_state_dict(payload["optimizer"])
                    scheduler.load_state_dict(payload["scheduler"])
            while state["update"] < settings["updates"]:
                start = state["update"]
                # Warmup always ends on a checkpoint/feedback boundary.
                end = min(settings["updates"], start + settings["round_updates"])
                if start < settings["warmup_updates"]:
                    end = min(end, settings["warmup_updates"])
                reset = settings["optimizer_lifecycle"] == "reset-per-round"
                if reset:
                    optimizer = torch.optim.AdamW(
                        params,
                        lr=TrainingSettings.from_protocol(data.protocol).learning_rate,
                        weight_decay=0,
                    )
                    scheduler = torch.optim.lr_scheduler.LambdaLR(
                        optimizer, lambda step: max(0.0, 1 - step / (end - start))
                    )
                # Diagnostic/checkpoint scores are never read for selection.
                observation = state["evaluations"][-1]
                weakness = (
                    observation["seen_training"]["categories"]
                    if selection == "loss"
                    else (
                        observation["curriculum"]["categories"]
                        if selection == "validation"
                        else {}
                    )
                )
                replay_losses = state["replay_losses"]
                started = time.perf_counter()
                for update in range(start, end):
                    slots = plan["schedules"][str(seed)][update]
                    ids, traces = select_group(
                        records,
                        slots,
                        seen,
                        selection,
                        replay,
                        weakness,
                        replay_losses,
                        seed * 1000003 + update,
                    )
                    set_seed(seed * 1000033 + update)
                    model.train()
                    result = optimizer_step(
                        model, optimizer, [encoded_cache[i] for i in ids]
                    )
                    expected = sum(slot["length"] for slot in slots)
                    if result["answer_tokens"] != expected:
                        raise RuntimeError("Matched token contract violated.")
                    result.update(
                        update=update + 1,
                        optimizer_reset=bool(reset and update == start),
                        new_examples=sum(not t["replay"] for t in traces),
                        replay_examples=sum(t["replay"] for t in traces),
                    )
                    for identity_id, loss in zip(ids, result["example_losses"]):
                        replay_losses[identity_id] = loss
                    scheduler.step()
                    state["tokens"] += result["answer_tokens"]
                    state["updates"].append(result)
                    state["selection_trace"].extend(
                        {"update": update + 1, **trace} for trace in traces
                    )
                    seen.update(ids)
                state["update"] = end
                state["seen"] = sorted(seen)
                observation, replay_losses = evaluate(
                    model,
                    tokenizer,
                    data,
                    plan,
                    records,
                    seen,
                    end,
                    state["tokens"],
                    seed=seed,
                )
                state["evaluations"].append(observation)
                state["replay_losses"] = replay_losses
                checkpoint = directory / f"checkpoint_{end:06d}"
                save_adapter(
                    model,
                    tokenizer,
                    checkpoint,
                    {
                        "plan_sha256": state["identity"]["plan_sha256"],
                        "update": end,
                        "tokens": state["tokens"],
                    },
                )
                if not reset:
                    optimizer_file = checkpoint / "optimizer.pt"
                    torch.save(
                        {
                            "optimizer": optimizer.state_dict(),
                            "scheduler": scheduler.state_dict(),
                        },
                        optimizer_file,
                    )
                    state.update(
                        optimizer_file=str(optimizer_file),
                        optimizer_sha256=file_fingerprint(optimizer_file),
                    )
                state["adapter"] = str(checkpoint)
                state["rounds"].append(
                    {
                        "start_update": start,
                        "end_update": end,
                        "seconds": time.perf_counter() - started,
                        "optimizer_reset": reset,
                    }
                )
                if end == settings["updates"]:
                    if state["tokens"] != plan["tokens_by_seed"][str(seed)]:
                        raise RuntimeError("Final token budget mismatch.")
                    state["status"] = "complete"
                write_json_file(directory / "state.json", state)
                print(
                    f"{selection} replay={replay} seed={seed} updates={end}/{settings['updates']} tokens={state['tokens']} validation={observation['checkpoint']['macro_nll']:.4f}",
                    flush=True,
                )
            return state
        except BaseException as exc:
            committed = read_json_file(directory / "state.json", {})
            write_json_file(
                directory / "last_failure.json",
                {
                    "type": type(exc).__name__,
                    "message": str(exc)[:500],
                    "last_committed_update": committed.get("update", 0),
                    "scope": "Failed/uncommitted rounds do not advance canonical state; rerun the same arm to resume.",
                },
            )
            raise
        finally:
            del model
            cleanup()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "run", "report"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--candidate-bank", type=Path)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--updates", type=int, default=25)
    parser.add_argument("--microbatches", type=int, default=4)
    parser.add_argument("--warmup-updates", type=int, default=2)
    parser.add_argument("--round-updates", type=int, default=2)
    parser.add_argument("--per-category", type=int, default=8)
    parser.add_argument(
        "--optimizer-lifecycle",
        choices=("reset-per-round", "continuous"),
        default="reset-per-round",
    )
    parser.add_argument("--replay-ablation", action="store_true")
    parser.add_argument("--selection", choices=SELECTIONS)
    parser.add_argument("--replay", choices=REPLAYS, default="off")
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()
    if args.command == "prepare":
        if not args.dataset or not args.model:
            parser.error("prepare requires --dataset and --model")
        result = prepare(
            args.output,
            args.dataset,
            args.model,
            args.candidate_bank,
            seeds=args.seeds,
            updates=args.updates,
            microbatches=args.microbatches,
            warmup_updates=args.warmup_updates,
            round_updates=args.round_updates,
            replay_ablation=args.replay_ablation,
            optimizer_lifecycle=args.optimizer_lifecycle,
            per_category=args.per_category,
        )
    elif args.command == "run":
        if args.selection:
            if args.seed is None:
                parser.error("Single-arm execution requires --seed")
            state = run_arm(args.output, args.selection, args.replay, args.seed)
            result = {
                "status": state["status"],
                "updates": state["update"],
                "tokens": state["tokens"],
            }
        else:
            plan = read_json_file(args.output / "plan.json", {})
            for seed in plan["settings"]["seeds"]:
                for replay in plan["settings"]["replays"]:
                    for selection in SELECTIONS:
                        run_arm(args.output, selection, replay, seed)
            from selection_report import report

            result = report(args.output)
    else:
        from selection_report import report

        result = report(args.output)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
