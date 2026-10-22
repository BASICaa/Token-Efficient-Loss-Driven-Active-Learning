"""Offline fixed-curriculum optimizer diagnostic; no teacher/API client is used.

This measures the combined update-size, optimizer-lifecycle and LR recipe.
The saved adaptive curricula are held fixed rather than generated again.
"""

from __future__ import annotations

import math
from pathlib import Path
import random
import time

from .dataschema import Training_Sample
from .dataset_curriculum import DATASET_METHODS
from .encoding import encode_sample, supervised_tokens
from .experiment_runner import save_adapter, require_adapter
from .helper import read_json_file, write_json_file, stable_json_hash
from .runtime import load_runtime, cleanup, set_seed, file_fingerprint
from .token_budget import optimizer_group, limit_targets
from .transactions import section_lock
from .validation_data import SplitData, score_samples

RECIPES = ("original_four_examples", "fixed_250_tokens")
METHODS = ("tefld_validation", "tefld_loss", "dolly_baseline")
SUPPORTED_METHODS = METHODS + tuple(DATASET_METHODS)


def target_slice(encoded, start, count):
    """Retain exactly a contiguous range of shifted target positions."""
    total = supervised_tokens(encoded)
    if (
        type(start) is not int
        or type(count) is not int
        or start < 0
        or count < 1
        or start + count > total
    ):
        raise ValueError("Invalid target-position slice.")
    result = {key: value.clone() for key, value in encoded.items()}
    positions = (result["labels"][1:] != -100).nonzero().flatten() + 1
    result["labels"].fill_(-100)
    result["labels"][positions[start : start + count]] = encoded["labels"][
        positions[start : start + count]
    ]
    assert supervised_tokens(result) == count
    return result


def build_groups(records, recipe, targets_per_update=250):
    if recipe not in RECIPES:
        raise ValueError("Unknown optimizer diagnostic recipe.")
    if type(targets_per_update) is not int or targets_per_update < 1:
        raise ValueError("Invalid target batch size.")
    groups = []
    parts = []
    size = 0
    previous = None
    for index, row in enumerate(records):
        count = row["answer_tokens"]
        if type(count) is not int or count < 1:
            raise ValueError("Saved presentation has no target supervision.")
        if recipe == RECIPES[0]:
            key = (row["round"], row["tokens_after_group"])
            if previous is not None and key != previous:
                groups.append({"parts": parts, "targets": size, "round": previous[0]})
                parts = []
                size = 0
            parts.append({"record": index, "start": 0, "count": count})
            size += count
            previous = key
        else:
            offset = 0
            while offset < count:
                take = min(count - offset, targets_per_update - size)
                parts.append({"record": index, "start": offset, "count": take})
                size += take
                offset += take
                if size == targets_per_update:
                    groups.append({"parts": parts, "targets": size, "round": None})
                    parts = []
                    size = 0
    if parts:
        groups.append(
            {
                "parts": parts,
                "targets": size,
                "round": previous[0] if recipe == RECIPES[0] else None,
            }
        )
    if sum(g["targets"] for g in groups) != sum(r["answer_tokens"] for r in records):
        raise ValueError("Token coverage mismatch.")
    return groups


def encode_saved_stream(source, tokenizer, max_length=512):
    source = Path(source)
    saved = read_json_file(source / "state.json", {})
    if saved.get("status") != "complete":
        raise ValueError("Use a completed saved pilot curriculum.")
    records = saved["losses"]
    encoded = []
    batches = {}
    hashes = {"state.json": file_fingerprint(source / "state.json")}
    for row in records:
        round_id = row["round"]
        if round_id not in batches:
            path = source / f"batch_{round_id:04d}.json"
            batches[round_id] = [
                Training_Sample.model_validate(x) for x in read_json_file(path, [])
            ]
            hashes[path.name] = file_fingerprint(path)
        sample = batches[round_id][row["batch_index"]]
        from .candidate_pool import candidate_id

        if candidate_id(sample) != row["candidate_id"]:
            raise ValueError("Saved presentation no longer matches the source batch.")
        if row["answer_tokens"] > supervised_tokens(
            encode_sample(sample, tokenizer, max_length)
        ):
            raise ValueError("Saved target count exceeds complete encoded answer.")
        encoded.append(
            limit_targets(
                encode_sample(sample, tokenizer, max_length), row["answer_tokens"]
            )
        )
    if sum(supervised_tokens(x) for x in encoded) != saved["tokens"]:
        raise ValueError("Reconstructed stream differs from saved budget.")
    round_steps = {
        round_id: math.ceil(len(batch) * 3 / 4) for round_id, batch in batches.items()
    }
    return saved, records, encoded, round_steps, hashes


def verify_base_model(source, model_path, expected):
    """Reuse the prior content fingerprint only if every file stat still matches."""
    path = Path(model_path)
    if not path.is_dir():
        raise FileNotFoundError("The local base model is unavailable.")
    files = sorted(
        p
        for p in path.iterdir()
        if p.suffix == ".safetensors"
        or p.name
        in {
            "config.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "chat_template.jinja",
            "generation_config.json",
            "model.safetensors.index.json",
        }
    )
    stats = {
        p.name: {"bytes": p.stat().st_size, "mtime_ns": p.stat().st_mtime_ns}
        for p in files
    }
    cache = read_json_file(Path(source).parents[1] / "model_fingerprint.json", {})
    if cache.get("identity_sha256") == expected and cache.get("file_stats") == stats:
        return
    actual = stable_json_hash({"files": {p.name: file_fingerprint(p) for p in files}})
    if actual != expected:
        raise ValueError("The base model differs from the completed pilot.")


def capture_rng():
    import numpy as np
    import torch

    numpy_state = np.random.get_state()
    return {
        "python": random.getstate(),
        "numpy": [
            numpy_state[0],
            numpy_state[1].tolist(),
            numpy_state[2],
            numpy_state[3],
            numpy_state[4],
        ],
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def restore_rng(state):
    import numpy as np
    import torch

    random.setstate(state["python"])
    value = state["numpy"]
    np.random.set_state(
        (value[0], np.array(value[1], dtype="uint32"), value[2], value[3], value[4])
    )
    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"]:
        torch.cuda.set_rng_state_all([x.cpu() for x in state["cuda"]])


def execute_groups(
    model,
    encoded,
    records,
    groups,
    recipe,
    round_steps,
    *,
    seed,
    state,
    optimizer=None,
    after_step=None,
):
    import torch

    params = [p for p in model.parameters() if p.requires_grad]
    total = sum(r["answer_tokens"] for r in records)
    if state["next_group"] == 0:
        set_seed(seed)
    for index in range(state["next_group"], len(groups)):
        group = groups[index]
        if optimizer is None or (
            recipe == RECIPES[0] and state.get("active_round") != group["round"]
        ):
            optimizer = torch.optim.AdamW(
                params, lr=0.0002, betas=(0.9, 0.999), eps=1e-8, weight_decay=0
            )
            state["round_updates"] = 0
            if recipe == RECIPES[0]:
                set_seed(seed + group["round"])
        state["active_round"] = group["round"]
        lr = (
            0.0002 * (1 - state["tokens"] / total)
            if recipe == RECIPES[1]
            else 0.0002
            * max(0.0, 1 - state["round_updates"] / round_steps[group["round"]])
        )
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr
        batch = [
            target_slice(encoded[x["record"]], x["start"], x["count"])
            for x in group["parts"]
        ]
        model.train()
        started = time.perf_counter()
        count, losses = optimizer_group(model, optimizer, batch)
        state["training_seconds"] = (
            state.get("training_seconds", 0) + time.perf_counter() - started
        )
        state["tokens"] += count
        state["next_group"] = index + 1
        state["round_updates"] += 1
        state.setdefault("updates", []).append(
            {
                "step": index + 1,
                "tokens": state["tokens"],
                "lr": lr,
                "target_tokens": count,
                "forward_parts": len(batch),
                "source_records": [x["record"] for x in group["parts"]],
                "mean_target_weighted_loss": sum(
                    loss * supervised_tokens(item) for loss, item in zip(losses, batch)
                )
                / count,
            }
        )
        if after_step:
            after_step(state, optimizer)
    return optimizer


def save_training_checkpoint(model, tokenizer, optimizer, state, directory, identity):
    import torch

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    rng = capture_rng()
    save_adapter(
        model,
        tokenizer,
        directory,
        {"identity": identity, "tokens": state["tokens"], "step": state["next_group"]},
    )
    # A completed state pointer is written only after both weights and optimizer/RNG files exist.
    temporary = directory / "training_state.tmp"
    torch.save(
        {
            "optimizer": optimizer.state_dict() if optimizer else None,
            "rng": rng,
            "state": state,
        },
        temporary,
    )
    temporary.replace(directory / "training_state.pt")
    marker = read_json_file(directory / "completion.json", {})
    marker["training_state_sha256"] = file_fingerprint(directory / "training_state.pt")
    write_json_file(directory / "completion.json", marker)
    restore_rng(rng)
    return str(directory.resolve())


def load_training_checkpoint(path):
    import torch

    metadata = require_adapter(path)
    file = Path(path) / "training_state.pt"
    if metadata.get("training_state_sha256") != file_fingerprint(file):
        raise ValueError("Optimizer/RNG checkpoint is incomplete or changed.")
    return torch.load(file, map_location="cpu", weights_only=True)


def run_condition(
    source,
    output,
    protocol,
    recipe,
    *,
    milestones=(1000, 2500, 5000),
    per_category=8,
    model_path=None,
):
    import torch

    data = SplitData(protocol)
    source = Path(source).resolve()
    output = Path(output).resolve()
    saved = read_json_file(source / "state.json", {})
    seed = saved["config"]["seed"]
    total = saved["tokens"]
    model_path = model_path or saved["config"]["model_path"]
    if (
        recipe not in RECIPES
        or sorted(set(milestones)) != list(milestones)
        or any(x < 1 or x > total for x in milestones)
        or not milestones
        or milestones[-1] != total
    ):
        raise ValueError("Invalid recipe or milestones.")
    with section_lock(output):
        state_path = output / "state.json"
        previous = read_json_file(state_path, {})
        code = stable_json_hash(
            {
                p.name: file_fingerprint(p)
                for p in sorted(Path(__file__).resolve().parent.glob("*.py"))
            }
        )
        source_hashes = {
            "state.json": file_fingerprint(source / "state.json"),
            **{
                p.name: file_fingerprint(p) for p in sorted(source.glob("batch_*.json"))
            },
        }
        identity = stable_json_hash(
            {
                "source": source_hashes,
                "recipe": recipe,
                "milestones": list(milestones),
                "validation_per_category": per_category,
                "protocol_sha256": file_fingerprint(protocol),
                "code_sha256": code,
                "model_path": str(model_path),
                "base_identity": saved["config"]["model_sha256"],
            }
        )
        if previous and previous.get("identity") != identity:
            raise ValueError(
                "Diagnostic identity changed; use a fresh output directory."
            )
        if previous.get("status") == "complete":
            return previous
        verify_base_model(source, model_path, saved["config"]["model_sha256"])
        checkpoint = previous.get("checkpoint")
        payload = load_training_checkpoint(checkpoint) if checkpoint else None
        if checkpoint and require_adapter(checkpoint).get("identity") != identity:
            raise ValueError("Diagnostic checkpoint identity changed.")
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        model, tokenizer = load_runtime(model_path, seed=seed, adapter=checkpoint)
        state = (
            payload["state"]
            if payload
            else {
                "identity": identity,
                "recipe": recipe,
                "source_method": saved["config"]["method"],
                "source_hashes": source_hashes,
                "seed": seed,
                "status": "running",
                "tokens": 0,
                "next_group": 0,
                "updates": [],
                "evaluations": [],
                "training_seconds": 0.0,
                "evaluation_seconds": 0.0,
                "scope": "Offline fixed saved curriculum; combined optimization-recipe sensitivity, not adaptive regeneration.",
            }
        )
        optimizer = None
        try:
            _, records, encoded, round_steps, _ = encode_saved_stream(
                source, tokenizer, data.protocol["common_training"]["max_length"]
            )
            groups = build_groups(records, recipe)
            evaluation_samples = data.samples("checkpoint_validation", per_category)

            def measure(nominal):
                started = time.perf_counter()
                result = score_samples(model, tokenizer, evaluation_samples)
                state["evaluation_seconds"] += time.perf_counter() - started
                state["evaluations"].append(
                    {
                        "nominal_tokens": nominal,
                        "actual_tokens": state["tokens"],
                        "step": state["next_group"],
                        "evaluation": result,
                    }
                )
                print(
                    f"{saved['config']['method']} {recipe} nominal={nominal} actual={state['tokens']} updates={state['next_group']} nll={result['macro_nll']:.6f}",
                    flush=True,
                )

            if payload:
                if payload["optimizer"]:
                    optimizer = torch.optim.AdamW(
                        [p for p in model.parameters() if p.requires_grad],
                        lr=0.0002,
                        betas=(0.9, 0.999),
                        eps=1e-8,
                        weight_decay=0,
                    )
                    optimizer.load_state_dict(payload["optimizer"])
                restore_rng(payload["rng"])
            else:
                measure(0)

            def after_step(current, current_optimizer):
                done = {row["nominal_tokens"] for row in current["evaluations"]}
                reached = [
                    m for m in milestones if m <= current["tokens"] and m not in done
                ]
                if reached:
                    for nominal in reached:
                        measure(nominal)
                    path = output / f"checkpoint_step_{current['next_group']:04d}"
                    current["peak_gpu_bytes"] = (
                        int(torch.cuda.max_memory_allocated())
                        if torch.cuda.is_available()
                        else None
                    )
                    saved_path = save_training_checkpoint(
                        model, tokenizer, current_optimizer, current, path, identity
                    )
                    write_json_file(state_path, {**current, "checkpoint": saved_path})

            optimizer = execute_groups(
                model,
                encoded,
                records,
                groups,
                recipe,
                round_steps,
                seed=seed,
                state=state,
                optimizer=optimizer,
                after_step=after_step,
            )
            if state["tokens"] != total:
                raise RuntimeError("Diagnostic failed to match total target exposure.")
            final = read_json_file(state_path, {})
            state.update(
                status="complete",
                checkpoint=final["checkpoint"],
                optimizer_updates=len(groups),
                forward_parts=sum(len(g["parts"]) for g in groups),
                provider_calls=0,
                final_test_scored=False,
            )
            write_json_file(state_path, state)
            return state
        finally:
            model = tokenizer = optimizer = None
            cleanup()


def summarize_diagnostic(output, methods=METHODS):
    output = Path(output)
    if (
        not methods
        or len(set(methods)) != len(methods)
        or any(m not in SUPPORTED_METHODS for m in methods)
    ):
        raise ValueError("Declare unique supported diagnostic methods.")
    rows = {}
    for method in methods:
        rows[method] = {}
        for recipe in RECIPES:
            state = read_json_file(output / method / recipe / "state.json", {})
            if state.get("status") != "complete":
                raise ValueError(
                    "Complete both diagnostic recipes for every declared method before reporting."
                )
            final = state["evaluations"][-1]
            rows[method][recipe] = {
                "macro_nll": final["evaluation"]["macro_nll"],
                "actual_tokens": final["actual_tokens"],
                "updates": state["optimizer_updates"],
                "forward_parts": state["forward_parts"],
                "training_seconds": state["training_seconds"],
                "evaluation_seconds": state["evaluation_seconds"],
                "curve": [
                    {
                        "nominal": x["nominal_tokens"],
                        "actual": x["actual_tokens"],
                        "steps": x["step"],
                        "macro_nll": x["evaluation"]["macro_nll"],
                    }
                    for x in state["evaluations"]
                ],
            }
    report = {
        "status": "completed_offline_optimizer_diagnostic",
        "results": rows,
        "provider_calls": 0,
        "final_test_scored": False,
        "interpretation": "Compare combined optimization recipes on fixed saved streams. Only final 5000-token exposures match exactly across all conditions; original-recipe intermediate evaluations occur at its first complete update after a nominal milestone. One seed; no new adaptive curriculum or answer-quality verdict.",
    }
    write_json_file(output / "diagnostic_report.json", report)
    return report
