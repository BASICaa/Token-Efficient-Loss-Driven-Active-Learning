"""Two-run validation-guided replay pilot with genuine replay category choices.

Preserves the existing engine on disk; a scoped, fingerprinted selector extension
adds validation-weighted replay. Preparation/reporting never load a model/API.
"""

import argparse
from collections import Counter, defaultdict
from contextlib import contextmanager
import copy
import json
import math
from pathlib import Path
import random
import shutil

import run_selection_diagnostic as engine
from TEFLD.src.helper import read_json_file, stable_json_hash, write_json_file
from TEFLD.src.runtime import file_fingerprint
from TEFLD.src.transactions import section_lock

BASE_IDENTITY = engine.code_identity
BASE_SELECTOR = engine.select_group
POLICIES = ("random", "validation")


def identity():
    return stable_json_hash(
        {
            "base_code_sha256": BASE_IDENTITY(),
            "validation_replay_sha256": file_fingerprint(Path(__file__)),
        }
    )


def balanced_schedule(records, original, warmup_updates, seed):
    schedule = copy.deepcopy(original)
    microbatches = len(schedule[0])
    warmup_size = warmup_updates * microbatches
    if (
        warmup_size < 2
        or warmup_size % 2
        or any(len(group) != microbatches for group in schedule)
    ):
        raise ValueError("Require an even common warmup and fixed microbatch count.")
    groups = defaultdict(lambda: defaultdict(list))
    for candidate, row in records.items():
        groups[row["answer_tokens"]][row["category"]].append(candidate)
    # New-example lengths remain the source's schedule outside the replay slot.
    demands = Counter(
        slot["length"] for group in schedule[warmup_updates:] for slot in group[1:]
    )
    viable = [
        length
        for length, categories in groups.items()
        if len(categories) >= 2
        and sum(map(len, categories.values())) >= demands[length] + 2
    ]
    rng = random.Random(seed)
    rng.shuffle(viable)
    viable.sort(
        key=lambda length: (
            -len(groups[length]),
            -sum(map(len, groups[length].values())),
        )
    )
    anchors = viable[: warmup_size // 2]
    if len(anchors) != warmup_size // 2:
        raise ValueError("Bank lacks enough cross-category, same-length warmup pairs.")
    warmup, used_categories = [], set()
    for length in anchors:
        categories = sorted(groups[length])
        first = rng.choice(
            [c for c in categories if c not in used_categories] or categories
        )
        second = rng.choice(
            [c for c in categories if c != first and c not in used_categories]
            or [c for c in categories if c != first]
        )
        for category in (first, second):
            candidate = rng.choice(sorted(groups[length][category]))
            warmup.append(
                {"length": length, "warmup_id": candidate, "replay_slot": False}
            )
            used_categories.add(category)
    for update in range(warmup_updates):
        schedule[update] = warmup[update * microbatches : (update + 1) * microbatches]
    for update in range(warmup_updates, len(schedule)):
        schedule[update][0] = {
            "length": anchors[(update - warmup_updates) % len(anchors)],
            "warmup_id": None,
            "replay_slot": True,
        }
        for slot in schedule[update][1:]:
            slot.update(warmup_id=None, replay_slot=False)
    # Every replay length has two different categories from the common warmup.
    return schedule


def select_group(
    records, slots, seen, selection, replay, weakness, replay_losses, seed
):
    if selection != "validation" or replay not in POLICIES:
        raise ValueError(
            "This pilot requires validation selection and random/validation replay."
        )
    # Consume the old random-replay RNG stream for new-example selection in both
    # arms. Extra category draws use an independent stream, avoiding incidental
    # changes to new-example choices merely from replay RNG consumption.
    selected, traces = BASE_SELECTOR(
        records, slots, seen, selection, "random", weakness, replay_losses, seed
    )
    for index, (slot, trace) in enumerate(zip(slots, traces)):
        if not trace["replay"]:
            continue
        options = [
            i for i in sorted(seen) if records[i]["answer_tokens"] == slot["length"]
        ]
        categories = defaultdict(list)
        for candidate in options:
            categories[records[candidate]["category"]].append(candidate)
        if len(categories) < 2:
            raise ValueError(
                "Replay slot lacks a real category choice; no silent forced-choice test."
            )
        weights = {}
        for category in sorted(categories):
            value = weakness.get(category)
            if type(value) not in (float, int) or not math.isfinite(value) or value < 0:
                raise ValueError(
                    "Replay requires finite curriculum-validation category losses."
                )
            weights[category] = max(1e-5, value)
        if replay == "validation":
            rng = random.Random(f"validation-replay-v1:{seed}:{index}")
            category = rng.choices(list(weights), weights=list(weights.values()), k=1)[
                0
            ]
            selected[index] = rng.choice(sorted(categories[category]))
        candidate = selected[index]
        category = records[candidate]["category"]
        probability = (
            weights[category] / sum(weights.values())
            if replay == "validation"
            else len(categories[category]) / len(options)
        )
        trace.update(
            candidate_id=candidate,
            category=category,
            output_origin=records[candidate]["origin"],
            replay_training_loss=replay_losses.get(candidate),
            replay_policy=replay,
            replay_feedback_source=(
                "curriculum_validation_category_nll"
                if replay == "validation"
                else "uniform_candidate_sampling"
            ),
            replay_category_weights=weights if replay == "validation" else None,
            selected_replay_category_probability=probability,
            replay_category_candidate_counts={c: len(v) for c, v in categories.items()},
        )
    return selected, traces


@contextmanager
def extension():
    old_identity, old_selector = engine.code_identity, engine.select_group
    engine.code_identity, engine.select_group = identity, select_group
    try:
        yield
    finally:
        engine.code_identity, engine.select_group = old_identity, old_selector


def prepare(source, output, seed=42):
    source, output = Path(source).resolve(), Path(output).resolve()
    if (
        source == output
        or source in output.parents
        or output in source.parents
        or output.exists()
    ):
        raise ValueError("Use a fresh separate output directory.")
    with section_lock(source):
        plan = read_json_file(source / "plan.json", {})
        if plan["code_sha256"] != BASE_IDENTITY():
            raise ValueError("Source engine identity changed.")
        if (
            seed not in plan["settings"]["seeds"]
            or plan["settings"]["optimizer_lifecycle"] != "reset-per-round"
        ):
            raise ValueError("Require a prepared reset-per-round source for this seed.")
        for filename, field in (
            ("protocol.json", "protocol_sha256"),
            ("bank.json", "bank_sha256"),
        ):
            if file_fingerprint(source / filename) != plan[field]:
                raise ValueError("Source bank or protocol changed.")
        records = read_json_file(source / "bank.json", {})
        schedule = balanced_schedule(
            records,
            plan["schedules"][str(seed)],
            plan["settings"]["warmup_updates"],
            seed,
        )
        new = copy.deepcopy(plan)
        new.pop("reuse_controls", None)
        new["settings"].update(seeds=[seed], replays=list(POLICIES))
        new["schedules"] = {str(seed): schedule}
        new["tokens_by_seed"] = {
            str(seed): sum(slot["length"] for group in schedule for slot in group)
        }
        new["code_sha256"] = identity()
        new["validation_replay"] = {
            "seed": seed,
            "selection": "validation",
            "new_runs": 2,
            "base_code_sha256": BASE_IDENTITY(),
            "source_plan_sha256": file_fingerprint(source / "plan.json"),
            "scope": "Fresh two-run development pilot: validation-guided new data; compare uniform-candidate replay with validation-category-weighted replay, random within category.",
            "reason_for_new_controls": "Old replay slots were mostly single-candidate under exact-length constraints. New cross-category warmup changes the schedule; old scores are contextual only, not reused controls.",
            "feedback": "Curriculum category mean NLL only. No heldout examples enter training; no checkpoint-validation signal enters selection.",
            "sampling": "Probability proportional to positive category NLL among available same-length seen training examples; then uniform within chosen category.",
            "rng": "New-example RNG consumes identical random-replay draws; validation replay uses an independent seeded stream.",
        }
        output.mkdir(parents=True)
        for filename in ("protocol.json", "bank.json"):
            shutil.copy2(source / filename, output / filename)
        write_json_file(output / "plan.json", new)
        return {
            "status": "prepared",
            "new_runs": 2,
            "seed": seed,
            "updates_per_run": new["settings"]["updates"],
            "tokens_per_run": new["tokens_by_seed"][str(seed)],
            "replay_slots_per_run": len(schedule) - new["settings"]["warmup_updates"],
            "minimum_replay_categories": 2,
            "provider_calls": 0,
            "training_started": False,
        }


def run(output, policy):
    output = Path(output).resolve()
    plan = read_json_file(output / "plan.json", {})
    if plan["code_sha256"] != identity() or plan["validation_replay"]["new_runs"] != 2:
        raise ValueError("Validation replay identity changed.")
    with extension():
        state = engine.run_arm(
            output, "validation", policy, plan["validation_replay"]["seed"]
        )
    return {
        "status": state["status"],
        "policy": policy,
        "updates": state["update"],
        "tokens": state["tokens"],
    }


def report(output, *, figures=True):
    output = Path(output).resolve()
    plan = read_json_file(output / "plan.json", {})
    if plan["code_sha256"] != identity():
        raise ValueError("Validation replay identity changed.")
    seed = plan["validation_replay"]["seed"]
    rows, states = [], []
    for policy in POLICIES:
        path = output / f"seed_{seed}/validation__replay_{policy}/state.json"
        if not path.exists():
            continue
        state = read_json_file(path, {})
        expected = {
            "plan_sha256": file_fingerprint(output / "plan.json"),
            "selection": "validation",
            "replay": policy,
            "seed": seed,
        }
        if state["identity"] != expected:
            raise ValueError("Run identity differs from plan.")
        if state["status"] == "complete" and (
            state["tokens"] != plan["tokens_by_seed"][str(seed)]
            or state["update"] != plan["settings"]["updates"]
        ):
            raise ValueError("Completed budgets differ from plan.")
        replays = [x for x in state["selection_trace"] if x["replay"]]
        if any(x["eligible_categories"] < 2 for x in replays):
            raise ValueError("Replay policy had no meaningful category choice.")
        states.append(state)
        rows.append(
            {
                "policy": policy,
                "status": state["status"],
                "tokens": state["tokens"],
                "updates": state["update"],
                "unique_examples": len(state["seen"]),
                "replay_presentations": len(replays),
                "replay_categories": dict(Counter(x["category"] for x in replays)),
                "minimum_eligible_replay_categories": min(
                    (x["eligible_categories"] for x in replays), default=None
                ),
                "macro_nll": state["evaluations"][-1]["checkpoint"]["macro_nll"],
                "token_weighted_nll": state["evaluations"][-1]["checkpoint"][
                    "token_weighted_nll"
                ],
            }
        )
    if len(states) == 2:
        if states[0]["evaluations"][0] != states[1]["evaluations"][0]:
            raise ValueError("Initial evaluations differ.")
        n = plan["settings"]["warmup_updates"]
        warmups = [
            [x["candidate_id"] for x in s["selection_trace"] if x["update"] <= n]
            for s in states
        ]
        if all(s["update"] >= n for s in states) and warmups[0] != warmups[1]:
            raise ValueError("Warmup differs.")
    complete = len(rows) == 2 and all(x["status"] == "complete" for x in rows)
    result = {
        "scope": plan["validation_replay"]["scope"],
        "results": rows,
        "remaining_runs": 2 - sum(x["status"] == "complete" for x in rows),
        "delta_nll_validation_minus_random": (
            rows[1]["macro_nll"] - rows[0]["macro_nll"] if complete else None
        ),
        "old_scores_directly_comparable": False,
        "final_test_scored": False,
        "provider_calls": 0,
    }
    write_json_file(output / "validation_replay_comparison.json", result)
    lines = [
        "# Validation-guided replay pilot",
        "",
        result["scope"],
        "",
        "Fresh balanced warmup: old replay results are not direct controls. Lower NLL is better; final fixed-budget scores only.",
        "",
        "| Replay | Status | Updates | Tokens | Unique | Macro NLL | Minimum available replay categories |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    lines += [
        f"| {r['policy']} | {r['status']} | {r['updates']} | {r['tokens']} | {r['unique_examples']} | {r['macro_nll']:.6f} | {r['minimum_eligible_replay_categories']} |"
        for r in rows
    ]
    lines += [
        "",
        "One seed, exact-length-constrained replay; guidance uses curriculum category losses and may change later training feedback. No early stopping or best-checkpoint selection. No final holdout scored.",
    ]
    (output / "VALIDATION_REPLAY_REPORT.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    if figures and states:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        figure, axes = plt.subplots(1, 2, figsize=(12, 4))
        for row, state in zip(rows, states):
            axes[0].plot(
                [x["tokens"] for x in state["evaluations"]],
                [x["checkpoint"]["macro_nll"] for x in state["evaluations"]],
                label=row["policy"],
                marker=".",
            )
            counts = row["replay_categories"]
            axes[1].plot(
                sorted(counts),
                [counts[c] for c in sorted(counts)],
                marker=".",
                label=row["policy"],
            )
        axes[0].set(
            xlabel="Answer-token exposures",
            ylabel="Checkpoint macro NLL",
            title="Final-budget replay comparison",
        )
        axes[1].set(ylabel="Replay presentations", title="Replayed category allocation")
        axes[1].tick_params(axis="x", rotation=50)
        axes[0].legend()
        figure.tight_layout()
        figure.savefig(output / "validation_replay_comparison.png", dpi=150)
        plt.close(figure)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "run", "report"))
    parser.add_argument("--source", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--policy", choices=POLICIES)
    args = parser.parse_args()
    if args.command == "prepare":
        if not args.source:
            parser.error("prepare requires --source")
        result = prepare(args.source, args.output, args.seed)
    elif args.command == "run":
        if not args.policy:
            parser.error("run requires --policy")
        result = run(args.output, args.policy)
    else:
        result = report(args.output)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
