"""Numeric summaries and static scientific figures; never load a model or API."""

from collections import Counter, defaultdict
from pathlib import Path
import statistics

from TEFLD.src.helper import read_json_file, write_json_file


def _figures(output, rows):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not any(row.get("curve") for row in rows):
        figure, axes = plt.subplots(1, 2, figsize=(12, 5))
        labels = [x["arm"] for x in rows]
        axes[0].barh(labels, [x["checkpoint_nll"] for x in rows])
        axes[0].set(
            xlabel="Final checkpoint macro NLL (lower is better)",
            title="Existing validation scores",
        )
        axes[1].barh(labels, [x["tokens"] for x in rows])
        axes[1].set(xlabel="Answer-token exposures", title="Existing training exposure")
        figure.tight_layout()
        figure.savefig(output / "validation_and_tokens.png", dpi=150)
        figure.savefig(output / "validation_and_tokens.svg")
        plt.close(figure)
        return
    figure, axes = plt.subplots(2, 2, figsize=(13, 9))
    for row in rows:
        label = row["arm"] + f" / {row.get('seed', 'historical')}"
        evaluations = row.get("curve", [])
        if evaluations:
            axes[0, 0].plot(
                [x["tokens"] for x in evaluations],
                [x["checkpoint_nll"] for x in evaluations],
                marker=".",
                label=label,
            )
            axes[0, 1].plot(
                [x["update"] for x in evaluations],
                [x["checkpoint_nll"] for x in evaluations],
                marker=".",
                label=label,
            )
            axes[1, 0].plot(
                [x["tokens"] for x in evaluations],
                [x["warmup_probe_nll"] for x in evaluations],
                marker=".",
                label=label,
            )
    axes[0, 0].set(
        xlabel="Answer-token exposures",
        ylabel="Checkpoint validation macro NLL",
        title="Validation vs training tokens",
    )
    axes[0, 1].set(
        xlabel="Optimizer updates",
        ylabel="Checkpoint validation macro NLL",
        title="Validation vs optimizer updates",
    )
    axes[1, 0].set(
        xlabel="Answer-token exposures",
        ylabel="Shared warmup-probe macro NLL",
        title="Fixed exposed training examples",
    )
    labels = [x["arm"] + f" / {x.get('seed', 'historical')}" for x in rows]
    axes[1, 1].barh(labels, [x["tokens"] for x in rows])
    axes[1, 1].set(xlabel="Answer-token exposures", title="Actual training exposure")
    for axis in axes.flat:
        axis.grid(alpha=0.2)
    if any(row.get("curve") for row in rows):
        axes[0, 0].legend(fontsize=6)
    figure.tight_layout()
    figure.savefig(output / "learning_curves.png", dpi=150)
    figure.savefig(output / "learning_curves.svg")
    plt.close(figure)
    if rows:
        figure, axis = plt.subplots(figsize=(10, max(4, len(rows) * 0.35)))
        axis.barh(labels, [x["checkpoint_nll"] for x in rows])
        axis.set(
            xlabel="Checkpoint validation macro NLL (lower is better)",
            title="Fixed-budget final scores; incomplete arms use latest score",
        )
        axis.invert_yaxis()
        figure.tight_layout()
        figure.savefig(output / "validation_comparison.png", dpi=150)
        plt.close(figure)


def summarize_state(state, plan):
    evaluations = state["evaluations"]
    final = evaluations[-1]
    traces = state["selection_trace"]
    updates = state["updates"]
    categories = defaultdict(
        lambda: {"new_tokens": 0, "replay_tokens": 0, "presentations": 0}
    )
    for trace in traces:
        counts = categories[trace["category"]]
        counts["replay_tokens" if trace["replay"] else "new_tokens"] += trace[
            "answer_tokens"
        ]
        counts["presentations"] += 1
    choices = [
        x
        for x in traces
        if x["update"] > plan["settings"]["warmup_updates"] and not x["replay"]
    ]
    return {
        "arm": state["identity"]["selection"]
        + "__replay_"
        + state["identity"]["replay"],
        "seed": state["identity"]["seed"],
        "status": state["status"],
        "updates": state["update"],
        "tokens": state["tokens"],
        "unique_examples": len(state["seen"]),
        "checkpoint_nll": final["checkpoint"]["macro_nll"],
        "starting_checkpoint_nll": evaluations[0]["checkpoint"]["macro_nll"],
        "improvement_nll": evaluations[0]["checkpoint"]["macro_nll"]
        - final["checkpoint"]["macro_nll"],
        "category_exposures": dict(categories),
        "replay_presentations": sum(x["replay"] for x in traces),
        "clipped_fraction": (
            sum(x["clipped"] for x in updates) / len(updates) if updates else None
        ),
        "optimizer_resets": sum(x["optimizer_reset"] for x in updates),
        "mean_eligible_candidates": (
            statistics.mean(x["eligible_candidates"] for x in choices)
            if choices
            else None
        ),
        "single_choice_fraction": (
            statistics.mean(x["eligible_candidates"] == 1 for x in choices)
            if choices
            else None
        ),
        "category_choice_fraction": (
            statistics.mean(x["eligible_categories"] > 1 for x in choices)
            if choices
            else None
        ),
        "curve": [
            {
                "update": x["update"],
                "tokens": x["tokens"],
                "checkpoint_nll": x["checkpoint"]["macro_nll"],
                "training_probe_nll": x["training_probe"]["macro_nll"],
                "warmup_probe_nll": x["warmup_probe"]["macro_nll"],
                "curriculum_nll": x["curriculum"]["macro_nll"],
            }
            for x in evaluations
        ],
    }


def report(output, *, figures=True):
    output = Path(output)
    plan = read_json_file(output / "plan.json", {})
    rows = []
    states = {}
    for seed in plan["settings"]["seeds"]:
        for replay in plan["settings"]["replays"]:
            for selection in ("random", "loss", "validation"):
                name = f"{selection}__replay_{replay}"
                path = output / f"seed_{seed}" / name / "state.json"
                if not path.exists():
                    continue
                state = read_json_file(path, {})
                row = summarize_state(state, plan)
                rows.append(row)
                states[(seed, name)] = state
    integrity = []
    paired = []
    for seed in plan["settings"]["seeds"]:
        members = [(name, state) for (s, name), state in states.items() if s == seed]
        complete = [
            (name, state) for name, state in members if state["status"] == "complete"
        ]
        for name, state in complete:
            if (
                state["update"] != plan["settings"]["updates"]
                or state["tokens"] != plan["tokens_by_seed"][str(seed)]
            ):
                raise ValueError("A completed run violates matched resources.")
        warmups = [
            [
                x["candidate_id"]
                for x in state["selection_trace"]
                if x["update"] <= plan["settings"]["warmup_updates"]
            ]
            for _, state in members
            if state["update"] >= plan["settings"]["warmup_updates"]
        ]
        if warmups and any(x != warmups[0] for x in warmups):
            raise ValueError("Initial training batches differ across arms.")
        integrity.append(
            {
                "seed": seed,
                "complete_arms": len(complete),
                "planned_arms": 3 * len(plan["settings"]["replays"]),
                "common_warmup": bool(warmups),
                "expected_tokens": plan["tokens_by_seed"][str(seed)],
            }
        )
        baseline = next(
            (
                r
                for r in rows
                if r["seed"] == seed
                and r["arm"] == "random__replay_off"
                and r["status"] == "complete"
            ),
            None,
        )
        if baseline:
            for row in rows:
                if row["seed"] == seed and row["status"] == "complete":
                    paired.append(
                        {
                            "seed": seed,
                            "arm": row["arm"],
                            "delta_nll_vs_random_off": row["checkpoint_nll"]
                            - baseline["checkpoint_nll"],
                        }
                    )
    aggregates = []
    for arm in sorted({x["arm"] for x in paired}):
        values = [x["delta_nll_vs_random_off"] for x in paired if x["arm"] == arm]
        aggregates.append(
            {
                "arm": arm,
                "paired_seeds": len(values),
                "mean_delta_nll": statistics.mean(values),
                "sample_std_delta_nll": (
                    statistics.stdev(values) if len(values) > 1 else None
                ),
            }
        )
    comparisons = []
    # Separate selection and replay effects; negative delta means lower loss.
    for seed in plan["settings"]["seeds"]:
        scores = {
            x["arm"]: x["checkpoint_nll"]
            for x in rows
            if x["seed"] == seed and x["status"] == "complete"
        }
        for selection in ("random", "loss", "validation"):
            for left, right in (("off", "random"), ("random", "guided")):
                a, b = f"{selection}__replay_{left}", f"{selection}__replay_{right}"
                if a in scores and b in scores:
                    comparisons.append(
                        {
                            "seed": seed,
                            "selection": selection,
                            "contrast": f"{right}_minus_{left}",
                            "delta_nll": scores[b] - scores[a],
                        }
                    )
    result = {
        "scope": plan["scope"],
        "checkpoint_rule": plan["checkpoint_rule"],
        "results": rows,
        "resource_checks": integrity,
        "paired_vs_random": paired,
        "paired_seed_summary": aggregates,
        "replay_contrasts": comparisons,
        "new_provider_calls": 0,
        "final_test_scored": False,
        "interpretation_limits": [
            "Selection is conditional on exact answer length; inspect eligible-choice counts.",
            "This is a new one-pass diagnostic recipe, not the historical three-epoch 100-example experiment.",
            "Replay is a separate history-replay ablation, not the historical threshold/cooldown rule.",
            "Checkpoint curves are development diagnostics; only final fixed-budget checkpoints are compared.",
        ],
    }
    write_json_file(output / "diagnostic_summary.json", result)
    lines = [
        "# Controlled selection diagnostic",
        "",
        plan["scope"],
        "",
        "Lower NLL is better. Final fixed-budget scores; incomplete arms are explicitly labeled.",
        "",
        "| Arm | Seed | Status | Tokens | Updates | Unique | Validation NLL | Improvement from start |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        lines.append(
            f"| {row['arm']} | {row['seed']} | {row['status']} | {row['tokens']} | {row['updates']} | {row['unique_examples']} | {row['checkpoint_nll']:.5f} | {row['improvement_nll']:.5f} |"
        )
    lines += [
        "",
        "## Interpretation limits",
        "",
        *["- " + text for text in result["interpretation_limits"]],
        "",
        "Inspect diagnostic_summary.json for category/replay exposure, clipping frequency, optimizer resets, choice constraints and paired seed/replay contrasts.",
        "A falling training curve with a rising validation curve is a diagnostic clue, not a causal finding. No automatic stopping or rollback is enabled.",
    ]
    (output / "DIAGNOSTIC_REPORT.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    if figures and rows:
        _figures(output, rows)
        _health_figures(output, states)
    return {
        "available_arms": len(rows),
        "complete_arms": sum(x["status"] == "complete" for x in rows),
        "summary": str(output / "diagnostic_summary.json"),
        "report": str(output / "DIAGNOSTIC_REPORT.md"),
    }


def _health_figures(output, states):
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    for (seed, name), state in states.items():
        updates = state["updates"]
        axes[0].plot(
            [x["update"] for x in updates],
            [x["learning_rate"] for x in updates],
            label=f"{name}/{seed}",
        )
        axes[1].plot(
            [x["update"] for x in updates],
            [x["gradient_norm_before_clipping"] for x in updates],
        )
    axes[0].set(ylabel="Learning rate", title="Optimizer lifecycle and gradient health")
    axes[1].set(xlabel="Optimizer update", ylabel="Gradient norm before clipping")
    axes[1].axhline(1.0, color="black", linestyle="--", label="Clipping threshold")
    axes[0].legend(fontsize=6)
    figure.tight_layout()
    figure.savefig(output / "optimizer_health.png", dpi=150)
    plt.close(figure)


def historical_report(comparison, output, *, figures=True):
    """Summarize existing measurements without recomputing or inventing probes."""
    comparison, output = Path(comparison), Path(output)
    if output.exists():
        raise ValueError("Use a fresh report directory.")
    rows = []
    import json

    # Windows PowerShell's Set-Content -Encoding utf8 writes a BOM.
    saved_comparison = json.loads(comparison.read_text(encoding="utf-8-sig"))
    for entry in saved_comparison["Results"]:
        state_path = Path(entry["State"])
        state = read_json_file(state_path, {})
        evaluation = read_json_file(
            Path(state["final_checkpoint"]["path"]) / "validation.json", {}
        )
        curriculum = []
        for path in sorted(state_path.parent.glob("curriculum_*.json")):
            round_id = int(path.stem.split("_")[-1])
            canonical = next(x for x in state["rounds"] if x["round"] == round_id)
            curriculum.append(
                {
                    "round": round_id,
                    "tokens": canonical["tokens"],
                    "macro_nll": read_json_file(path, {})["macro_nll"],
                }
            )
        categories = Counter()
        replay_tokens = 0
        by_id = {x["id"]: x for x in state["ledger"]}
        for loss in state["losses"]:
            category = by_id[loss["candidate_id"]]["sample"]["tag"]
            categories[category] += loss["answer_tokens"]
            if loss["source"] == "recycled":
                replay_tokens += loss["answer_tokens"]
        rows.append(
            {
                "arm": entry["Condition"],
                "seed": state["config"]["seed"],
                "status": state["status"],
                "tokens": state["tokens"],
                "updates": len(
                    {(x["round"], x["tokens_after_group"]) for x in state["losses"]}
                ),
                "checkpoint_nll": evaluation["macro_nll"],
                "category_tokens": dict(categories),
                "replay_tokens": replay_tokens,
                "curriculum_curve": curriculum,
                "unavailable": [
                    "Untrained baseline score",
                    "Fixed training probe curve",
                    "Gradient norms and clipping frequency",
                    "Executed learning-rate trace",
                ],
                "scope": "Historical count-matched run; tokens and updates differ across methods.",
            }
        )
    output.mkdir(parents=True)
    write_json_file(
        output / "existing_run_diagnostics.json",
        {
            "source_comparison": str(comparison.resolve()),
            "results": rows,
            "new_provider_calls": 0,
            "new_model_evaluations": 0,
            "final_test_scored": False,
        },
    )
    lines = [
        "# Existing-run diagnostics",
        "",
        "Read-only reconstruction from saved state. No new model evaluation or API requests.",
        "",
        "| Arm | Tokens | Updates | Validation NLL | Replay token share |",
        "| --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        lines.append(
            f"| {row['arm']} | {row['tokens']} | {row['updates']} | {row['checkpoint_nll']:.5f} | {row['replay_tokens']/row['tokens']:.1%} |"
        )
    lines += [
        "",
        "Missing historical measurements cannot be reconstructed: untrained scores, fixed training probes, gradient norms/clipping and executed learning rates.",
        "Category exposure and available curriculum curves are in existing_run_diagnostics.json. These runs do not isolate selection: banks, initial batches, tokens and update counts differ.",
    ]
    (output / "EXISTING_RUN_REPORT.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    if figures:
        _figures(output, rows)
        import matplotlib.pyplot as plt

        figure, axes = plt.subplots(1, 2, figsize=(13, 5))
        category_names = sorted({c for row in rows for c in row["category_tokens"]})
        palette = {c: plt.get_cmap("tab10")(i) for i, c in enumerate(category_names)}
        for row in rows:
            curve = row["curriculum_curve"]
            if curve:
                axes[0].plot(
                    [x["tokens"] for x in curve],
                    [x["macro_nll"] for x in curve],
                    marker=".",
                    label=row["arm"],
                )
            categories = row["category_tokens"]
            total = sum(categories.values())
            left = 0
            for category in sorted(categories):
                width = categories[category] / total
                axes[1].barh(
                    row["arm"],
                    width,
                    left=left,
                    color=palette[category],
                    label=category if row is rows[0] else None,
                )
                left += width
        axes[0].set(
            xlabel="Training answer-token exposures",
            ylabel="Curriculum macro NLL",
            title="Available curriculum curves (not checkpoint validation)",
        )
        axes[0].legend(fontsize=7)
        axes[1].set(
            xlabel="Share of answer-token exposures",
            title="Category allocation, including replay",
        )
        axes[1].legend(fontsize=6, loc="upper left", bbox_to_anchor=(1, 1))
        figure.tight_layout()
        figure.savefig(output / "existing_curriculum_and_exposure.png", dpi=150)
        plt.close(figure)
    return {
        "report": str(output / "EXISTING_RUN_REPORT.md"),
        "arms": len(rows),
        "new_provider_calls": 0,
    }


if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--existing-comparison", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            (
                historical_report(args.existing_comparison, args.output)
                if args.existing_comparison
                else report(args.output)
            ),
            indent=2,
        )
    )
