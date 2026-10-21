"""Prepare/report a single continuous-optimizer run against a saved reset control.

Random selection and replay off hold every selected example/order fixed while
optimizer and LR lifecycle change. Preparation/reporting never load a model.
"""

import argparse
import copy
import json
from pathlib import Path
import shutil

from prepare_replay_study import validate_control
from run_selection_diagnostic import code_identity
from TEFLD.src.helper import read_json_file, write_json_file
from TEFLD.src.runtime import file_fingerprint
from TEFLD.src.transactions import section_lock


def prepare(source, output, seed=42):
    source, output = Path(source).resolve(), Path(output).resolve()
    if source == output or source in output.parents or output in source.parents:
        raise ValueError("Use a separate optimizer-study directory.")
    if output.exists():
        raise ValueError("Optimizer study exists; rerun its launcher to resume.")
    with section_lock(source):
        plan = read_json_file(source / "plan.json", {})
        if plan["code_sha256"] != code_identity():
            raise ValueError("Source training code changed.")
        if (
            plan["settings"]["optimizer_lifecycle"] != "reset-per-round"
            or seed not in plan["settings"]["seeds"]
            or plan["settings"]["replays"] != ["off"]
        ):
            raise ValueError(
                "Require an existing reset-per-round, selection-only control for this seed."
            )
        for filename, field in (
            ("bank.json", "bank_sha256"),
            ("protocol.json", "protocol_sha256"),
        ):
            if file_fingerprint(source / filename) != plan[field]:
                raise ValueError("Source bank or protocol changed.")
        state_path = source / f"seed_{seed}/random__replay_off/state.json"
        control = read_json_file(state_path, {})
        validate_control(
            control, plan, file_fingerprint(source / "plan.json"), seed, "random"
        )
        new = copy.deepcopy(plan)
        new["settings"].update(seeds=[seed], optimizer_lifecycle="continuous")
        new["schedules"] = {str(seed): plan["schedules"][str(seed)]}
        new["tokens_by_seed"] = {str(seed): plan["tokens_by_seed"][str(seed)]}
        new["optimizer_comparison"] = {
            "planned_selection": "random",
            "planned_replay": "off",
            "planned_new_runs": 1,
            "control_path": str(state_path),
            "control_sha256": file_fingerprint(state_path),
            "source_plan": str(source / "plan.json"),
            "source_plan_sha256": file_fingerprint(source / "plan.json"),
            "contrast": "Continuous AdamW plus global linear LR versus round-reset AdamW plus round-local linear LR.",
            "scope": "One-seed development diagnostic, chosen after observing earlier results. No primary holdout scored.",
        }
        output.mkdir(parents=True)
        for filename in ("bank.json", "protocol.json"):
            shutil.copy2(source / filename, output / filename)
        write_json_file(output / "reset_control.json", control)
        new["optimizer_comparison"]["copied_control_sha256"] = file_fingerprint(
            output / "reset_control.json"
        )
        write_json_file(output / "plan.json", new)
        return {
            "status": "prepared",
            "seed": seed,
            "new_runs": 1,
            "reused_reset_controls": 1,
            "updates": new["settings"]["updates"],
            "answer_tokens": new["tokens_by_seed"][str(seed)],
            "selection": "random",
            "replay": "off",
            "training_started": False,
            "provider_calls": 0,
        }


def report(output, *, figures=True):
    output = Path(output).resolve()
    plan = read_json_file(output / "plan.json", {})
    comparison = plan["optimizer_comparison"]
    if plan["code_sha256"] != code_identity():
        raise ValueError("Prepared training implementation changed.")
    if (
        file_fingerprint(output / "reset_control.json")
        != comparison["copied_control_sha256"]
    ):
        raise ValueError("Copied reset control changed.")
    control = read_json_file(output / "reset_control.json", {})
    seed = plan["settings"]["seeds"][0]
    path = output / f"seed_{seed}/random__replay_off/state.json"
    if not path.exists():
        return {"status": "not started", "remaining_runs": 1}
    continuous = read_json_file(path, {})
    expected_identity = {
        "plan_sha256": file_fingerprint(output / "plan.json"),
        "selection": "random",
        "replay": "off",
        "seed": seed,
    }
    if continuous["identity"] != expected_identity:
        raise ValueError("Continuous run identity differs from prepared plan.")
    executed = continuous["selection_trace"]
    old = control["selection_trace"]
    keys = ("update", "candidate_id", "answer_tokens", "replay")
    if any(
        {key: actual[key] for key in keys} != {key: expected[key] for key in keys}
        for actual, expected in zip(executed, old)
    ) or len(executed) > len(old):
        raise ValueError(
            "Selected data/order differs; this is not an optimizer-only recipe comparison."
        )
    if continuous["evaluations"][0] != control["evaluations"][0]:
        raise ValueError("Starting evaluations differ.")
    if continuous["status"] == "complete":
        if (
            continuous["tokens"] != control["tokens"]
            or continuous["update"] != control["update"]
            or len(executed) != len(old)
        ):
            raise ValueError("Completed resource budgets differ.")
    rows = []
    for name, state in (("reset-per-round", control), ("continuous", continuous)):
        final = state["evaluations"][-1]
        rows.append(
            {
                "optimizer_lifecycle": name,
                "status": state["status"],
                "updates": state["update"],
                "tokens": state["tokens"],
                "unique_examples": len(state["seen"]),
                "macro_nll": final["checkpoint"]["macro_nll"],
                "token_weighted_nll": final["checkpoint"]["token_weighted_nll"],
                "round_resets": sum(x["optimizer_reset"] for x in state["updates"]),
                "clipped_fraction": (
                    sum(x["clipped"] for x in state["updates"]) / len(state["updates"])
                    if state["updates"]
                    else None
                ),
            }
        )
    complete = continuous["status"] == "complete"
    result = {
        "scope": comparison["scope"],
        "contrast": comparison["contrast"],
        "seed": seed,
        "results": rows,
        "same_selected_data_and_order": True,
        "delta_nll_continuous_minus_reset": (
            rows[1]["macro_nll"] - rows[0]["macro_nll"] if complete else None
        ),
        "remaining_runs": 0 if complete else 1,
        "final_test_scored": False,
        "provider_calls": 0,
        "limitation": "Changes both AdamW state retention and LR schedule; one seed, replay off, random selection. Does not establish an optimizer-moments-only or adaptive-TEFLD effect.",
    }
    write_json_file(output / "optimizer_comparison.json", result)
    lines = [
        "# Continuous optimizer diagnostic",
        "",
        comparison["contrast"],
        "",
        result["limitation"],
        "",
        "| Recipe | Status | Updates | Tokens | Validation macro NLL |",
        "| --- | --- | --- | --- | --- |",
    ]
    lines += [
        f"| {r['optimizer_lifecycle']} | {r['status']} | {r['updates']} | {r['tokens']} | {r['macro_nll']:.6f} |"
        for r in rows
    ]
    lines += [
        "",
        "Same selected data and order verified. Only final matched-budget scores are a complete comparison.",
        "Negative continuous-minus-reset NLL favors continuous; do not choose an intermediate checkpoint after viewing these curves.",
    ]
    (output / "OPTIMIZER_REPORT.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    if figures:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        figure, axes = plt.subplots(1, 3, figsize=(14, 4))
        for name, state in (("Reset each round", control), ("Continuous", continuous)):
            observations = state["evaluations"]
            axes[0].plot(
                [x["tokens"] for x in observations],
                [x["checkpoint"]["macro_nll"] for x in observations],
                marker=".",
                label=name,
            )
            axes[1].plot(
                [x["update"] for x in state["updates"]],
                [x["learning_rate"] for x in state["updates"]],
                label=name,
            )
            axes[2].plot(
                [x["update"] for x in observations],
                [x["warmup_probe"]["macro_nll"] for x in observations],
                label=name,
            )
        axes[0].set(
            xlabel="Answer-token exposures",
            ylabel="Checkpoint macro NLL",
            title="Validation",
        )
        axes[1].set(
            xlabel="Optimizer updates",
            ylabel="Executed learning rate",
            title="Learning-rate lifecycle",
        )
        axes[2].set(
            xlabel="Optimizer updates",
            ylabel="Fixed warmup macro NLL",
            title="Same exposed training probe",
        )
        axes[0].legend()
        figure.tight_layout()
        figure.savefig(output / "optimizer_comparison.png", dpi=150)
        plt.close(figure)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "report"))
    parser.add_argument("--source", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.command == "prepare" and not args.source:
        parser.error("prepare requires --source")
    print(
        json.dumps(
            (
                prepare(args.source, args.output, args.seed)
                if args.command == "prepare"
                else report(args.output)
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
