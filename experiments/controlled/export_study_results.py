"""Export compact measured results from a private workspace, without raw text or weights."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

LABELS = {
    "dolly_baseline": "Dolly",
    "tefld_dataset_loss": "Dataset TEFLD (loss)",
    "sol61_validation": "GPT-6.1 validation",
    "sol61_loss": "GPT-6.1 loss",
    "luna_validation": "Luna validation",
}


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def export(workspace, output):
    study = workspace / "experiment_runs_five_way_100_training"
    source = read(study / "comparison.json")
    costs = {}
    for group in source["teacher_costs"]["attributed_groups"]:
        item = costs.setdefault(
            group["owner"]["arm"], {"generation": 0, "verification": 0}
        )
        item[group["kind"]] += group["known_toman"]
    methods, ids, warmup_hashes = {}, None, []
    for arm, summary in source["results"].items():
        directory = next((study / "arms" / arm / "pilot").iterdir())
        state = read(directory / "state.json")
        checkpoint = Path(state["final_checkpoint"]["path"])
        scores = read(checkpoint / "validation.json")
        contract = [
            (r["row_id"], r["category"], r["answer_tokens"]) for r in scores["rows"]
        ]
        if ids is not None and contract != ids:
            raise ValueError("Evaluation rows or target masks differ.")
        ids = contract
        if (
            state["status"] != "complete"
            or len({x["id"] for x in state["ledger"]}) != 100
        ):
            raise ValueError("Publish only completed 100-example arms.")
        if state["tokens"] != sum(x["answer_tokens"] for x in state["losses"]):
            raise ValueError("Exposure ledger does not reconcile.")
        marker = read(checkpoint / "completion.json")
        if (
            not marker.get("complete")
            or sha(checkpoint / "adapter_model.safetensors") != marker["weights_sha256"]
        ):
            raise ValueError("Checkpoint integrity failed.")
        warmup_hashes.append(sha(directory / "batch_0000.json"))
        curve = []
        for r in state["rounds"]:
            round_id = r["round"]
            point = {
                "round": round_id,
                "unique_examples": sum(
                    x["first_round"] <= round_id for x in state["ledger"]
                ),
                "token_exposures": r["tokens"],
            }
            curriculum = directory / f"curriculum_{round_id:04d}.json"
            if curriculum.exists():
                point["curriculum_validation_nll"] = read(curriculum)["macro_nll"]
            curve.append(point)
        method = {key: value for key, value in summary.items() if key != "checkpoint"}
        method.update(
            label=LABELS[arm],
            categories=scores["categories"],
            evaluation_rows=scores["rows"],
            training_curve=curve,
            unique_examples_by_category=dict(
                Counter(x["sample"]["tag"] for x in state["ledger"])
            ),
            api_cost_toman=costs.get(arm, {"generation": 0, "verification": 0}),
            training_core_sha256=state["config"]["code_sha256"],
            state_sha256=sha(directory / "state.json"),
            weights_sha256=marker["weights_sha256"],
            validation_sha256=sha(checkpoint / "validation.json"),
        )
        admissions = directory / "candidate_admissions.json"
        if admissions.exists():
            bank = read(admissions)
            method["saved_bank_inventory"] = {
                "accepted": len(bank),
                "categories": dict(Counter(x["sample"]["tag"] for x in bank.values())),
            }
        methods[arm] = method
    if len(set(warmup_hashes)) != 1:
        raise ValueError("Warmup differs across arms.")
    snapshot = {
        "schema_version": 1,
        "date": "2026-10-19",
        "seed": 42,
        "base_model": "gemma-4-E2B-it",
        "unique_training_examples_per_arm": 100,
        "evaluation_role": "checkpoint_validation",
        "evaluation_examples_per_arm": 64,
        "evaluation_per_category": 8,
        "warmup_examples_included_in_100": 10,
        "warmup_identical": True,
        "warmup_sha256": warmup_hashes[0],
        "metric": "reference-answer macro category NLL; lower is better; not accuracy",
        "ranking": source["ranking"],
        "methods": methods,
        "total_api_cost_toman": source["teacher_costs"]["counted_spend"]["toman"],
        "token_matched": False,
        "final_test_scored": False,
        "local_validation_guided_result": None,
        "limitations": [
            "One seed",
            "64 reused development examples",
            "Unequal token exposure and updates",
            "No final-test scoring",
            "No blinded generated-answer assessment",
            "Integration code has changed since the measured study",
        ],
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "snapshot.json").write_text(
        json.dumps(snapshot, indent=2) + "\n", encoding="utf-8"
    )
    return snapshot


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    snapshot = export(args.workspace.resolve(), args.output.resolve())
    print(
        json.dumps(
            {
                "arms": len(snapshot["methods"]),
                "examples_per_arm": 100,
                "evaluation_per_arm": 64,
                "provider_calls": 0,
                "raw_training_text_exported": False,
            }
        )
    )
