"""Portable preparation and explicit execution of local/API validation curricula.

Preparation loads a local tokenizer, never model weights or an API client.
Saved-bank execution disables network calls even for API-derived training data.
"""

import argparse
from collections import Counter
import copy
import json
import math
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parent
MODES = {
    "local-validation": "tefld_dataset_validation",
    "local-loss": "tefld_dataset_loss",
    "api-validation": "tefld_validation",
    "api-loss": "tefld_loss",
    "dolly": "dolly_baseline",
}


def prepare(
    root, mode, dataset, model, *, full_validation=False, examples=100, seed=42
):
    from transformers import AutoTokenizer
    from TEFLD.src.candidate_pool import LeakageGuard, candidate_id
    from TEFLD.src.dataset_curriculum import dataset_protocol, reference_sample
    from TEFLD.src.experiment_runner import ExperimentRunner
    from TEFLD.src.helper import read_json_file, write_json_file, stable_json_hash
    from TEFLD.src.runtime import file_fingerprint

    base = read_json_file(
        ROOT / "experiment_design_2026-10-18/experiment_protocol_hibana.json", {}
    )
    protocol = dataset_protocol(base, "both")
    protocol["dataset"]["source"] = str(Path(dataset).resolve())
    protocol["dataset"]["manifest"] = str(
        ROOT / "experiment_design_2026-10-18/split_manifest.json"
    )
    protocol["common_training"]["model_path"] = str(Path(model).resolve())
    protocol["example_count_control"] = {"unique_training_examples": examples}
    protocol["teacher_comparison"] = {
        "arm": mode,
        "primary_ready": False,
        "shared_budget_required": True,
        "purpose": "Fresh integration pilot; no historical results attributed to changed code.",
    }
    controls = protocol["tefld_controls"]
    controls["verifier"] = {
        "model_id": controls["teacher_model_id"],
        "provider": copy.deepcopy(controls["provider"]),
    }
    protocol["controlled_entry"] = {
        "mode": mode,
        "method": MODES[mode],
        "seed": seed,
        "per_category": None if full_validation else 8,
        "safety_tokens": examples
        * 3
        * protocol["common_training"]["epochs_per_round"]
        * 512,
    }
    path = root / "protocol.json"
    if path.exists() and read_json_file(path, {}) != protocol:
        raise ValueError("Prepared settings changed; choose a fresh output directory.")
    write_json_file(path, protocol)
    runner = ExperimentRunner(path, root, allow_paid_api=False)
    tokenizer = AutoTokenizer.from_pretrained(
        model, local_files_only=True, trust_remote_code=False
    )
    guard = LeakageGuard(runner.data)
    warmup, groups = [], set()
    for row_id in runner._baseline_order(seed):
        try:
            sample = reference_sample(runner.data, row_id, tokenizer, guard)
        except ValueError:
            continue
        identity = candidate_id(sample)
        if identity in groups:
            continue
        groups.add(identity)
        warmup.append(sample.model_dump(mode="json"))
        if len(warmup) == 10:
            break
    if len(warmup) != 10:
        raise ValueError("Cannot construct the common training-only warmup.")
    warmup_path = root / f"shared_initial_seed_{seed}.json"
    previous = read_json_file(warmup_path, None)
    if previous is not None and previous != warmup:
        raise ValueError("Prepared warmup changed.")
    write_json_file(warmup_path, warmup)
    plan = {
        "protocol_sha256": file_fingerprint(path),
        "core_code_sha256": runner.code_hash,
        "entrypoint_sha256": file_fingerprint(Path(__file__)),
        "warmup_sha256": {str(seed): file_fingerprint(warmup_path)},
        "curriculum_validation_rows": [
            i
            for i, _ in runner.data.samples(
                "curriculum_validation", protocol["controlled_entry"]["per_category"]
            )
        ],
        "comparison_validation_rows": [
            i
            for i, _ in runner.data.samples(
                "checkpoint_validation", protocol["controlled_entry"]["per_category"]
            )
        ],
        "final_test_scored": False,
    }
    prior = read_json_file(root / "study_plan.json", None)
    if prior is not None and prior != plan:
        raise ValueError("Prepared code or data changed; use a fresh study directory.")
    write_json_file(root / "study_plan.json", plan)
    reference_audit = ROOT / "reference/verifier_audit.json"
    audit = read_json_file(reference_audit, {})
    if (
        audit.get("status") == "passed"
        and audit.get("teacher_identity") == runner.verifier.identity
    ):
        write_json_file(root / "verification_audit.json", audit)
    if not (root / "study_budget.json").exists():
        write_json_file(
            root / "study_budget.json",
            {"approved": False, "currency": "toman", "max_spend": 0},
        )
    return {
        "status": "prepared",
        "method": MODES[mode],
        "unique_training_examples": examples,
        "comparison_validation_examples": len(plan["comparison_validation_rows"]),
        "provider_calls": 0,
    }


def run(root, bank_path=None, *, allow_paid=False, budget=None):
    from TEFLD.src.experiment_runner import ExperimentRunner
    from TEFLD.src.helper import read_json_file, write_json_file
    from TEFLD.src.runtime import file_fingerprint
    from TEFLD.src.teacher_accounting import study_spend
    from run_five_way_training import configure_client_limits

    path = root / "protocol.json"
    protocol = read_json_file(path, {})
    plan = read_json_file(root / "study_plan.json", {})
    settings = protocol["controlled_entry"]
    local = settings["mode"].startswith("local") or settings["mode"] == "dolly"
    if bank_path and local:
        raise ValueError(
            "Local methods retrieve original dataset references, not synthetic banks."
        )
    if allow_paid and (local or bank_path):
        raise ValueError(
            "Local and saved-bank execution require paid API calls to remain disabled."
        )
    runner = ExperimentRunner(path, root, allow_paid_api=allow_paid)
    if (
        plan.get("core_code_sha256") != runner.code_hash
        or plan.get("protocol_sha256") != file_fingerprint(path)
        or plan.get("entrypoint_sha256") != file_fingerprint(Path(__file__))
    ):
        raise ValueError(
            "Prepared code or protocol changed; use a fresh study directory."
        )
    if not local and not bank_path:
        if not allow_paid or budget is None or not math.isfinite(budget) or budget <= 0:
            raise ValueError(
                "Live generation requires explicit --allow-paid-api and --teacher-budget-toman."
            )
        if budget < study_spend(root).get("toman", 0):
            raise ValueError("Requested total cap is below recorded spending.")
        write_json_file(
            root / "study_budget.json",
            {"approved": True, "currency": "toman", "max_spend": budget},
        )
        configure_client_limits(runner, budget)
        last_request = [0.0]
        for client in (runner.teacher, runner.verifier):
            original = client.ask

            def paced(prompt, kind, _ask=original):
                delay = 13 - (time.monotonic() - last_request[0])
                if delay > 0:
                    time.sleep(delay)
                last_request[0] = time.monotonic()
                return _ask(prompt, kind)

            client.ask = paced
    bank = read_json_file(Path(bank_path), None) if bank_path else None
    if bank_path and not bank:
        raise ValueError("Saved candidate bank is empty or missing; no live fallback.")
    if bank:
        from TEFLD.src.dataschema import Training_Sample
        from TEFLD.src.candidate_pool import candidate_id

        parsed = [Training_Sample.model_validate(item) for item in bank]
        if len({candidate_id(sample) for sample in parsed}) != len(parsed):
            raise ValueError("Saved bank contains duplicate prompts.")
        if any(
            sample.verification.get("status") != "accepted_probabilistic_review"
            or any(
                sample.verification.get("result", {}).get(key) is not True
                for key in ("correct", "supported", "format_valid", "unambiguous")
            )
            for sample in parsed
        ):
            raise ValueError("Use an exported, previously verified synthetic bank.")
    return runner.run(
        settings["method"],
        settings["seed"],
        phase="pilot",
        budget=settings["safety_tokens"],
        per_category=settings["per_category"],
        candidate_bank=bank,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "run", "status"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=MODES)
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--examples", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--full-validation", action="store_true")
    parser.add_argument("--candidate-bank", type=Path)
    parser.add_argument("--allow-paid-api", action="store_true")
    parser.add_argument("--teacher-budget-toman", type=float)
    args = parser.parse_args()
    from TEFLD.src.transactions import section_lock
    from TEFLD.src.helper import read_json_file

    root = args.output.resolve()
    if args.command == "status":
        print(
            json.dumps(
                [
                    {
                        "path": str(p),
                        "status": read_json_file(p, {})["status"],
                        "unique": len(read_json_file(p, {})["ledger"]),
                    }
                    for p in root.glob("arms/*/pilot/*/state.json")
                ],
                indent=2,
            )
        )
        return
    with section_lock(root):
        if args.command == "prepare":
            if (
                not args.mode
                or not args.dataset
                or not args.model
                or args.examples < 10
            ):
                parser.error(
                    "prepare requires --mode, --dataset, --model and at least 10 examples."
                )
            if args.allow_paid_api or args.teacher_budget_toman is not None:
                parser.error("Preparation does not accept paid options.")
            print(
                json.dumps(
                    prepare(
                        root,
                        args.mode,
                        args.dataset,
                        args.model,
                        full_validation=args.full_validation,
                        examples=args.examples,
                        seed=args.seed,
                    ),
                    indent=2,
                )
            )
        else:
            result = run(
                root,
                args.candidate_bank,
                allow_paid=args.allow_paid_api,
                budget=args.teacher_budget_toman,
            )
            print(
                json.dumps(
                    {
                        "status": result["status"],
                        "unique_examples": len(result["ledger"]),
                        "answer_token_exposures": result["tokens"],
                        "final_checkpoint": result["final_checkpoint"],
                    },
                    indent=2,
                )
            )


if __name__ == "__main__":
    main()
