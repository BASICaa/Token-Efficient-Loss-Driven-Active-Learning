"""Command-line entry for the isolated experiment runner."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=[
            "preflight",
            "audit-verifier",
            "pilot",
            "train",
            "freeze",
            "evaluate",
            "status",
            "prepare-bank",
            "selector-pilot",
            "collect-probes",
            "fit-predictor",
            "audit-predictor",
            "report",
            "ablate",
            "export-review",
            "costs",
        ],
    )
    parser.add_argument(
        "--protocol",
        default=str(
            Path(__file__).resolve().parent
            / "experiment_design_2026-10-18/experiment_protocol_hibana.json"
        ),
    )
    parser.add_argument("--output", default="experiment_runs_hibana")
    parser.add_argument("--model-path")
    parser.add_argument(
        "--gradient-geometry", choices=["raw", "subspace"], default="raw"
    )
    parser.add_argument(
        "--projection-dimensions", type=int, choices=[256, 1024], default=1024
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["tefld_validation", "tefld_loss", "dolly_baseline"],
    )
    parser.add_argument("--seeds", type=int, nargs="+")
    parser.add_argument("--budget", type=int)
    parser.add_argument("--no-replay", action="store_true")
    parser.add_argument(
        "--resume-pilot-after-fix",
        action="store_true",
        help="Explicitly adopt code-only changes for an interrupted pilot; preserve adapter, token count and amendment history.",
    )
    parser.add_argument("--candidate-bank")
    parser.add_argument("--predictor")
    parser.add_argument("--generate-answers", action="store_true")
    parser.add_argument("--full-audit", action="store_true")
    parser.add_argument(
        "--with-jev",
        action="store_true",
        help="Add cached Jev features to probe observations; cache misses stop collection.",
    )
    parser.add_argument(
        "--feature-mode", choices=["gradient", "jev", "hybrid"], default="gradient"
    )
    parser.add_argument(
        "--allow-paid-api",
        action="store_true",
        help="Explicitly enable paid teacher/Jev requests; disabled by default.",
    )
    parser.add_argument("--bank-per-category", type=int, default=64)
    parser.add_argument(
        "--probe-role", choices=["fit", "calibration", "audit"], default="fit"
    )
    parser.add_argument("--probe-bank")
    parser.add_argument("--exclude-bank")
    parser.add_argument("--per-snapshot", type=int, default=16)
    args = parser.parse_args()
    if args.resume_pilot_after_fix and args.command != "pilot":
        parser.error("--resume-pilot-after-fix applies only to pilot.")
    from TEFLD.src.experiment_runner import ExperimentRunner
    from TEFLD.src.helper import write_json_file

    runner = ExperimentRunner(
        args.protocol,
        args.output,
        model_path=args.model_path,
        allow_paid_api=args.allow_paid_api,
        gradient_geometry=args.gradient_geometry,
        projection_dimensions=args.projection_dimensions,
    )
    if args.command == "preflight":
        from TEFLD.src.encoding import encode_sample
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            runner.model_path, local_files_only=True, trust_remote_code=False
        )
        for role in ["training_pool", "curriculum_validation", "checkpoint_validation"]:
            for _, sample in runner.data.samples(role):
                encode_sample(sample, tokenizer)
        model_identity = runner.model_identity()
        report = {
            "model_sha256": model_identity,
            "status": "passed",
            "split_sizes": {k: len(v) for k, v in runner.data.ids.items()},
            "protocol_sha256": runner.protocol_hash,
            "code_sha256": runner.code_hash,
            "final_test_scored": False,
        }
        write_json_file(runner.output / "preflight.json", report)
        print(json.dumps(report, indent=2))
    elif args.command == "audit-verifier":
        from TEFLD.src.verifier_audit import audit_verifier

        with runner.verifier.account_to(
            arm="shared_verifier", phase="verifier_audit", method="audit"
        ):
            report = audit_verifier(runner.verifier, runner.verification_audit_path)
        print(json.dumps(report, indent=2))
        if report["status"] != "passed":
            raise SystemExit(1)
    elif args.command == "prepare-bank":
        from transformers import AutoTokenizer
        from TEFLD.src.selector_study import prepare_bank

        tokenizer = AutoTokenizer.from_pretrained(
            runner.model_path, local_files_only=True, trust_remote_code=False
        )
        path = (
            Path(args.candidate_bank)
            if args.candidate_bank
            else runner.output / "common_bank.json"
        )
        excluded = []
        if args.exclude_bank:
            from TEFLD.src.selector_study import group_id
            from TEFLD.src.dataschema import Training_Sample

            excluded = [
                group_id(Training_Sample.model_validate(x))
                for x in json.loads(Path(args.exclude_bank).read_text(encoding="utf-8"))
            ]
        prepare_bank(
            runner.data,
            tokenizer,
            path,
            per_category=args.bank_per_category,
            exclude_groups=excluded,
        )
        print("Prepared training-only reference bank:", path)
    elif args.command in {"fit-predictor", "audit-predictor"}:
        from TEFLD.src.selector_study import fit_grouped_predictor, audit_predictor

        action = (
            fit_grouped_predictor
            if args.command == "fit-predictor"
            else audit_predictor
        )
        report = (
            action(
                runner.output / "benefit_calibration", feature_mode=args.feature_mode
            )
            if args.command == "fit-predictor"
            else action(runner.output / "benefit_calibration")
        )
        print(json.dumps(report, indent=2))
        if args.command == "audit-predictor" and args.full_audit:
            from TEFLD.src.selector_study import full_view_audit

            full_view_audit(
                runner.output / "benefit_calibration", runner.data, runner.model_path
            )
    elif args.command == "collect-probes":
        from TEFLD.src.selector_study import ProbeCollector

        if not args.probe_bank:
            raise ValueError(
                "--probe-bank must contain verified training-only candidates, separate from trajectory candidates."
            )
        probe_bank = json.loads(Path(args.probe_bank).read_text(encoding="utf-8"))
        bank = (
            json.loads(Path(args.candidate_bank).read_text(encoding="utf-8"))
            if args.candidate_bank
            else None
        )
        role = args.probe_role
        settings = runner.protocol["benefit_prediction"]["calibration"][
            {
                "fit": "fit",
                "calibration": "calibration",
                "audit": "locked_selector_audit",
            }[role]
        ]
        collector = ProbeCollector(
            runner,
            runner.output / "benefit_calibration",
            role,
            settings["snapshots"],
            per_snapshot=args.per_snapshot,
            probe_bank=probe_bank,
            with_jev=args.with_jev,
        )
        runner.run(
            "tefld_validation",
            settings["trajectory_seed"],
            phase="auxiliary",
            budget=args.budget or 30000,
            per_category=None,
            candidate_bank=bank,
            probe_callback=collector,
        )
    elif args.command in {"pilot", "train", "selector-pilot", "ablate"}:
        bank = (
            json.loads(Path(args.candidate_bank).read_text(encoding="utf-8"))
            if args.candidate_bank
            else None
        )
        predictor = None
        if args.predictor:
            from TEFLD.src.benefit_predictor import BenefitPredictor

            predictor = BenefitPredictor.load(Path(args.predictor))
        if args.command == "selector-pilot" and not bank:
            raise ValueError("A common bank is required for selector isolation.")
        seeds = args.seeds or (
            [42, 123, 2026] if args.command in {"train", "ablate"} else [42]
        )
        budget = args.budget or (30000 if args.command in {"train", "ablate"} else 5000)
        if args.command == "train" and args.no_replay:
            raise ValueError("Use ablate for replay-policy changes.")
        for method in args.methods:
            for seed in seeds:
                runner.run(
                    method,
                    seed,
                    phase={
                        "pilot": "pilot",
                        "train": "primary",
                        "selector-pilot": "selector",
                        "ablate": "ablation",
                    }[args.command],
                    budget=budget,
                    per_category=None if args.command in {"train", "ablate"} else 8,
                    replay=not args.no_replay,
                    candidate_bank=bank,
                    predictor=predictor,
                    resume_pilot_after_fix=args.resume_pilot_after_fix,
                )
    elif args.command == "freeze":
        predictor = None
        if args.predictor:
            from TEFLD.src.benefit_predictor import BenefitPredictor

            predictor = BenefitPredictor.load(Path(args.predictor))
        runner.freeze(
            args.methods,
            args.seeds or [42, 123, 2026],
            args.budget or 30000,
            predictor=predictor,
        )
    elif args.command == "evaluate":
        runner.evaluate_final(include_generated=args.generate_answers)
    elif args.command == "export-review":
        from TEFLD.src.reporting import export_blinded_review

        print(json.dumps(export_blinded_review(runner.output, runner.data), indent=2))
    elif args.command == "report":
        from TEFLD.src.reporting import summarize_study

        print(json.dumps(summarize_study(runner.output, runner.data), indent=2))
    elif args.command == "costs":
        from TEFLD.src.teacher_accounting import summarize_teacher_costs

        report = summarize_teacher_costs(runner.study_root)
        write_json_file(runner.study_root / "teacher_cost_report.json", report)
        print(json.dumps(report, indent=2))
    else:
        for path in sorted(runner.output.glob("*/*/state.json")):
            state = json.loads(path.read_text(encoding="utf-8"))
            print(
                json.dumps(
                    {
                        "run": str(path.parent),
                        "status": state["status"],
                        "tokens": state["tokens"],
                        "round": state["round"],
                    }
                )
            )


if __name__ == "__main__":
    main()
