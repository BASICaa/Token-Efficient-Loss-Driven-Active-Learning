"""Prepare two teacher arms and compact-prompt comparisons; no paid calls by default."""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=[
            "prepare",
            "preview",
            "prompt-pilot",
            "costs",
            "import-verifier-audit",
        ],
    )
    parser.add_argument("--output", default="experiment_runs_two_teachers")
    parser.add_argument(
        "--base-protocol",
        default=str(
            Path(__file__).resolve().parent
            / "experiment_design_2026-10-18/experiment_protocol_hibana.json"
        ),
    )
    parser.add_argument("--allow-paid-api", action="store_true")
    parser.add_argument(
        "--dataset-feedback",
        choices=["loss", "validation", "both"],
        help="Prepare a fresh study with teacher-free dataset TEFLD; existing five-condition studies stay unchanged.",
    )
    parser.add_argument(
        "--audit-file",
        help="Existing matching fixed-verifier audit to reuse without provider calls.",
    )
    args = parser.parse_args()
    if args.dataset_feedback and args.command != "prepare":
        parser.error("--dataset-feedback applies only to prepare.")
    from TEFLD.src.helper import read_json_file, write_json_file
    from TEFLD.src.teacher_study import prepare_study, preview_prompts, prompt_pilot
    from TEFLD.src.teacher_accounting import summarize_teacher_costs

    root = Path(args.output).resolve()
    if args.command == "import-verifier-audit":
        if not args.audit_file:
            parser.error("--audit-file is required for import-verifier-audit.")
        from TEFLD.src.teacher_study import import_verifier_audit

        result = import_verifier_audit(root, Path(args.audit_file))
    elif args.command == "costs":
        result = summarize_teacher_costs(root)
        write_json_file(root / "teacher_cost_report.json", result)
    else:
        from transformers import AutoTokenizer

        protocol = read_json_file(Path(args.base_protocol), {})
        tokenizer = AutoTokenizer.from_pretrained(
            protocol["common_training"]["model_path"],
            local_files_only=True,
            trust_remote_code=False,
        )
        if args.command == "prepare":
            result = prepare_study(
                args.base_protocol,
                root,
                tokenizer=tokenizer,
                dataset_feedback=args.dataset_feedback,
            )
        elif args.command == "preview":
            result = preview_prompts(root, tokenizer)
        else:
            result = prompt_pilot(root, tokenizer, allow_paid_api=args.allow_paid_api)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
