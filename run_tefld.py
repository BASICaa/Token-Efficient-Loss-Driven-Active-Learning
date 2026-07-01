from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from typing import Any

from TEFLD.src.helper import create_new_section
from TEFLD.src.orchestrator import Orchestrator, OrchestratorResult


def result_payload(result: OrchestratorResult) -> dict[str, Any]:
    payload = asdict(result)
    payload["example_path"] = str(result.example_path)
    payload["adapter_path"] = str(result.adapter_path) if result.adapter_path else None
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the TEFLD active-learning loop without comparison baselines."
    )
    parser.add_argument("--section-id", help="Existing section id, such as section_001.")
    parser.add_argument(
        "--new-section",
        action="store_true",
        help="Create a fresh section before running.",
    )
    parser.add_argument(
        "--rounds",
        type=int,
        default=1,
        help="Number of TEFLD rounds to run. Defaults to 1.",
    )
    parser.add_argument(
        "--student-model-id",
        default="gemma-4-E2B-it",
        help="Local model id or path for the student model.",
    )
    parser.add_argument(
        "--instructor-model-id",
        default="gpt-5.4-mini",
        help="OpenAI-compatible model id used for synthetic data generation.",
    )
    parser.add_argument(
        "--example-format",
        choices=["auto", "plain", "contextual"],
        default="auto",
        help="Expected example.txt format.",
    )
    parser.add_argument(
        "--build-only",
        action="store_true",
        help="Build and save the next 10-sample batch without training.",
    )
    parser.add_argument(
        "--bootstrap-only",
        action="store_true",
        help="Create/check section files and example.txt, then stop.",
    )
    parser.add_argument("--no-train", action="store_true", help="Do not train.")
    parser.add_argument(
        "--no-evaluate",
        action="store_true",
        help="Do not evaluate or advance the current batch.",
    )
    parser.add_argument(
        "--disable-api-grouping",
        action="store_true",
        help="Use deterministic tag averages instead of API semantic grouping.",
    )
    parser.add_argument(
        "--disable-shared-validation",
        action="store_true",
        help=(
            "Use training-batch loss for checkpoint decisions instead of the "
            "fixed shared_validation.json set."
        ),
    )
    parser.add_argument(
        "--shared-validation-size",
        type=int,
        default=10,
        help="Number of fixed shared-validation samples. Defaults to 10.",
    )
    parser.add_argument(
        "--epochs-per-round",
        type=float,
        default=1.0,
        help="LoRA training epochs per TEFLD round. Defaults to 1.",
    )
    parser.add_argument(
        "--interactive-section-selection",
        action="store_true",
        help="Choose an existing section or create a new one interactively.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    section_id = args.section_id
    if args.new_section:
        if section_id is not None:
            raise ValueError("Use either --new-section or --section-id, not both.")
        section_id = create_new_section()
        print(f"Created fresh section: {section_id}")

    orchestrator = Orchestrator(
        section_id=section_id,
        student_model_id=args.student_model_id,
        instructor_model_id=args.instructor_model_id,
        use_api_grouping=not args.disable_api_grouping,
        interactive_section_selection=args.interactive_section_selection,
        example_format=args.example_format,
        use_shared_validation=not args.disable_shared_validation,
        shared_validation_size=args.shared_validation_size,
        num_train_epochs=args.epochs_per_round,
    )

    if args.bootstrap_only:
        orchestrator.bootstrap_components()
        setup_result = orchestrator.prepare_example_state()
        if setup_result is not None:
            print(json.dumps(result_payload(setup_result), indent=2))
            return

        ready_result = orchestrator.result(
            status="ready",
            message="Section is ready for TEFLD generation/training.",
        )
        print(json.dumps(result_payload(ready_result), indent=2))
        return

    result = orchestrator.run(
        max_rounds=args.rounds,
        build_only=args.build_only,
        train=not args.no_train,
        evaluate=not args.no_evaluate,
    )
    print(json.dumps(result_payload(result), indent=2))


if __name__ == "__main__":
    main()
