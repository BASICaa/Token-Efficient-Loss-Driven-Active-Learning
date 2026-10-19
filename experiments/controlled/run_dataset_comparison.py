"""Prepare and pilot dataset-only TEFLD versus direct Dolly training, without APIs."""

import argparse
import json
from pathlib import Path


def main():
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["prepare", "pilot", "status"])
    parser.add_argument("--feedback", choices=["loss", "validation", "both"])
    parser.add_argument(
        "--base-protocol",
        default=str(
            root / "experiment_design_2026-10-18/experiment_protocol_hibana.json"
        ),
    )
    parser.add_argument(
        "--protocol",
        default=str(
            root / "experiment_design_2026-10-19/experiment_protocol_dataset_tefld.json"
        ),
    )
    parser.add_argument("--output", default="experiment_runs_dataset_tefld")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42])
    parser.add_argument("--budget", type=int, default=5000)
    parser.add_argument("--model-path")
    args = parser.parse_args()
    from TEFLD.src.dataset_curriculum import dataset_protocol, require_dataset_policy
    from TEFLD.src.helper import read_json_file, write_json_file

    path = Path(args.protocol)
    if args.command == "prepare":
        protocol = dataset_protocol(
            read_json_file(Path(args.base_protocol), {}), args.feedback or "loss"
        )
        if path.exists() and read_json_file(path, {}) != protocol:
            raise ValueError(
                "Dataset protocol already prepared differently; use a fresh --protocol path."
            )
        write_json_file(path, protocol)
        print(
            json.dumps(
                {
                    "status": "prepared",
                    "protocol": str(path.resolve()),
                    "methods": ["dolly_baseline"]
                    + protocol["dataset_tefld"]["enabled_methods"],
                    "provider_calls": 0,
                    "training_started": False,
                },
                indent=2,
            )
        )
    elif args.command == "status":
        rows = []
        for state_path in sorted(Path(args.output).glob("pilot/*/state.json")):
            state = read_json_file(state_path, {})
            rows.append(
                {
                    "run": str(state_path.parent),
                    "status": state["status"],
                    "tokens": state["tokens"],
                    "round": state["round"],
                }
            )
        print(json.dumps(rows, indent=2))
    else:
        if args.feedback:
            parser.error(
                "--feedback applies only to prepare; pilots use the saved protocol."
            )
        protocol = read_json_file(path, {})
        policy = require_dataset_policy(protocol)
        from TEFLD.src.experiment_runner import ExperimentRunner

        runner = ExperimentRunner(
            path, args.output, model_path=args.model_path, allow_paid_api=False
        )
        for method in ["dolly_baseline"] + policy["enabled_methods"]:
            for seed in args.seeds:
                runner.run(method, seed, budget=args.budget)


if __name__ == "__main__":
    main()
