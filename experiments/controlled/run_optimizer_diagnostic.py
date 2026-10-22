"""Run the local-only fixed-curriculum optimizer comparison; never calls a teacher."""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["run", "status", "report"])
    parser.add_argument("--source", default="experiment_runs_hibana/pilot")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="optimizer_diagnostic_2026-10-19")
    parser.add_argument(
        "--protocol",
        default=str(
            Path(__file__).resolve().parent
            / "experiment_design_2026-10-18/experiment_protocol_hibana.json"
        ),
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["tefld_validation", "tefld_loss", "dolly_baseline"],
    )
    parser.add_argument(
        "--recipes", nargs="+", default=["original_four_examples", "fixed_250_tokens"]
    )
    args = parser.parse_args()
    from TEFLD.src.optimizer_diagnostic import (
        run_condition,
        summarize_diagnostic,
        RECIPES,
        SUPPORTED_METHODS,
    )
    from TEFLD.src.helper import read_json_file

    if any(x not in SUPPORTED_METHODS for x in args.methods) or any(
        x not in RECIPES for x in args.recipes
    ):
        parser.error("Unknown diagnostic method or recipe.")
    output = Path(args.output)
    if args.command == "run":
        for method in args.methods:
            for recipe in args.recipes:
                run_condition(
                    Path(args.source) / f"{method}_seed_{args.seed}",
                    output / method / recipe,
                    args.protocol,
                    recipe,
                )
        if set(args.recipes) == set(RECIPES):
            print(json.dumps(summarize_diagnostic(output, args.methods), indent=2))
    elif args.command == "report":
        print(json.dumps(summarize_diagnostic(output, args.methods), indent=2))
    else:
        for path in sorted(output.glob("*/*/state.json")):
            state = read_json_file(path, {})
            print(
                json.dumps(
                    {
                        "condition": str(path.parent),
                        "status": state["status"],
                        "tokens": state["tokens"],
                        "updates": state["next_group"],
                    }
                )
            )


if __name__ == "__main__":
    main()
