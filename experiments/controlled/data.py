"""Explicit dataset download utility; importing performs no I/O."""

import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", help="Hugging Face dataset ID")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    from datasets import load_dataset

    dataset = load_dataset(args.dataset)
    dataset.save_to_disk(str(args.output))
    for split, rows in dataset.items():
        rows.to_json(
            str(args.output.with_name(args.output.name + "_" + split + ".jsonl"))
        )


if __name__ == "__main__":
    main()
