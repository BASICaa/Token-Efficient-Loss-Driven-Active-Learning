"""Explicit model download utility; importing performs no I/O."""

import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", help="Hugging Face model ID")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    from huggingface_hub import snapshot_download

    snapshot_download(repo_id=args.model, local_dir=str(args.output))


if __name__ == "__main__":
    main()
