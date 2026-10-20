"""Load local credentials into this child process, then invoke recovery.

Kept separate from the fingerprinted training code: credential loading does not
amend the experiment or require recreating its prepared recovery plans.
"""

import argparse
import os
from pathlib import Path
import runpy
import sys


def load_credentials(path):
    from dotenv import dotenv_values

    path = Path(path)
    if not path.is_file():
        raise ValueError("Requested environment file does not exist.")
    # Read only the requested credential. Disable interpolation so unrelated
    # variables cannot substitute themselves into the key.
    value = dotenv_values(path, encoding="utf-8-sig", interpolate=False).get(
        "HIBANA_API_KEY"
    )
    if value and value.strip() and not os.environ.get("HIBANA_API_KEY", "").strip():
        os.environ["HIBANA_API_KEY"] = value.strip()
    if not os.environ.get("HIBANA_API_KEY", "").strip():
        raise ValueError("HIBANA_API_KEY is missing or empty in the environment and specified file.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-file", type=Path, required=True)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("recovery_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    try:
        load_credentials(args.env_file)
    except ValueError as exc:
        parser.error(str(exc))
    if args.check:
        print("Hibana credential is available; value hidden. No API calls made.")
        return
    recovery_args = args.recovery_args
    if recovery_args[:1] == ["--"]:
        recovery_args = recovery_args[1:]
    target = Path(__file__).resolve().with_name("run_bank_recovery.py")
    sys.argv = [str(target), *recovery_args]
    runpy.run_path(str(target), run_name="__main__")


if __name__ == "__main__":
    main()
