"""Legacy pipeline entry; prefer run_experiments.py for controlled studies."""


def main():
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rounds", type=int, default=1)
    parser.add_argument("--build-only", action="store_true")
    args = parser.parse_args()
    try:
        from .src.orchestrator import Orchestrator
    except ImportError:
        from src.orchestrator import Orchestrator
    print(Orchestrator().run(max_rounds=args.rounds, build_only=args.build_only))


if __name__ == "__main__":
    main()
