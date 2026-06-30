from __future__ import annotations

from TEFLD.src.orchestrator import Orchestrator


if __name__ == "__main__":
    result = Orchestrator().run(max_rounds=1)
    print(result)

