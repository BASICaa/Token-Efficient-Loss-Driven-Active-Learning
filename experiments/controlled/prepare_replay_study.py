"""Fork a compatible selection-only plan and reuse completed no-replay controls.

No tokenizer, model or provider is loaded. Training runs are separate commands.
"""

import argparse
import copy
import json
from pathlib import Path
import shutil

from run_selection_diagnostic import code_identity, SELECTIONS, REPLAYS
from TEFLD.src.experiment_runner import require_adapter
from TEFLD.src.helper import read_json_file, write_json_file
from TEFLD.src.runtime import file_fingerprint
from TEFLD.src.transactions import section_lock


def validate_control(state, plan, plan_hash, seed, selection):
    expected_identity = {
        "plan_sha256": plan_hash,
        "selection": selection,
        "replay": "off",
        "seed": seed,
    }
    if state.get("identity") != expected_identity or state.get("status") != "complete":
        raise ValueError("A completed, identity-matched no-replay control is required.")
    schedule = plan["schedules"][str(seed)]
    settings = plan["settings"]
    if (
        state["update"] != settings["updates"]
        or state["tokens"] != plan["tokens_by_seed"][str(seed)]
    ):
        raise ValueError("Control resource totals differ from the plan.")
    if len(state["updates"]) != len(schedule) or len(state["selection_trace"]) != sum(
        map(len, schedule)
    ):
        raise ValueError("Control has incomplete update or selection traces.")
    seen = set()
    for update, slots in enumerate(schedule, 1):
        actual = state["updates"][update - 1]
        traces = [x for x in state["selection_trace"] if x["update"] == update]
        if (
            actual["update"] != update
            or actual["answer_tokens"] != sum(x["length"] for x in slots)
            or len(traces) != len(slots)
        ):
            raise ValueError("Control per-update token schedule differs.")
        for slot, trace in zip(slots, traces):
            if (
                trace["replay"]
                or trace["answer_tokens"] != slot["length"]
                or trace["candidate_id"] in seen
            ):
                raise ValueError("Control is not a full-answer, no-repeat schedule.")
            if slot["warmup_id"] and trace["candidate_id"] != slot["warmup_id"]:
                raise ValueError("Control warmup differs from the plan.")
            seen.add(trace["candidate_id"])
    if seen != set(state["seen"]):
        raise ValueError("Control exposure identities differ from its trace.")
    for observation in state["evaluations"]:
        if [x["row_id"] for x in observation["checkpoint"]["rows"]] != plan[
            "checkpoint_rows"
        ] or [x["row_id"] for x in observation["curriculum"]["rows"]] != plan[
            "curriculum_rows"
        ]:
            raise ValueError("Control evaluation rows differ from the plan.")
    if (
        state["evaluations"][0]["update"] != 0
        or state["evaluations"][-1]["update"] != settings["updates"]
    ):
        raise ValueError("Control lacks starting/final evaluations.")
    require_adapter(state["adapter"])


def prepare(source, output, seeds=(42,)):
    source, output = Path(source).resolve(), Path(output).resolve()
    if output == source or source in output.parents or output in source.parents:
        raise ValueError("Use separate source and replay output directories.")
    if output.exists():
        raise ValueError("Output already exists; use the run command to resume.")
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("Choose distinct prepared seeds.")
    with section_lock(source):
        plan = read_json_file(source / "plan.json", {})
        source_hash = file_fingerprint(source / "plan.json")
        if plan["code_sha256"] != code_identity():
            raise ValueError(
                "Training implementation changed; controls cannot be reused."
            )
        for filename, field in (
            ("protocol.json", "protocol_sha256"),
            ("bank.json", "bank_sha256"),
        ):
            if file_fingerprint(source / filename) != plan[field]:
                raise ValueError("Source protocol or bank changed.")
        if plan["settings"]["replays"] != ["off"] or any(
            seed not in plan["settings"]["seeds"] for seed in seeds
        ):
            raise ValueError(
                "Source must be a selection-only study containing the requested seeds."
            )
        controls = []
        for seed in seeds:
            schedule = plan["schedules"][str(seed)]
            if not any(slot["replay_slot"] for group in schedule for slot in group):
                raise ValueError("Prepared seed has no replay opportunities.")
            for selection in SELECTIONS:
                path = source / f"seed_{seed}/{selection}__replay_off/state.json"
                state = read_json_file(path, {})
                validate_control(state, plan, source_hash, seed, selection)
                controls.append((seed, selection, path, state))
            seed_controls = [state for s, _, _, state in controls if s == seed]
            optimizer_traces = [
                [
                    (x.get("learning_rate"), x.get("optimizer_reset"))
                    for x in state["updates"]
                ]
                for state in seed_controls
            ]
            initial_scores = [state["evaluations"][0] for state in seed_controls]
            if any(trace != optimizer_traces[0] for trace in optimizer_traces) or any(
                scores != initial_scores[0] for scores in initial_scores
            ):
                raise ValueError(
                    "Controls differ in executed optimizer schedule or initial evaluations."
                )
        # No output is created until every reuse compatibility check has passed.
        replay_plan = copy.deepcopy(plan)
        replay_plan["settings"].update(seeds=list(seeds), replays=list(REPLAYS))
        replay_plan["schedules"] = {
            str(seed): plan["schedules"][str(seed)] for seed in seeds
        }
        replay_plan["tokens_by_seed"] = {
            str(seed): plan["tokens_by_seed"][str(seed)] for seed in seeds
        }
        replay_plan["reuse_controls"] = {
            "source_plan": str(source / "plan.json"),
            "source_plan_sha256": source_hash,
            "scope": "Completed no-replay controls reused after checking unchanged code, bank, protocol, schedules, warmup, per-update exposures, evaluation rows and adapter integrity.",
            "states": [
                {
                    "seed": seed,
                    "selection": selection,
                    "path": str(path),
                    "sha256": file_fingerprint(path),
                }
                for seed, selection, path, state in controls
            ],
        }
        output.mkdir(parents=True)
        for filename in ("bank.json", "protocol.json"):
            shutil.copy2(source / filename, output / filename)
        write_json_file(output / "plan.json", replay_plan)
        new_hash = file_fingerprint(output / "plan.json")
        for seed, selection, path, state in controls:
            imported = copy.deepcopy(state)
            imported["reused_control"] = {
                "source_state": str(path),
                "source_state_sha256": file_fingerprint(path),
                "original_identity": copy.deepcopy(state["identity"]),
                "scope": "Existing measured result; no training repeated. Adapter remains at original path.",
            }
            imported["identity"]["plan_sha256"] = new_hash
            write_json_file(
                output / f"seed_{seed}/{selection}__replay_off/state.json", imported
            )
        return {
            "status": "prepared",
            "seeds": list(seeds),
            "total_runs": 9 * len(seeds),
            "reused_completed_controls": 3 * len(seeds),
            "new_training_runs": 6 * len(seeds),
            "updates_per_new_run": plan["settings"]["updates"],
            "tokens_by_seed": replay_plan["tokens_by_seed"],
            "provider_calls": 0,
            "training_started": False,
        }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42])
    args = parser.parse_args()
    print(json.dumps(prepare(args.source, args.output, args.seeds), indent=2))


if __name__ == "__main__":
    main()
