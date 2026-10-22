"""Small, recoverable round summaries; canonical training state stays authoritative."""

from collections import Counter

from .helper import write_json_file


def write_round_events(directory, state):
    events = []
    for round_record in state["rounds"]:
        round_id = round_record["round"]
        losses = [row for row in state["losses"] if row["round"] <= round_id]
        events.append(
            {
                "round": round_id,
                "unique_training_examples": sum(
                    entry["first_round"] <= round_id for entry in state["ledger"]
                ),
                "answer_token_exposures": round_record["tokens"],
                "optimizer_updates": len(
                    {(row["round"], row["tokens_after_group"]) for row in losses}
                ),
                "round_presentations": sum(row["round"] == round_id for row in losses),
                "round_training_categories": dict(
                    Counter(
                        entry["sample"]["tag"]
                        for entry in state["ledger"]
                        if entry["first_round"] == round_id
                    )
                ),
            }
        )
    write_json_file(directory / "round_events.json", events)
