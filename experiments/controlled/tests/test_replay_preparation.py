import pytest

import prepare_replay_study as replay
from TEFLD.src.helper import read_json_file, write_json_file
from TEFLD.src.runtime import file_fingerprint


def source_study(tmp_path, monkeypatch):
    source = tmp_path / "source"
    monkeypatch.setattr(replay, "code_identity", lambda: "fixture-code")
    monkeypatch.setattr(replay, "require_adapter", lambda path: {"complete": True})
    write_json_file(source / "protocol.json", {"fixture": True})
    write_json_file(source / "bank.json", {"fixture": True})
    plan = {
        "code_sha256": "fixture-code",
        "protocol_sha256": file_fingerprint(source / "protocol.json"),
        "bank_sha256": file_fingerprint(source / "bank.json"),
        "settings": {"seeds": [42, 43], "replays": ["off"], "updates": 2},
        "schedules": {
            str(s): [
                [{"length": 4, "warmup_id": "initial", "replay_slot": False}],
                [{"length": 4, "warmup_id": None, "replay_slot": True}],
            ]
            for s in [42, 43]
        },
        "tokens_by_seed": {"42": 8, "43": 8},
        "checkpoint_rows": [10],
        "curriculum_rows": [20],
    }
    write_json_file(source / "plan.json", plan)
    digest = file_fingerprint(source / "plan.json")
    for seed in [42, 43]:
        for selection in replay.SELECTIONS:
            state = {
                "identity": {
                    "plan_sha256": digest,
                    "selection": selection,
                    "replay": "off",
                    "seed": seed,
                },
                "status": "complete",
                "tokens": 8,
                "update": 2,
                "seen": ["initial", "new"],
                "adapter": "fixture",
                "updates": [{"update": i, "answer_tokens": 4} for i in [1, 2]],
                "selection_trace": [
                    {
                        "update": i,
                        "answer_tokens": 4,
                        "candidate_id": c,
                        "replay": False,
                    }
                    for i, c in [(1, "initial"), (2, "new")]
                ],
                "evaluations": [
                    {
                        "update": i,
                        "checkpoint": {"rows": [{"row_id": 10}]},
                        "curriculum": {"rows": [{"row_id": 20}]},
                    }
                    for i in [0, 2]
                ],
            }
            write_json_file(
                source / f"seed_{seed}/{selection}__replay_off/state.json", state
            )
    return source


def test_fork_preserves_controls_and_freezes_identical_schedule(tmp_path, monkeypatch):
    source = source_study(tmp_path, monkeypatch)
    snapshots = {path: path.read_bytes() for path in source.rglob("*.json")}
    destination = tmp_path / "replay"
    result = replay.prepare(source, destination)
    assert result["new_training_runs"] == 6 and result["reused_completed_controls"] == 3
    assert result["training_started"] is False and result["provider_calls"] == 0
    old, new = [
        read_json_file(root / "plan.json", {}) for root in (source, destination)
    ]
    assert new["schedules"]["42"] == old["schedules"]["42"]
    assert new["settings"]["seeds"] == [42] and new["settings"]["replays"] == [
        "off",
        "random",
        "guided",
    ]
    for selection in replay.SELECTIONS:
        imported = read_json_file(
            destination / f"seed_42/{selection}__replay_off/state.json", {}
        )
        original = read_json_file(
            source / f"seed_42/{selection}__replay_off/state.json", {}
        )
        assert (
            imported["tokens"] == original["tokens"]
            and imported["evaluations"] == original["evaluations"]
        )
        assert imported["reused_control"]["original_identity"] == original["identity"]
        assert imported["identity"]["plan_sha256"] == file_fingerprint(
            destination / "plan.json"
        )
    assert all(path.read_bytes() == data for path, data in snapshots.items())
    with pytest.raises(ValueError, match="exists"):
        replay.prepare(source, destination)


@pytest.mark.parametrize(
    "change",
    [
        "status",
        "tokens",
        "warmup",
        "replay",
        "evaluation",
        "code",
        "adapter",
        "optimizer",
        "initial",
    ],
)
def test_incompatible_reuse_fails_before_output(tmp_path, monkeypatch, change):
    source = source_study(tmp_path, monkeypatch)
    path = source / "seed_42/loss__replay_off/state.json"
    state = read_json_file(path, {})
    if change == "status":
        state["status"] = "running"
    if change == "tokens":
        state["tokens"] = 7
    if change == "warmup":
        state["selection_trace"][0]["candidate_id"] = "different"
    if change == "replay":
        state["selection_trace"][1]["replay"] = True
    if change == "evaluation":
        state["evaluations"][-1]["checkpoint"]["rows"][0]["row_id"] = 99
    if change == "optimizer":
        state["updates"][0]["learning_rate"] = 0.123
    if change == "initial":
        state["evaluations"][0]["checkpoint"]["macro_nll"] = 123
    if change == "code":
        monkeypatch.setattr(replay, "code_identity", lambda: "changed")
    if change == "adapter":

        def fail(path):
            raise ValueError("Adapter changed")

        monkeypatch.setattr(replay, "require_adapter", fail)
    write_json_file(path, state)
    destination = tmp_path / "replay"
    with pytest.raises(ValueError):
        replay.prepare(source, destination)
    assert not destination.exists()


def test_expansion_reuses_each_seed_controls(tmp_path, monkeypatch):
    source = source_study(tmp_path, monkeypatch)
    result = replay.prepare(source, tmp_path / "all", seeds=[42, 43])
    assert result["new_training_runs"] == 12 and result["total_runs"] == 18
