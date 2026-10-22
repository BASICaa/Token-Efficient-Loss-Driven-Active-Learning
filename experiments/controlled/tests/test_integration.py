import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from pydantic import ValidationError

from TEFLD.src.experiment_settings import TrainingSettings
from TEFLD.src.helper import read_json_file, write_json_file
from test_runner import fixture_runner  # noqa: F401


def test_configuration_defaults_match_completed_study_and_reject_nonfinite():
    settings = TrainingSettings.from_protocol({"common_training": {}})
    assert settings.lora_rank == 4 and settings.epochs_per_round == 3
    assert (
        settings.learning_rate == 0.0002 and settings.gradient_accumulation_steps == 4
    )
    for change in (
        {"learning_rate": float("nan")},
        {"lora_rank": 0},
        {"lora_dropout": 1},
    ):
        with pytest.raises(ValidationError):
            TrainingSettings.model_validate(change)
    with pytest.raises(ValueError, match="micro_batch_size"):
        TrainingSettings.from_protocol({"common_training": {"micro_batch_size": 2}})
    with pytest.raises(ValueError, match="optimizer contract"):
        TrainingSettings.from_protocol(
            {"common_training": {"optimizer_contract": {"adam_beta1": 0.1}}}
        )


def test_configured_accumulation_is_used_and_round_events_reconcile(fixture_runner):
    runner = fixture_runner("configured")
    runner.protocol["common_training"]["gradient_accumulation_steps"] = 2
    runner.training_settings = TrainingSettings.from_protocol(runner.protocol)
    original = runner._optimizer_group
    sizes = []

    def record(model, optimizer, group):
        sizes.append(len(group))
        assert optimizer.param_groups[0]["initial_lr"] == 0.0002
        return original(model, optimizer, group)

    runner._optimizer_group = record
    result = runner.run("dolly_baseline", 42, budget=150)
    assert max(sizes) == 2
    events = read_json_file(
        runner.run_directory("pilot", "dolly_baseline", 42) / "round_events.json", []
    )
    assert len(events) == result["round"]
    assert events[-1]["answer_token_exposures"] == result["tokens"]
    assert events[-1]["unique_training_examples"] == len(result["ledger"])


def test_export_existing_admissions_preserves_verified_answers(tmp_path):
    from export_candidate_bank import export_bank
    from test_generation_recovery import sample

    item = sample()
    item.verification = {
        "status": "accepted_probabilistic_review",
        "result": {
            key: True for key in ("correct", "supported", "format_valid", "unambiguous")
        },
    }
    source, destination = tmp_path / "admissions.json", tmp_path / "bank.json"
    write_json_file(
        source,
        {
            "slot": {
                "teacher_identity": "teacher-fixture",
                "sample": item.model_dump(mode="json"),
            }
        },
    )
    result = export_bank(source, destination)
    assert result["candidates"] == 1 and result["provider_calls"] == 0
    assert read_json_file(destination, [])[0]["gold_summary"] == item.gold_summary
    assert export_bank(source, destination) == result
    item.verification["result"]["correct"] = False
    write_json_file(
        source,
        {
            "slot": {
                "teacher_identity": "teacher-fixture",
                "sample": item.model_dump(mode="json"),
            }
        },
    )
    with pytest.raises(ValueError, match="passing verifier"):
        export_bank(source, tmp_path / "invalid.json")


def test_portable_preparation_uses_only_training_warmup_and_separates_validation(
    tmp_path, monkeypatch
):
    import transformers
    import run_controlled
    import TEFLD.src.experiment_runner as module
    from test_runner import FixtureData, SavedTokenizer

    original_data = FixtureData

    class Data(original_data):
        def __init__(self, path):
            super().__init__(path)
            self.protocol.update(json.loads(Path(path).read_text(encoding="utf-8")))

    monkeypatch.setattr(module, "SplitData", Data)
    monkeypatch.setattr(
        transformers.AutoTokenizer, "from_pretrained", lambda *a, **k: SavedTokenizer()
    )
    monkeypatch.setattr(
        module.Teacher, "ask", lambda *a, **k: pytest.fail("Unexpected network call")
    )
    root = tmp_path / "study"
    report = run_controlled.prepare(
        root,
        "local-validation",
        tmp_path / "data.jsonl",
        tmp_path / "model",
        full_validation=True,
    )
    assert report["provider_calls"] == 0
    protocol = read_json_file(root / "protocol.json", {})
    assert protocol["controlled_entry"]["per_category"] is None
    assert protocol["example_count_control"]["unique_training_examples"] == 100
    warmup = read_json_file(root / "shared_initial_seed_42.json", [])
    assert all(item["verification"]["role"] == "training_pool" for item in warmup)
    plan = read_json_file(root / "study_plan.json", {})
    assert not set(plan["curriculum_validation_rows"]) & set(
        plan["comparison_validation_rows"]
    )
    assert not read_json_file(root / "study_budget.json", {})["approved"]
    assert (
        run_controlled.prepare(
            root,
            "local-validation",
            tmp_path / "data.jsonl",
            tmp_path / "model",
            full_validation=True,
        )
        == report
    )
