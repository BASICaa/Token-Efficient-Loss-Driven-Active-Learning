from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file, save_file

from TEFLD.src import experiment_runner as module
from TEFLD.src.dataschema import Training_Sample
from TEFLD.src.helper import read_json_file
from test_correctness import TinyLM, Tokenizer

CATEGORIES = [
    "brainstorming",
    "classification",
    "closed_qa",
    "creative_writing",
    "general_qa",
    "information_extraction",
    "open_qa",
    "summarization",
]


class FixtureData:
    def __init__(self, path):
        self.protocol_path = Path(path)
        self.protocol = {
            "dataset": {"categories": CATEGORIES},
            "common_training": {
                "model_path": "fixture",
                "max_length": 512,
                "epochs_per_round": 3,
            },
            "checkpoint_selection": {"nominal_supervised_token_milestones": [25, 50]},
            "tefld_controls": {
                "teacher_model_id": "fixture",
                "teacher_temperature": 0.7,
            },
        }
        self.rows = [
            {
                "instruction": f"Question {i}?",
                "context": "",
                "response": "A" * (i % 5 + 1),
                "category": CATEGORIES[i % 8],
            }
            for i in range(264)
        ]
        self.ids = {
            "training_pool": set(range(240)),
            "curriculum_validation": set(range(240, 248)),
            "checkpoint_validation": set(range(248, 256)),
            "final_test": set(range(256, 264)),
        }
        self.manifest = {"excluded": {"historical_and_connected": []}}
        self.access = []

    def sample(self, i, role):
        self.access.append(role)
        assert i in self.ids[role]
        row = self.rows[i]
        return Training_Sample(
            instruct=row["instruction"],
            sample="",
            gold_summary=row["response"],
            tag=row["category"],
            source="generated",
        )

    def samples(self, role, per_category=None):
        return [(i, self.sample(i, role)) for i in sorted(self.ids[role])]


class SavedTiny(TinyLM):
    def save_pretrained(self, path, **kwargs):
        save_file(self.state_dict(), str(Path(path) / "adapter_model.safetensors"))


class SavedTokenizer(Tokenizer):
    def save_pretrained(self, path):
        (Path(path) / "tokenizer.json").write_text("{}")


@pytest.fixture
def fixture_runner(tmp_path, monkeypatch):
    protocol = tmp_path / "protocol.json"
    protocol.write_text("{}")
    monkeypatch.setattr(module, "SplitData", FixtureData)
    monkeypatch.setattr(
        module.ExperimentRunner, "model_identity", lambda self: "fixture-model"
    )

    def load_runtime(path, seed=42, adapter=None, **kwargs):
        torch.manual_seed(seed)
        model = SavedTiny()
        if adapter:
            model.load_state_dict(
                load_file(str(Path(adapter) / "adapter_model.safetensors"))
            )
        return model, SavedTokenizer()

    monkeypatch.setattr(module, "load_runtime", load_runtime)
    monkeypatch.setattr(
        module.Teacher, "ask", lambda *a, **k: pytest.fail("Unexpected paid API call.")
    )
    return lambda name: module.ExperimentRunner(protocol, tmp_path / name)


def test_runner_exact_budget_checkpoint_selection_and_completed_resume(fixture_runner):
    runner = fixture_runner("normal")
    result = runner.run("dolly_baseline", 42, budget=77)
    assert result["status"] == "complete" and result["tokens"] == 77
    assert result["rounds"][-1]["partial_final_round"]
    assert len(result["candidates"]) == 3
    assert runner.run("dolly_baseline", 42, budget=77) == result
    assert "curriculum_validation" not in runner.data.access
    assert "final_test" not in runner.data.access


def test_interrupted_training_restarts_from_last_committed_state(
    fixture_runner, monkeypatch
):
    clean = fixture_runner("clean").run("dolly_baseline", 42, budget=77)
    runner = fixture_runner("interrupted")
    original = module.optimizer_group
    calls = 0

    def fail_after_one(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("simulated training interruption")
        return original(*args, **kwargs)

    monkeypatch.setattr(module, "optimizer_group", fail_after_one)
    with pytest.raises(RuntimeError, match="interruption"):
        runner.run("dolly_baseline", 42, budget=77)
    monkeypatch.setattr(module, "optimizer_group", original)
    recovered = runner.run("dolly_baseline", 42, budget=77)
    assert recovered["tokens"] == clean["tokens"]
    assert recovered["losses"] == clean["losses"]
    a = load_file(str(Path(clean["last_adapter"]) / "adapter_model.safetensors"))
    b = load_file(str(Path(recovered["last_adapter"]) / "adapter_model.safetensors"))
    for key in a:
        torch.testing.assert_close(a[key], b[key], rtol=0, atol=0)


@pytest.mark.parametrize("method", ["tefld_validation", "tefld_loss"])
def test_fixed_bank_feedback_isolation_without_teacher_calls(fixture_runner, method):
    runner = fixture_runner(method)
    bank = [
        runner.data.sample(i, "training_pool").model_dump(mode="json")
        for i in sorted(runner.data.ids["training_pool"])
    ]
    result = runner.run(method, 42, budget=150, candidate_bank=bank)
    assert result["tokens"] == 150
    assert ("curriculum_validation" in runner.data.access) == (
        method == "tefld_validation"
    )
    assert "final_test" not in runner.data.access


def test_final_test_stays_locked_when_any_primary_run_is_incomplete(fixture_runner):
    runner = fixture_runner("locked")
    runner.freeze(["dolly_baseline"], [42], 77)
    with pytest.raises(PermissionError, match="incomplete"):
        runner.evaluate_final()
    assert "final_test" not in runner.data.access


def test_primary_training_requires_frozen_settings(fixture_runner):
    runner = fixture_runner("primary_guard")
    with pytest.raises(PermissionError, match="Freeze"):
        runner.run("dolly_baseline", 42, phase="primary", budget=77, per_category=None)
    runner.freeze(["dolly_baseline"], [42], 77)
    with pytest.raises(PermissionError, match="settings"):
        runner.run("dolly_baseline", 123, phase="primary", budget=77, per_category=None)
    state = runner.run(
        "dolly_baseline", 42, phase="primary", budget=77, per_category=None
    )
    assert state["tokens"] == 77


def test_missing_hybrid_head_does_not_create_run_state(fixture_runner):
    runner = fixture_runner("missing_head")
    with pytest.raises(ValueError, match="predictor"):
        runner.run("tefld_benefit", 42, budget=77)
    assert not (
        runner.run_directory("pilot", "tefld_benefit", 42) / "state.json"
    ).exists()


def test_unpromoted_hybrid_cannot_enter_frozen_main_study(fixture_runner):
    runner = fixture_runner("unpromoted")
    with pytest.raises(PermissionError, match="promotion"):
        runner.freeze(["tefld_benefit"], [42], 77)


def test_resume_code_amendment_keeps_committed_adapter_and_tokens(
    fixture_runner, monkeypatch
):
    runner = fixture_runner("code_repair")
    # Interrupt after the first committed round, before next batch admission.
    original = runner._batch
    calls = 0

    def fail(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("generation interruption")
        return original(*args, **kwargs)

    monkeypatch.setattr(runner, "_batch", fail)
    with pytest.raises(RuntimeError):
        runner.run("dolly_baseline", 42, budget=1000)
    state_path = runner.run_directory("pilot", "dolly_baseline", 42) / "state.json"
    before = read_json_file(state_path, {})
    monkeypatch.setattr(runner, "_batch", original)
    runner.code_hash = "patched-fixture-code"
    with pytest.raises(ValueError, match="identity"):
        runner.run("dolly_baseline", 42, budget=1000)
    result = runner.run("dolly_baseline", 42, budget=1000, resume_pilot_after_fix=True)
    assert result["tokens"] == 1000
    assert result["pilot_code_amendments"][0]["tokens"] == before["tokens"] > 0
    assert len(result["rounds"]) > len(before["rounds"])


def test_code_amendment_cannot_change_training_budget(fixture_runner):
    runner = fixture_runner("wrong_settings")
    runner.run("dolly_baseline", 42, budget=77)
    runner.code_hash = "changed"
    with pytest.raises(ValueError, match="identity"):
        runner.run("dolly_baseline", 42, budget=78, resume_pilot_after_fix=True)


def dataset_runner(fixture_runner, name):
    from TEFLD.src.dataset_curriculum import dataset_protocol

    runner = fixture_runner(name)
    runner.protocol = dataset_protocol(runner.protocol, "both")
    runner.data.protocol = runner.protocol
    return runner


@pytest.mark.parametrize("method", ["tefld_dataset_loss", "tefld_dataset_validation"])
def test_dataset_tefld_matches_baseline_warmup_and_isolates_feedback(
    fixture_runner, method
):
    baseline = dataset_runner(fixture_runner, "dataset_baseline_" + method)
    control = dataset_runner(fixture_runner, method)
    baseline_state = baseline.run("dolly_baseline", 42, budget=350)
    result = control.run(method, 42, budget=350)
    assert result["tokens"] == baseline_state["tokens"] == 350
    assert result["config"]["source_mode"] == "adaptive_dolly_training_pool"
    assert result["dataset_control"]["provider_calls"] == 0
    assert len(result["rounds"]) > 1
    base_batch = read_json_file(
        baseline.run_directory("pilot", "dolly_baseline", 42) / "batch_0000.json", []
    )
    adaptive_batch = read_json_file(
        control.run_directory("pilot", method, 42) / "batch_0000.json", []
    )
    assert base_batch == adaptive_batch
    for entry in result["ledger"]:
        sample = entry["sample"]
        metadata = sample["verification"]
        assert metadata["role"] == "training_pool"
        row_id = metadata["row_id"]
        assert row_id in control.data.ids["training_pool"]
        assert sample["gold_summary"] == control.data.rows[row_id]["response"]
    assert any(x["source"] == "recycled" for x in result["presentations"])
    assert ("curriculum_validation" in control.data.access) == method.endswith(
        "_validation"
    )
    assert "curriculum_validation" not in baseline.data.access
    assert "final_test" not in control.data.access + baseline.data.access
    assert result["dataset_selection_traces"][1]["focus_quotas"]


def test_dataset_policy_suppresses_network_even_when_runner_was_enabled(
    fixture_runner, monkeypatch
):
    runner = dataset_runner(fixture_runner, "dataset_network")
    runner.teacher.allow_network = runner.verifier.allow_network = (
        runner.jev.allow_network
    ) = True
    original = runner._batch

    def batch(*args, **kwargs):
        assert not runner.teacher.allow_network
        assert not runner.verifier.allow_network
        assert not runner.jev.allow_network
        return original(*args, **kwargs)

    monkeypatch.setattr(runner, "_batch", batch)
    monkeypatch.setattr(
        runner.jev, "features", lambda *a, **k: pytest.fail("Unexpected Jev call")
    )
    runner.run("tefld_dataset_loss", 42, budget=100)
    assert (
        runner.teacher.allow_network
        and runner.verifier.allow_network
        and runner.jev.allow_network
    )


def test_dataset_resume_rejects_rewritten_cached_reference(fixture_runner, monkeypatch):
    runner = dataset_runner(fixture_runner, "dataset_forged")

    def interrupted(*args, **kwargs):
        raise RuntimeError("training interruption")

    monkeypatch.setattr(runner, "_optimizer_group", interrupted)
    with pytest.raises(RuntimeError, match="interruption"):
        runner.run("tefld_dataset_loss", 42, budget=350)
    path = runner.run_directory("pilot", "tefld_dataset_loss", 42) / "state.json"
    state = read_json_file(path, {})
    state["batch"][0]["gold_summary"] = "An altered synthetic answer"
    module.write_json_file(path, state)
    with pytest.raises(ValueError, match="original reference"):
        runner.run("tefld_dataset_loss", 42, budget=350)


@pytest.mark.parametrize("method", ["tefld_dataset_loss", "tefld_dataset_validation"])
def test_dataset_tefld_round_recovery_matches_uninterrupted(
    fixture_runner, monkeypatch, method
):
    clean = dataset_runner(fixture_runner, "clean_" + method).run(
        method, 42, budget=350
    )
    runner = dataset_runner(fixture_runner, "broken_" + method)
    original = runner._batch
    calls = 0

    def interrupted(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("admission interruption")
        return original(*args, **kwargs)

    monkeypatch.setattr(runner, "_batch", interrupted)
    with pytest.raises(RuntimeError, match="interruption"):
        runner.run(method, 42, budget=350)
    monkeypatch.setattr(runner, "_batch", original)
    recovered = runner.run(method, 42, budget=350)
    assert recovered["losses"] == clean["losses"]
    assert recovered["dataset_selection_traces"] == clean["dataset_selection_traces"]
    for key, value in load_file(
        str(Path(clean["last_adapter"]) / "adapter_model.safetensors")
    ).items():
        torch.testing.assert_close(
            value,
            load_file(
                str(Path(recovered["last_adapter"]) / "adapter_model.safetensors")
            )[key],
            rtol=0,
            atol=0,
        )


def test_dataset_arm_requires_declared_source_policy_and_rejects_external_bank(
    fixture_runner,
):
    runner = fixture_runner("dataset_missing_policy")
    with pytest.raises(ValueError, match="dataset-TEFLD protocol"):
        runner.run("tefld_dataset_loss", 42, budget=350)
    runner = dataset_runner(fixture_runner, "dataset_unexpected_bank")
    with pytest.raises(ValueError, match="original training pool"):
        runner.run("tefld_dataset_loss", 42, budget=350, candidate_bank=[])
    with pytest.raises(ValueError, match="original training pool"):
        runner.run("tefld_dataset_loss", 42, budget=350, probe_callback=lambda *a: None)
    with pytest.raises(PermissionError, match="optimizer"):
        runner.freeze(["tefld_dataset_loss", "dolly_baseline"], [42], 350)
    runner.protocol["dataset_tefld"]["primary_ready"] = True
    runner.freeze(["tefld_dataset_loss", "dolly_baseline"], [42], 350)
