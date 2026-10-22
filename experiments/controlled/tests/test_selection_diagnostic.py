import copy
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import load_file

import run_selection_diagnostic as diagnostic
from TEFLD.src.helper import read_json_file, write_json_file
from TEFLD.src.runtime import set_seed
from TEFLD.src.token_budget import optimizer_group
from test_correctness import TinyLM, Tokenizer, sample
from test_runner import FixtureData, SavedTiny, SavedTokenizer


@pytest.fixture(scope="module", autouse=True)
def tiny_cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def records(count=80):
    return {
        f"id{i:04d}": {
            "answer_tokens": 2 + i % 4,
            "category": f"cat{i % 3}",
            "origin": "fixture",
        }
        for i in range(count)
    }


def test_schedule_matches_full_answers_budget_and_seed():
    bank = records()
    first = diagnostic.make_schedule(bank, 42, 12, 4, 2)
    assert first == diagnostic.make_schedule(bank, 42, 12, 4, 2)
    assert first != diagnostic.make_schedule(bank, 43, 12, 4, 2)
    assert len(first) == 12 and all(len(x) == 4 for x in first)
    totals = []
    warmups = []
    for selection in diagnostic.SELECTIONS:
        for replay in diagnostic.REPLAYS:
            seen, trace, ids = set(), [], []
            for update, slots in enumerate(first):
                selected, detail = diagnostic.select_group(
                    bank,
                    slots,
                    seen,
                    selection,
                    replay,
                    {"cat0": 2, "cat1": 1, "cat2": 0.1},
                    {i: 1.0 for i in bank},
                    seed=update,
                )
                assert len(selected) == len(set(selected))
                assert all(
                    bank[i]["answer_tokens"] == s["length"]
                    for i, s in zip(selected, slots)
                )
                assert all((x["candidate_id"] in seen) == x["replay"] for x in detail)
                trace.extend(detail)
                ids.extend(selected)
                seen.update(selected)
            totals.append(sum(x["answer_tokens"] for x in trace))
            warmups.append(ids[:8])
            if replay == "off":
                assert len(seen) == 48
            else:
                assert any(x["replay"] for x in trace)
    assert len(set(totals)) == 1 and all(x == warmups[0] for x in warmups)


def test_random_selection_ignores_feedback_and_guided_replay_ranks_training_loss():
    bank = records()
    schedule = diagnostic.make_schedule(bank, 42, 5, 4, 1)
    seen = {x["warmup_id"] for x in schedule[0]}
    args = (bank, schedule[1], seen, "random", "off")
    one = diagnostic.select_group(*args, {"cat0": 999}, {}, seed=7)
    two = diagnostic.select_group(*args, {"cat2": 999}, {}, seed=7)
    assert one == two
    import random

    assert (
        diagnostic.choose(
            ["a", "b"], "guided", {}, random.Random(1), {"a": 0.1, "b": 2.0}
        )
        == "b"
    )
    with pytest.raises(ValueError, match="fresh"):
        diagnostic.choose(["a"], "guided", {}, random.Random(1), {})


def test_schedule_refuses_insufficient_bank():
    with pytest.raises(ValueError, match="at least"):
        diagnostic.make_schedule(records(5), 42, 10, 4, 1)


def test_instrumented_update_matches_original_and_reports_health():
    from TEFLD.src.encoding import encode_sample

    set_seed(12)
    model = TinyLM()
    original = copy.deepcopy(model)
    encoded = [
        encode_sample(sample(gold_summary=answer), Tokenizer())
        for answer in ("Yes", "A longer answer")
    ]
    a = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0)
    b = torch.optim.AdamW(original.parameters(), lr=0.001, weight_decay=0)
    count, losses = optimizer_group(original, b, encoded)
    health = diagnostic.optimizer_step(model, a, encoded)
    assert health["answer_tokens"] == count and health["example_losses"] == losses
    assert health["gradient_norm_before_clipping"] >= 0
    assert health["learning_rate"] == 0.001
    for x, y in zip(model.parameters(), original.parameters()):
        torch.testing.assert_close(x, y, rtol=0, atol=0)


def test_nonfinite_loss_cannot_update_weights():
    from TEFLD.src.encoding import encode_sample

    model = TinyLM()
    before = copy.deepcopy(model.state_dict())
    model.forward = lambda **kw: SimpleNamespace(
        loss=next(model.parameters()).sum() * float("nan")
    )
    with pytest.raises(FloatingPointError):
        diagnostic.optimizer_step(
            model,
            torch.optim.AdamW(model.parameters()),
            [encode_sample(sample(), Tokenizer())],
        )
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, before[key], rtol=0, atol=0)


@pytest.fixture
def study(tmp_path, monkeypatch):
    import transformers

    access = []

    class Data(FixtureData):
        def __init__(self, path):
            super().__init__(path)
            self.protocol.update(read_json_file(path, {}))

        def sample(self, i, role):
            access.append(role)
            assert role != "final_test"
            return super().sample(i, role)

    monkeypatch.setattr(diagnostic, "SplitData", Data)
    monkeypatch.setattr(diagnostic, "model_identity", lambda *a: "fixture-model")
    monkeypatch.setattr(
        transformers.AutoTokenizer, "from_pretrained", lambda *a, **kw: SavedTokenizer()
    )

    def runtime(path, *, seed, adapter=None, **kw):
        set_seed(seed)
        model = SavedTiny()
        if adapter:
            model.load_state_dict(
                load_file(str(Path(adapter) / "adapter_model.safetensors"))
            )
        return model, SavedTokenizer()

    monkeypatch.setattr(diagnostic, "load_runtime", runtime)
    # Any accidental provider call fails the offline test immediately.
    from TEFLD.src.candidate_pool import Teacher

    monkeypatch.setattr(Teacher, "ask", lambda *a, **kw: pytest.fail("API access"))

    def prepare(name="study", lifecycle="reset-per-round", ablation=False):
        output = tmp_path / name
        diagnostic.prepare(
            output,
            tmp_path / "dataset",
            tmp_path / "model",
            seeds=[42],
            updates=5,
            warmup_updates=1,
            microbatches=2,
            round_updates=2,
            per_category=1,
            optimizer_lifecycle=lifecycle,
            replay_ablation=ablation,
        )
        return output

    return prepare, access


def test_end_to_end_matched_selection_and_diagnostics(study):
    from selection_report import report

    prepare, access = study
    output = prepare()
    states = [
        diagnostic.run_arm(output, selection, "off", 42)
        for selection in diagnostic.SELECTIONS
    ]
    assert all(x["status"] == "complete" for x in states)
    assert len({x["tokens"] for x in states}) == 1
    assert all(x["update"] == 5 and len(x["seen"]) == 10 for x in states)
    for state in states:
        assert state["evaluations"][0]["update"] == 0
        assert len(state["evaluations"]) == 4
        assert all(x["checkpoint"]["rows"] for x in state["evaluations"])
        assert sum(x["optimizer_reset"] for x in state["updates"]) == 3
    assert "final_test" not in access
    result = report(output, figures=True)
    assert result["complete_arms"] == 3
    assert (output / "learning_curves.png").stat().st_size > 1000
    assert (output / "optimizer_health.png").stat().st_size > 1000
    summary = read_json_file(output / "diagnostic_summary.json", {})
    assert len(summary["paired_vs_random"]) == 3
    assert summary["resource_checks"][0]["common_warmup"]
    assert diagnostic.run_arm(output, "random", "off", 42) == states[0]


@pytest.mark.parametrize("lifecycle", ["reset-per-round", "continuous"])
def test_interruption_resumes_identically_from_committed_checkpoint(
    study, monkeypatch, lifecycle
):
    prepare, _ = study
    clean_dir = prepare("clean", lifecycle)
    broken_dir = prepare("broken", lifecycle)
    clean = diagnostic.run_arm(clean_dir, "loss", "off", 42)
    original = diagnostic.optimizer_step
    calls = [0]

    def interrupted(*a, **kw):
        calls[0] += 1
        if calls[0] == 3:
            raise RuntimeError("fixture interruption")
        return original(*a, **kw)

    monkeypatch.setattr(diagnostic, "optimizer_step", interrupted)
    with pytest.raises(RuntimeError, match="interruption"):
        diagnostic.run_arm(broken_dir, "loss", "off", 42)
    failure = read_json_file(
        broken_dir / "seed_42/loss__replay_off/last_failure.json", {}
    )
    assert failure["last_committed_update"] == 1
    monkeypatch.setattr(diagnostic, "optimizer_step", original)
    resumed = diagnostic.run_arm(broken_dir, "loss", "off", 42)
    assert clean["updates"] == resumed["updates"]
    assert clean["selection_trace"] == resumed["selection_trace"]
    assert clean["tokens"] == resumed["tokens"]
    a, b = [
        load_file(str(Path(s["adapter"]) / "adapter_model.safetensors"))
        for s in (clean, resumed)
    ]
    for key in a:
        torch.testing.assert_close(a[key], b[key], rtol=0, atol=0)


def test_bank_tampering_rejected_before_loading_model(study, monkeypatch):
    prepare, _ = study
    output = prepare()
    with (output / "bank.json").open("a") as file:
        file.write(" ")
    monkeypatch.setattr(
        diagnostic, "load_runtime", lambda *a, **kw: pytest.fail("Loaded changed bank")
    )
    with pytest.raises(ValueError, match="changed"):
        diagnostic.run_arm(output, "random", "off", 42)


def test_replay_arms_separate_contrasts_and_equal_resources(study):
    from selection_report import report

    prepare, _ = study
    output = prepare(ablation=True)
    for selection in diagnostic.SELECTIONS:
        for replay in diagnostic.REPLAYS:
            diagnostic.run_arm(output, selection, replay, 42)
    report(output, figures=False)
    summary = read_json_file(output / "diagnostic_summary.json", {})
    assert len(summary["results"]) == 9 and len(summary["replay_contrasts"]) == 6
    assert len({x["tokens"] for x in summary["results"]}) == 1
    assert len({x["updates"] for x in summary["results"]}) == 1
    assert all(
        x["replay_presentations"] > 0
        for x in summary["results"]
        if not x["arm"].endswith("off")
    )


def test_report_detects_resource_contract_violation(study):
    from selection_report import report

    prepare, _ = study
    output = prepare()
    diagnostic.run_arm(output, "random", "off", 42)
    path = output / "seed_42/random__replay_off/state.json"
    state = read_json_file(path, {})
    state["tokens"] += 1
    write_json_file(path, state)
    with pytest.raises(ValueError, match="matched resources"):
        report(output, figures=False)


def test_existing_report_uses_saved_metrics_and_marks_missing_data(tmp_path):
    from selection_report import historical_report

    checkpoint = tmp_path / "source/checkpoint"
    write_json_file(checkpoint / "validation.json", {"macro_nll": 1.5})
    state_path = tmp_path / "source/state.json"
    write_json_file(
        state_path,
        {
            "final_checkpoint": {"path": str(checkpoint)},
            "status": "complete",
            "tokens": 10,
            "config": {"seed": 42},
            "rounds": [{"round": 0, "tokens": 10}],
            "ledger": [{"id": "a", "sample": {"tag": "general_qa"}}],
            "losses": [
                {
                    "round": 0,
                    "tokens_after_group": 10,
                    "candidate_id": "a",
                    "answer_tokens": 10,
                    "source": "recycled",
                }
            ],
        },
    )
    write_json_file(tmp_path / "source/curriculum_0000.json", {"macro_nll": 1.4})
    comparison = tmp_path / "comparison.json"
    write_json_file(
        comparison, {"Results": [{"Condition": "fixture", "State": str(state_path)}]}
    )
    comparison.write_text(comparison.read_text(encoding="utf-8"), encoding="utf-8-sig")
    before = state_path.read_bytes()
    destination = tmp_path / "report"
    historical_report(comparison, destination, figures=False)
    saved = read_json_file(destination / "existing_run_diagnostics.json", {})
    assert saved["new_model_evaluations"] == 0 and saved["new_provider_calls"] == 0
    assert saved["results"][0]["replay_tokens"] == 10
    assert saved["results"][0]["category_tokens"] == {"general_qa": 10}
    assert "Untrained baseline score" in saved["results"][0]["unavailable"]
    assert state_path.read_bytes() == before
