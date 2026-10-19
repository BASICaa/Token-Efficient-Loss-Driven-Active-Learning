import copy
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from TEFLD.src.dataschema import Pipeline_State, Training_Sample
from TEFLD.src.encoding import (
    encode_sample,
    render_prompt,
    supervised_tokens,
    collate_answer_only,
)
from TEFLD.src.token_budget import TokenBudget, optimizer_group
from TEFLD.src.training_safety import install_gradient_guards
from TEFLD.src.validation import validate_sample, calculate


class Tokenizer:
    chat_template = None
    bos_token = "^"
    eos_token = "$"
    pad_token_id = 0

    def __call__(self, text, **kwargs):
        return {"input_ids": [ord(c) % 31 + 1 for c in text]}


def sample(**overrides):
    values = dict(
        sample="",
        instruct="Answer briefly.",
        gold_summary="Yes",
        tag="general_qa",
        source="generated",
    )
    values.update(overrides)
    return Training_Sample(**values)


class TinyLM(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = torch.nn.Embedding(40, 7)
        self.head = torch.nn.Linear(7, 40)

    def forward(self, input_ids, attention_mask, labels):
        logits = self.head(self.embed(input_ids))
        loss = torch.nn.functional.cross_entropy(
            logits[:, :-1].reshape(-1, 40), labels[:, 1:].reshape(-1), ignore_index=-100
        )
        return SimpleNamespace(loss=loss, logits=logits)


def test_complete_encoding_and_prompt_mask():
    t = Tokenizer()
    s = sample()
    encoded = encode_sample(s, t, 512)
    prefix = t(render_prompt(s, t))["input_ids"]
    assert encoded["input_ids"][: len(prefix)].tolist() == prefix
    assert encoded["labels"][: len(prefix)].eq(-100).all()
    assert supervised_tokens(encoded) == len(t("Yes$")["input_ids"])
    with pytest.raises(ValueError, match="truncation"):
        encode_sample(sample(instruct="x" * 512), t, 512)


def test_native_chat_suffix_and_prefix_match():
    class Chat(Tokenizer):
        chat_template = "test"

        def apply_chat_template(self, messages, **kwargs):
            prefix = "^user:" + messages[0]["content"] + "|model:"
            return (
                prefix
                if len(messages) == 1
                else prefix + messages[1]["content"] + "|end|"
            )

    t = Chat()
    encoded = encode_sample(sample(), t)
    assert supervised_tokens(encoded) == len("Yes|end|")


@pytest.mark.parametrize(
    "fields,reason",
    [
        (
            {
                "gold_summary": "The bus leaves at 8:00 a.m. It leaves 10 minutes before the event begins at 8:30 a.m."
            },
            "incorrect_time_interval",
        ),
        (
            {
                "sample": "Workshop A lasts 45 minutes and starts at 10:20 a.m. Workshop B lasts 1 hour 10 minutes and starts at 9:55 a.m.",
                "gold_summary": "Workshop A ends later.",
            },
            "incorrect_workshop_end_comparison",
        ),
        (
            {
                "instruct": "Return only JSON with the answer.",
                "gold_summary": "answer=4",
            },
            "invalid_required_json",
        ),
    ],
)
def test_regression_wrong_targets_are_rejected_without_private_ledger(fields, reason):
    s = sample(**fields)
    assert reason in validate_sample(s).errors


def test_correct_tie_and_format_are_accepted():
    s = sample(
        instruct="Compare the workshops in a bullet list.",
        sample="Workshop A lasts 45 minutes and starts at 10:20 a.m. Workshop B lasts 1 hour 10 minutes and starts at 9:55 a.m.",
        gold_summary="- Both end at 11:05 a.m.; neither ends later.",
    )
    assert validate_sample(s).accepted
    s = sample(
        instruct='Give JSON with two keys: "recommendation" and "reason".',
        gold_summary='{"recommendation":"Share a tracker","reason":"Coordination"}',
    )
    assert validate_sample(s).accepted
    assert not validate_sample(
        sample(instruct="Return JSON.", gold_summary='{"x":NaN}')
    ).accepted
    with pytest.raises(ValueError):
        calculate('__import__("os").system("bad")')


def test_nonfinite_gradient_cannot_change_parameters():
    model = torch.nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        model.weight.zero_()
    install_gradient_guards(model)
    before = model.weight.detach().clone()
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    with pytest.raises(FloatingPointError):
        model(torch.ones(1, 1)).sqrt().sum().backward()
        opt.step()
    assert torch.equal(before, model.weight)


def test_accumulation_matches_combined_batch_with_unequal_answers():
    torch.manual_seed(3)
    a = TinyLM()
    b = copy.deepcopy(a)
    data = [
        encode_sample(sample(gold_summary=x), Tokenizer())
        for x in ["A", "A longer answer"]
    ]
    oa = torch.optim.SGD(a.parameters(), lr=0.01)
    ob = torch.optim.SGD(b.parameters(), lr=0.01)
    optimizer_group(a, oa, data, max_grad_norm=1000)
    b(**collate_answer_only(data, 0)).loss.backward()
    ob.step()
    for pa, pb in zip(a.parameters(), b.parameters()):
        torch.testing.assert_close(pa, pb, atol=1e-7, rtol=1e-6)


def test_budget_counts_repeats_and_preserves_final_positive_targets():
    encoded = encode_sample(sample(gold_summary="abcd"), Tokenizer())
    budget = TokenBudget(12)
    counts = []
    while budget.remaining:
        item = budget.admit(encoded)
        n = supervised_tokens(item)
        assert n > 0
        budget.commit(n)
        counts.append(n)
    assert counts == [5, 5, 2] and budget.consumed == 12
    assert budget.admit(encoded) is None


def test_atomic_json_failure_preserves_previous_file(tmp_path):
    from TEFLD.src.helper import write_json_file

    p = tmp_path / "state.json"
    write_json_file(p, {"valid": 1})
    with pytest.raises(ValueError):
        write_json_file(p, {"loss": float("nan")})
    assert json.loads(p.read_text()) == {"valid": 1}


def test_transaction_recovers_every_file_boundary(tmp_path, monkeypatch):
    from TEFLD.src import helper, transactions

    actual = helper.write_json_file
    bundle = {
        "OverAllData.json": [{"ledger_id": 1}],
        "failure_vault.json": {},
        "round_result_000000.json": {"round_id": 0},
        "pipeline_state.json": {"current_round": 1},
    }
    for boundary in range(2, 7):
        directory = tmp_path / str(boundary)
        calls = 0

        def injected(path, data):
            nonlocal calls
            calls += 1
            if calls == boundary:
                raise OSError("simulated interruption")
            actual(path, data)

        monkeypatch.setattr(helper, "write_json_file", injected)
        with pytest.raises(OSError):
            transactions.commit(directory, bundle, "round_result_000000.json")
        monkeypatch.setattr(helper, "write_json_file", actual)
        transactions.recover(directory)
        transactions.recover(directory)
        for name, content in bundle.items():
            assert json.loads((directory / name).read_text()) == content


def test_build_only_existing_batch_never_trains(monkeypatch, tmp_path):
    from TEFLD.src import orchestrator as module

    runner = module.Orchestrator.__new__(module.Orchestrator)
    runner.section_id = "fixture"
    monkeypatch.setattr(module, "get_section_dir", lambda _: tmp_path)
    state = Pipeline_State(section_id="fixture", current_training_batch=[sample()])
    monkeypatch.setattr(module, "load_pipeline_state", lambda _: state)
    runner.bootstrap_components = Mock()
    runner.prepare_example_state = Mock(return_value=None)
    runner.recover_round_health = Mock(return_value=[])
    runner.example_path = lambda: Path("fixture.txt")
    runner.train_current_batch_with_model = Mock(side_effect=AssertionError("trained"))
    runner.evaluate_current_batch = Mock(side_effect=AssertionError("evaluated"))
    result = runner.run(build_only=True)
    assert result.status == "batch_ready"
    runner.train_current_batch_with_model.assert_not_called()
    runner.evaluate_current_batch.assert_not_called()


def test_adapter_target_order_and_full_configuration(tmp_path):
    from TEFLD.src.student import TrainyModel

    student = TrainyModel.__new__(TrainyModel)
    expected = SimpleNamespace(
        target_modules={"q_proj", "v_proj"},
        r=4,
        lora_alpha=8,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    student.lora_config = lambda: expected
    config = dict(vars(expected))
    config["target_modules"] = ["v_proj", "q_proj"]
    (tmp_path / "adapter_config.json").write_text(json.dumps(config))
    assert student.adapter_matches_current_lora_config(tmp_path)
    config["r"] = 8
    (tmp_path / "adapter_config.json").write_text(json.dumps(config))
    assert not student.adapter_matches_current_lora_config(tmp_path)
