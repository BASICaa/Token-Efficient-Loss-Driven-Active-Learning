from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from TEFLD.src.candidate_pool import Teacher
from TEFLD.src.counterfactual_probe import virtual_update
from TEFLD.src.encoding import encode_sample
from TEFLD.src.helper import read_json_file
from TEFLD.src.validation import validate_sample
from TEFLD.src.validation_data import score_samples
from test_correctness import TinyLM, Tokenizer, sample


@pytest.mark.parametrize(
    "answer,spec,accepted",
    [
        ("a,b\n1,2", {"format": "csv", "columns": ["a", "b"]}, True),
        ("a,b\n1,2,3", {"format": "csv", "columns": ["a", "b"]}, False),
        ("<answer>yes</answer>", {"format": "xml", "root": "answer"}, True),
        ("<other/>", {"format": "xml", "root": "answer"}, False),
        (
            "| A | B |\n| --- | --- |\n| 1 | 2 |",
            {"format": "markdown_table", "columns": ["A", "B"]},
            True,
        ),
        ("12.5", {"format": "number"}, True),
        ("Infinity", {"format": "number"}, False),
        ("One line.", {"line_count": 2}, False),
        ("A valid sentence.", {"format": "unknown"}, False),
        (
            '{"value":2}',
            {
                "format": "json",
                "json_schema": {
                    "type": "object",
                    "properties": {"value": {"type": "integer"}},
                    "required": ["value"],
                    "additionalProperties": False,
                },
            },
            True,
        ),
        (
            '{"value":"bad"}',
            {
                "format": "json",
                "json_schema": {
                    "type": "object",
                    "properties": {"value": {"type": "integer"}},
                    "required": ["value"],
                },
            },
            False,
        ),
    ],
)
def test_explicit_output_contracts(answer, spec, accepted):
    assert (
        validate_sample(sample(gold_summary=answer), output_spec=spec).accepted
        == accepted
    )


def test_json_word_in_a_question_does_not_imply_json_output():
    assert validate_sample(
        sample(
            instruct="What is JSON?",
            gold_summary="JSON is a text data interchange format.",
        )
    ).accepted
    assert not validate_sample(
        sample(instruct="Return only JSON.", gold_summary='```json\n{"x":1}\n```')
    ).accepted


def test_cache_only_never_constructs_api_client(tmp_path, monkeypatch):
    import openai

    constructor = Mock(side_effect=AssertionError("Network client constructed."))
    monkeypatch.setattr(openai, "OpenAI", constructor)
    with pytest.raises(PermissionError):
        Teacher(tmp_path).ask("fixture", "generation")
    constructor.assert_not_called()


def test_invalid_provider_json_still_records_known_usage(tmp_path, monkeypatch):
    import openai

    usage = SimpleNamespace(
        prompt_tokens=20,
        completion_tokens=10,
        prompt_tokens_details=SimpleNamespace(cached_tokens=0),
        model_dump=lambda **k: {"prompt_tokens": 20, "completion_tokens": 10},
    )
    response = SimpleNamespace(
        model="gpt-5.4-mini-2026-03-17",
        usage=usage,
        choices=[SimpleNamespace(message=SimpleNamespace(content="invalid json"))],
    )
    create = Mock(return_value=response)
    monkeypatch.setattr(
        openai,
        "OpenAI",
        lambda **k: SimpleNamespace(
            base_url="https://api.openai.com/v1/",
            chat=SimpleNamespace(completions=SimpleNamespace(create=create)),
        ),
    )
    with pytest.raises(ValueError):
        Teacher(tmp_path, model="gpt-5.4-mini", allow_network=True).ask(
            "fixture", "generation"
        )
    event = read_json_file(tmp_path / "usage.json", {})["requests"][0]
    assert event["status"] == "failed" and event["cost_usd"] > 0
    assert event["usage"]["completion_tokens"] == 10


def test_spend_cap_blocks_request_before_transport(tmp_path, monkeypatch):
    import openai

    create = Mock(side_effect=AssertionError("Paid request executed."))
    monkeypatch.setattr(
        openai,
        "OpenAI",
        lambda **k: SimpleNamespace(
            base_url="https://api.openai.com/v1/",
            chat=SimpleNamespace(completions=SimpleNamespace(create=create)),
        ),
    )
    with pytest.raises(RuntimeError, match="spend ceiling"):
        Teacher(
            tmp_path, model="gpt-5.4-mini", allow_network=True, max_spend_usd=0.000001
        ).ask("fixture", "generation")
    create.assert_not_called()


def test_virtual_gain_agrees_with_small_step_first_order_direction():
    torch.manual_seed(1)
    model = TinyLM()
    model.eval()
    encoded = encode_sample(sample(), Tokenizer())
    loss = model(**{k: v.unsqueeze(0) for k, v in encoded.items()}).loss
    params = list(model.named_parameters())
    gradients = torch.autograd.grad(loss, [p for _, p in params])
    outcome = virtual_update(
        model,
        encoded,
        measure=lambda: score_samples(model, Tokenizer(), [(1, sample())])["macro_nll"],
        lr=0.0001,
    )
    predicted = sum(
        float((g.detach().cpu() * outcome["displacement"][name]).sum())
        for (name, _), g in zip(params, gradients)
    )
    assert predicted > 0 and outcome["gain"] > 0
    assert outcome["gain"] == pytest.approx(predicted, rel=0.03, abs=1e-5)


def test_health_recovery_after_state_write_does_not_duplicate_event(
    tmp_path, monkeypatch
):
    from TEFLD.src import helper
    from TEFLD.src.dataschema import Pipeline_State, Evaluation_Output
    from TEFLD.src.orchestrator import Orchestrator

    monkeypatch.setattr(helper, "SECTIONS_DIR", tmp_path)
    helper.save_pipeline_state(
        Pipeline_State(section_id="fixture", current_round=1), "fixture"
    )
    evaluation = Evaluation_Output(
        sample="",
        instruct="Q",
        gold_summary="A",
        student_ans="",
        token_length=4,
        loss=0.8,
        tag="general_qa",
        source="generated",
    )
    helper.write_json_file(
        tmp_path / "fixture/round_result_000000.json",
        {
            "round_id": 0,
            "health_pending": True,
            "health_context": {},
            "evaluations": [evaluation.model_dump(mode="json")],
        },
    )
    runner = Orchestrator.__new__(Orchestrator)
    runner.section_id = "fixture"
    original = runner.finish_round_health
    runner.finish_round_health = Mock(
        side_effect=OSError("interrupted after state write")
    )
    with pytest.raises(OSError):
        runner.recover_round_health()
    runner.finish_round_health = original
    runner.recover_round_health()
    runner.recover_round_health()
    state = helper.load_pipeline_state("fixture")
    assert state.current_round == 1 and len(state.round_health_history) == 1


@pytest.mark.parametrize("section", ["..", "../other", "..\\other", "D:\\other"])
def test_section_paths_cannot_escape_workspace(section):
    from TEFLD.src.helper import get_section_dir

    with pytest.raises(ValueError):
        get_section_dir(section)
