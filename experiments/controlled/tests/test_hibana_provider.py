import copy
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from TEFLD.src.candidate_pool import Teacher
from TEFLD.src.helper import read_json_file, write_json_file
from TEFLD.src.provider import validate_provider


def profile():
    protocol = (
        Path(__file__).resolve().parents[1]
        / "experiment_design_2026-10-18/experiment_protocol_hibana.json"
    )
    return json.loads(protocol.read_text(encoding="utf-8"))["tefld_controls"][
        "provider"
    ]


def test_hibana_client_and_currency_without_network(tmp_path, monkeypatch):
    import openai

    monkeypatch.setenv("HIBANA_API_KEY", "fixture-hibana-key")
    usage = SimpleNamespace(
        prompt_tokens=100,
        completion_tokens=10,
        prompt_tokens_details=None,
        cost_rial=10000,
        model_dump=lambda **k: {
            "prompt_tokens": 100,
            "completion_tokens": 10,
            "cost_rial": 10000,
        },
    )
    response = SimpleNamespace(
        model="openai/gpt-6.1-sol",
        usage=usage,
        choices=[SimpleNamespace(message=SimpleNamespace(content='{"ok":true}'))],
    )
    create = Mock(return_value=response)
    constructor = Mock(
        return_value=SimpleNamespace(
            base_url=profile()["base_url"],
            chat=SimpleNamespace(completions=SimpleNamespace(create=create)),
        )
    )
    monkeypatch.setattr(openai, "OpenAI", constructor)
    teacher = Teacher(
        tmp_path, model="openai/gpt-6.1-sol", provider=profile(), allow_network=True
    )
    assert teacher.ask("fixture", "verification") == {"ok": True}
    assert constructor.call_args.kwargs["api_key"] == "fixture-hibana-key"
    assert (
        constructor.call_args.kwargs["base_url"] == "https://api-ai.hibanacloud.ir/v1"
    )
    args = create.call_args.kwargs
    assert args["temperature"] == 1.0 and "reasoning_effort" not in args
    assert "service_tier" not in args and "store" not in args
    event = read_json_file(tmp_path / "usage.json", {})["requests"][0]
    assert event["cost_toman"] == 1000 and "cost_usd" not in event
    assert teacher.ask("fixture", "verification") == {"ok": True}
    assert create.call_count == 1


def test_hibana_disabled_does_not_construct_client(tmp_path, monkeypatch):
    import openai

    constructor = Mock(side_effect=AssertionError("Network client constructed"))
    monkeypatch.setattr(openai, "OpenAI", constructor)
    with pytest.raises(PermissionError):
        Teacher(tmp_path, model="openai/gpt-6.1-sol", provider=profile()).ask(
            "fixture", "generation"
        )
    constructor.assert_not_called()


def test_hibana_credential_never_falls_back_to_openai(tmp_path, monkeypatch):
    import openai

    monkeypatch.delenv("HIBANA_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "wrong-provider-fixture")
    # Avoid local dotenv loading in this credential-isolation test.
    monkeypatch.setattr("dotenv.load_dotenv", lambda *a, **k: None)
    constructor = Mock(side_effect=AssertionError("Unexpected client"))
    monkeypatch.setattr(openai, "OpenAI", constructor)
    with pytest.raises(ValueError, match="HIBANA_API_KEY"):
        Teacher(
            tmp_path, model="openai/gpt-6.1-sol", provider=profile(), allow_network=True
        ).ask("fixture", "generation")
    constructor.assert_not_called()


@pytest.mark.parametrize(
    "change",
    [
        {"base_url": "http://api-ai.hibanacloud.ir/v1"},
        {"base_url": "https://example.com/v1"},
        {"api_key_env": "OPENAI_API_KEY"},
        {"currency": "usd"},
        {"max_spend": float("nan")},
    ],
)
def test_hibana_invalid_config(change):
    config = copy.deepcopy(profile())
    config.update(change)
    with pytest.raises(ValueError):
        validate_provider(config)


def test_hibana_spend_cap_blocks_transport(tmp_path, monkeypatch):
    import openai

    monkeypatch.setenv("HIBANA_API_KEY", "fixture-key")
    create = Mock(side_effect=AssertionError("Request executed"))
    monkeypatch.setattr(
        openai,
        "OpenAI",
        lambda **k: SimpleNamespace(
            base_url=profile()["base_url"],
            chat=SimpleNamespace(completions=SimpleNamespace(create=create)),
        ),
    )
    config = profile()
    config["max_spend"] = 0.0001
    with pytest.raises(RuntimeError, match="spend ceiling"):
        Teacher(
            tmp_path, model="openai/gpt-6.1-sol", provider=config, allow_network=True
        ).ask("fixture", "generation")
    create.assert_not_called()


def test_model_and_price_profile_cannot_diverge(tmp_path):
    with pytest.raises(ValueError, match="pricing"):
        Teacher(tmp_path, model="gpt-5.4-mini", provider=profile())


def test_sol_default_resolves_to_hibana_without_network(tmp_path):
    teacher = Teacher(tmp_path)
    assert teacher.model == "openai/gpt-6.1-sol"
    assert teacher.provider["name"] == "hibana"
    assert teacher.provider["model_id"] == teacher.model
    assert teacher.provider["input_per_million"] == 734000
    assert teacher.provider["output_per_million"] == 3672000
    assert not teacher.allow_network


def test_legacy_sol_client_uses_hibana_key_and_endpoint(monkeypatch):
    from TEFLD.src import helper

    monkeypatch.setenv("HIBANA_API_KEY", "fixture-key")
    monkeypatch.setenv("OPENAI_API_KEY", "wrong-provider-fixture")
    constructor = Mock(return_value=object())
    monkeypatch.setattr(helper, "_import_openai_client", lambda: constructor)
    helper.create_client("openai/gpt-6.1-sol")
    assert constructor.call_args.kwargs["api_key"] == "fixture-key"
    assert constructor.call_args.kwargs["base_url"] == profile()["base_url"]


def test_controlled_protocol_selects_sol_and_bound_rates():

    path = (
        Path(__file__).resolve().parents[1]
        / "experiment_design_2026-10-18/experiment_protocol_hibana.json"
    )
    controls = json.loads(path.read_text(encoding="utf-8"))["tefld_controls"]
    assert controls["teacher_model_id"] == profile()["model_id"] == "openai/gpt-6.1-sol"
    assert (
        controls["teacher_pricing"]["input_per_million"]
        == profile()["input_per_million"]
    )
    assert (
        controls["teacher_pricing"]["output_per_million"]
        == profile()["output_per_million"]
    )


def test_spending_override_preserves_provider_identity_and_cached_responses(
    tmp_path, monkeypatch
):
    import openai

    teacher = Teacher(tmp_path, provider=profile())
    before = teacher.identity
    write_json_file(
        tmp_path / "spend_limit.json",
        {"teacher_identity": before, "currency": "toman", "max_spend": 250000},
    )
    assert teacher.spend_ceiling() == 250000
    assert teacher.identity == before and teacher.provider["max_spend"] == 100000
    monkeypatch.setattr(
        openai, "OpenAI", Mock(side_effect=AssertionError("Network constructed"))
    )
    # Existing cached responses use the same unchanged provider configuration.
    from TEFLD.src.helper import stable_json_hash

    key = stable_json_hash(
        {
            "prompt": "cached",
            "model": teacher.model,
            "temperature": 1.0,
            "kind": "generation",
            "version": "verified-pool-v1",
            "provider": teacher.provider,
        }
    )
    write_json_file(tmp_path / (key + ".json"), {"content": '{"ok":true}'})
    assert teacher.ask("cached", "generation") == {"ok": True}


def test_higher_total_cap_uses_existing_spend_and_still_blocks_overrun(
    tmp_path, monkeypatch
):
    import openai

    monkeypatch.setenv("HIBANA_API_KEY", "fixture-key")
    usage = SimpleNamespace(
        prompt_tokens=100,
        completion_tokens=10,
        prompt_tokens_details=None,
        model_dump=lambda **k: {"prompt_tokens": 100, "completion_tokens": 10},
    )
    create = Mock(
        return_value=SimpleNamespace(
            model="openai/gpt-6.1-sol",
            usage=usage,
            choices=[SimpleNamespace(message=SimpleNamespace(content='{"ok":true}'))],
        )
    )
    monkeypatch.setattr(
        openai,
        "OpenAI",
        lambda **k: SimpleNamespace(
            base_url=profile()["base_url"],
            chat=SimpleNamespace(completions=SimpleNamespace(create=create)),
        ),
    )
    teacher = Teacher(tmp_path, provider=profile(), allow_network=True)
    write_json_file(
        tmp_path / "usage.json",
        {"identity": teacher.identity, "requests": [{"cost_toman": 120000}]},
    )
    with pytest.raises(RuntimeError, match="spend ceiling"):
        teacher.ask("first", "generation")
    create.assert_not_called()
    write_json_file(
        tmp_path / "spend_limit.json",
        {
            "teacher_identity": teacher.identity,
            "currency": "toman",
            "max_spend": 250000,
        },
    )
    assert teacher.ask("first", "generation") == {"ok": True}
    ledger = read_json_file(tmp_path / "usage.json", {})
    assert (
        ledger["requests"][0]["cost_toman"] == 120000
        and ledger["requests"][1]["spend_ceiling"] == 250000
    )
    ledger["requests"].append({"reserved_cost_toman": 129999})
    write_json_file(tmp_path / "usage.json", ledger)
    with pytest.raises(RuntimeError, match="spend ceiling"):
        teacher.ask("second", "generation")
    assert create.call_count == 1


@pytest.mark.parametrize(
    "change",
    [
        {"teacher_identity": "wrong"},
        {"currency": "usd"},
        {"max_spend": 0},
        {"max_spend": "250000"},
    ],
)
def test_invalid_spending_override_rejected(tmp_path, change):
    teacher = Teacher(tmp_path, provider=profile())
    override = {
        "teacher_identity": teacher.identity,
        "currency": "toman",
        "max_spend": 250000,
    }
    override.update(change)
    write_json_file(tmp_path / "spend_limit.json", override)
    with pytest.raises(ValueError):
        teacher.spend_ceiling()
