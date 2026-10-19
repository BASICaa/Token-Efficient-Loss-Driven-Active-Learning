import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from TEFLD.src.candidate_pool import Teacher
from TEFLD.src.generation_prompts import compact_prompt
from TEFLD.src.helper import read_json_file, write_json_file
from TEFLD.src.provider import hibana_profile_for_model
from TEFLD.src.teacher_accounting import require_study_budget, summarize_teacher_costs
from TEFLD.src.teacher_study import comparison_protocol
from test_correctness import sample

ROOT = Path(__file__).resolve().parents[1]


def base():
    return read_json_file(
        ROOT / "experiment_design_2026-10-18/experiment_protocol_hibana.json", {}
    )


@pytest.mark.parametrize(
    "teacher,expected", [("sol6", "openai/gpt-6-sol"), ("sol61", "openai/gpt-6.1-sol")]
)
def test_profiles_bind_generator_and_fixed_verifier(teacher, expected):
    original = base()
    protocol = comparison_protocol(original, teacher, "compact")
    controls = protocol["tefld_controls"]
    assert controls["teacher_model_id"] == controls["provider"]["model_id"] == expected
    assert (
        controls["verifier"]["model_id"]
        == controls["verifier"]["provider"]["model_id"]
        == "openai/gpt-6.1-sol"
    )
    assert controls["generation_prompt_variant"] == "compact"
    assert original == base()
    assert not protocol["teacher_comparison"]["primary_ready"]


def test_legacy_sol6_routes_to_distinct_bound_profile(tmp_path):
    config = hibana_profile_for_model("openai/gpt-6-sol")
    assert config["model_id"] == "openai/gpt-6-sol"
    assert Teacher(tmp_path, model="openai/gpt-6-sol").provider == config


def test_compact_prompt_keeps_only_complete_relevant_seed_and_bounded_feedback():
    relevant = sample(
        instruct="Summarize this.",
        sample="A short complete passage.",
        gold_summary="A complete summary.",
        tag="summarization",
    )
    unrelated = sample(sample="UNRELATED_SECRET" * 100, tag="general_qa")
    prompt = compact_prompt(
        "summarization",
        [unrelated, relevant],
        "slot:0",
        "gardening",
        weakness={"summarization": 3.0, "general_qa": 2.0},
        avoid_questions=["a" * 200] * 30,
        rejection_feedback=["b" * 300] * 10,
    )
    data = json.loads(prompt.split("\n", 1)[1])
    assert data["format_example"]["sample"] == relevant.sample
    assert data["format_example"]["gold_summary"] == relevant.gold_summary
    assert "UNRELATED_SECRET" not in prompt
    assert len(data["avoid_questions"]) == 6 and all(
        len(x) <= 100 for x in data["avoid_questions"]
    )
    assert len(data["retry_feedback"]) == 2
    assert "weakness_summary" not in data and "weak_categories" in data
    assert "Return only JSON" in prompt and "correct" in prompt


def test_long_reference_seed_is_omitted_without_truncation():
    long = sample(sample="COMPLETE" * 200, tag="general_qa")
    payload = json.loads(
        compact_prompt("general_qa", [long], "id", "light").split("\n", 1)[1]
    )
    assert "format_example" not in payload


def test_shared_budget_counts_both_generators_verifier_and_unknown_charges(tmp_path):
    write_json_file(
        tmp_path / "arms/sol6/teacher_cache/usage.json",
        {"requests": [{"cost_toman": 50}]},
    )
    write_json_file(
        tmp_path / "arms/sol61/teacher_cache/usage.json",
        {"requests": [{"cost_toman": 30}]},
    )
    write_json_file(
        tmp_path / "verifier_cache/usage.json",
        {"requests": [{"reserved_cost_toman": 10}]},
    )
    with pytest.raises(PermissionError):
        require_study_budget(tmp_path, "toman", 1)
    write_json_file(
        tmp_path / "study_budget.json",
        {"approved": True, "currency": "toman", "max_spend": 100},
    )
    assert require_study_budget(tmp_path, "toman", 10) == 100
    with pytest.raises(RuntimeError, match="Shared"):
        require_study_budget(tmp_path, "toman", 11)


def test_cost_report_keeps_cache_hits_separate_from_billed_requests(tmp_path):
    owner = {
        "arm": "sol6-compact",
        "phase": "pilot",
        "method": "tefld_loss",
        "seed": 42,
        "round": 1,
    }
    write_json_file(
        tmp_path / "teacher_cache/usage.json",
        {
            "requests": [
                {
                    "owner": owner,
                    "model": "openai/gpt-6-sol",
                    "kind": "generation",
                    "status": "completed",
                    "cost_toman": 123.0,
                    "usage": {"prompt_tokens": 10, "completion_tokens": 3},
                }
            ]
        },
    )
    write_json_file(
        tmp_path / "teacher_cache/accesses.json",
        [{"cache_hit": False}, {"cache_hit": True}, {"cache_hit": True}],
    )
    report = summarize_teacher_costs(tmp_path)
    assert report["requests"] == 1 and report["cache_hits"] == 2
    assert report["known_cost"]["toman"] == 123.0
    assert report["attributed_groups"][0]["owner"] == owner


def test_common_verifier_inherits_calling_arm_ownership(tmp_path):
    verifier = Teacher(tmp_path / "verifier")
    verifier.ask = Mock(
        return_value={
            "correct": True,
            "supported": True,
            "format_valid": True,
            "unambiguous": True,
        }
    )
    observed = []
    verifier.ask.side_effect = lambda *a, **k: observed.append(
        dict(verifier.owner)
    ) or {"correct": True, "supported": True, "format_valid": True, "unambiguous": True}
    generator = Teacher(
        tmp_path / "generator", model="openai/gpt-6-sol", verifier=verifier
    )
    with generator.account_to(arm="sol6-compact", phase="pilot", round=2):
        result = generator.verify(sample())
    assert result["verifier_model"] == "openai/gpt-6.1-sol"
    assert observed == [{"arm": "sol6-compact", "phase": "pilot", "round": 2}]
    assert verifier.owner == {} and generator.owner == {}


def test_no_paid_transport_when_study_budget_is_unapproved(tmp_path, monkeypatch):
    import openai

    monkeypatch.setenv("HIBANA_API_KEY", "fixture")
    create = Mock(side_effect=AssertionError("Request sent"))
    monkeypatch.setattr(
        openai,
        "OpenAI",
        lambda **k: SimpleNamespace(
            base_url="https://api-ai.hibanacloud.ir/v1",
            chat=SimpleNamespace(completions=SimpleNamespace(create=create)),
        ),
    )
    teacher = Teacher(
        tmp_path / "teacher_cache", allow_network=True, study_budget_root=tmp_path
    )
    with pytest.raises(PermissionError, match="approved"):
        teacher.ask("example", "generation")
    create.assert_not_called()
    assert not (tmp_path / "teacher_cache/usage.json").exists()


def test_sol6_and_sol61_cache_namespaces_are_distinct(tmp_path):
    sol6 = Teacher(tmp_path / "sol6", model="openai/gpt-6-sol")
    sol61 = Teacher(tmp_path / "sol61")
    assert sol6.identity != sol61.identity
    assert sol6.provider["model_id"] != sol61.provider["model_id"]


def test_prompt_pilot_runs_all_32_fixture_slots_and_resumes_without_regeneration(
    tmp_path, monkeypatch
):
    import hashlib
    from TEFLD.src import teacher_study
    from test_correctness import Tokenizer
    from test_runner import CATEGORIES

    profiles = {}
    for arm in ("sol61-full", "sol61-compact", "sol6-full", "sol6-compact", "dolly"):
        path = tmp_path / "protocols" / (arm + ".json")
        write_json_file(path, {})
        profiles[arm] = str(path)
    write_json_file(
        tmp_path / "study_plan.json",
        {
            "identity": "fixture",
            "profiles": profiles,
            "profile_sha256": {
                key: hashlib.sha256(Path(path).read_bytes()).hexdigest()
                for key, path in profiles.items()
            },
            "warmup_sha256": {},
        },
    )
    verifier = Teacher(tmp_path / "verifier_cache")
    verifier.ask = Mock(
        return_value={
            "correct": True,
            "supported": True,
            "format_valid": True,
            "unambiguous": True,
        }
    )
    write_json_file(
        tmp_path / "verification_audit.json",
        {"status": "passed", "teacher_identity": verifier.identity},
    )
    calls = []

    def runner(path, root, **kwargs):
        arm = Path(path).stem
        generator = Teacher(
            tmp_path / "arms" / arm / "teacher_cache",
            model=(
                "openai/gpt-6-sol" if arm.startswith("sol6-") else "openai/gpt-6.1-sol"
            ),
            verifier=verifier,
        )

        def generate(category, *args, **kwargs):
            calls.append((arm, category))
            return sample(
                instruct="Explain " + category + ".",
                tag=category,
            )

        generator.generate = generate
        data = SimpleNamespace(
            ids={
                "curriculum_validation": set(),
                "checkpoint_validation": set(),
                "final_test": set(),
            },
            manifest={"excluded": {"historical_and_connected": []}},
            rows=[],
            teacher_seeds=lambda: [],
            protocol={"common_training": {"max_length": 512}},
        )
        return SimpleNamespace(
            teacher=generator,
            verifier=verifier,
            data=data,
            categories=CATEGORIES,
            verification_audit_path=tmp_path / "verification_audit.json",
            output=tmp_path / "arms" / arm,
        )

    monkeypatch.setattr(teacher_study, "ExperimentRunner", runner)
    first = teacher_study.prompt_pilot(tmp_path, Tokenizer())
    assert first["slots"] == 32 and all(
        x == 8 for x in first["accepted_by_arm"].values()
    )
    assert len(calls) == 32
    second = teacher_study.prompt_pilot(tmp_path, Tokenizer())
    assert second["slots"] == 32 and len(calls) == 32


def test_prepared_plan_rejects_changed_profile_or_warmup(tmp_path):
    import hashlib
    from TEFLD.src.teacher_study import verify_plan

    path = tmp_path / "profile.json"
    write_json_file(path, {})
    warmup = tmp_path / "shared_initial_seed_42.json"
    write_json_file(warmup, [])
    plan = {
        "profiles": {"arm": str(path)},
        "profile_sha256": {"arm": hashlib.sha256(path.read_bytes()).hexdigest()},
        "warmup_sha256": {"42": hashlib.sha256(warmup.read_bytes()).hexdigest()},
    }
    write_json_file(tmp_path / "study_plan.json", plan)
    assert verify_plan(tmp_path) == plan
    write_json_file(warmup, [{"changed": True}])
    with pytest.raises(ValueError, match="warmup"):
        verify_plan(tmp_path)
    write_json_file(warmup, [])
    write_json_file(path, {"changed": True})
    with pytest.raises(ValueError, match="protocol"):
        verify_plan(tmp_path)


def test_existing_matching_audit_reuse_makes_no_requests(tmp_path,monkeypatch):
    from TEFLD.src import teacher_study
    verifier=Teacher(tmp_path/'verifier')
    monkeypatch.setattr(teacher_study,'verify_plan',lambda root:{'profiles':{'sol61-compact':'fixture'}})
    monkeypatch.setattr(teacher_study,'ExperimentRunner',lambda *a,**k:SimpleNamespace(verifier=verifier,verification_audit_path=tmp_path/'verification_audit.json'))
    source=tmp_path/'old_audit.json'
    audit={'status':'passed','model':verifier.model,'teacher_identity':verifier.identity,'valid_cases':8,'invalid_cases':8,'false_accepts':0,'false_rejects':0,
           'results':[{'expected_accept':bool(i%2),'accepted':bool(i%2)} for i in range(16)]}
    write_json_file(source,audit)
    result=teacher_study.import_verifier_audit(tmp_path,source)
    assert result['provider_calls']==0
    assert read_json_file(source,{})==audit
    assert read_json_file(tmp_path/'verification_audit.json',{})['reused_from']==str(source.resolve())
    audit['teacher_identity']='wrong';write_json_file(source,audit)
    with pytest.raises(ValueError,match='contract'):teacher_study.import_verifier_audit(tmp_path,source)
