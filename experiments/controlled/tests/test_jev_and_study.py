import copy
import importlib

import pytest

from TEFLD.src.dataschema import Training_Sample
from TEFLD.src.jev_scorer import JevScorer, MODEL, QUESTIONS
from TEFLD.src.selector_study import matched_probe_group
from TEFLD.src.encoding import supervised_tokens
from TEFLD.src.validation import validate_sample
from TEFLD.src.benefit_predictor import feature_subset


def sample():
    return Training_Sample(
        instruct="Classify cat as animal or plant.",
        sample="",
        gold_summary="animal",
        tag="classification",
        source="generated",
    )


def response(request):
    answers = {}
    for name, question in request["questions"].items():
        if question["type"] == "noul":
            answers[name] = {"type": "noul", "noul": 0.7}
        else:
            answers[name] = {
                "type": "score",
                "score": 1.5,
                "confidence": 0.5,
                "legend": {
                    str(i): level for i, level in enumerate(question["criteria"])
                },
                "probabilities": {"0": 0.0, "1": 0.5, "2": 0.5},
            }
    return {
        "model": MODEL,
        "answers": answers,
        "usage": {"input_tokens": 100, "output_tokens": 20},
    }


def test_jev_fixture_cache_and_provider_separation(tmp_path):
    calls = []

    def fixture(request):
        calls.append(request)
        return response(request)

    scorer = JevScorer(tmp_path, fixture=fixture)
    first = scorer.features(sample(), {"classification": 3.0})
    assert first == scorer.features(sample(), {"classification": 3.0})
    assert len(calls) == 1 and first["jev_relevance"] == 0.75
    assert "confidence" not in first
    with pytest.raises(RuntimeError, match="disabled"):
        JevScorer(tmp_path).features(sample(), {"classification": 3.0})


def test_jev_exports_category_summary_only(tmp_path):
    request = JevScorer(tmp_path).request(
        sample(), {"classification": 3.0, "general_qa": 1.0}
    )
    assert set(request["state"]) == {"candidate", "weak_categories"}
    assert request["state"]["weak_categories"] == ["classification", "general_qa"]
    with pytest.raises(ValueError):
        JevScorer(tmp_path).request(sample(), {"protected_row": {"answer": "SECRET"}})


@pytest.mark.parametrize(
    "corrupt", ["version", "nan", "sum", "score", "usage", "legend"]
)
def test_jev_rejects_bad_responses(tmp_path, corrupt):
    result = response({"questions": QUESTIONS})
    if corrupt == "version":
        result["model"] = "jev-latest"
    if corrupt == "nan":
        result["answers"]["benefit_prior"]["noul"] = float("nan")
    if corrupt == "sum":
        result["answers"]["relevance"]["probabilities"]["0"] = 0.5
    if corrupt == "score":
        result["answers"]["relevance"]["score"] = 2.0
    if corrupt == "usage":
        result["usage"]["output_tokens"] = -1
    if corrupt == "legend":
        result["answers"]["relevance"]["legend"]["0"] = "changed"
    with pytest.raises(ValueError):
        JevScorer(tmp_path).parse(result)


def test_feature_modes_keep_jev_control_free_of_gradients():
    features = {
        "candidate_nll": 2.0,
        "gradient_norm": 1.0,
        "jev_relevance": 0.7,
        "family_qa": 1.0,
    }
    assert "gradient_norm" not in feature_subset(features, "jev")
    assert "jev_relevance" not in feature_subset(features, "gradient")
    assert feature_subset(features, "hybrid") == features
    with pytest.raises(ValueError):
        feature_subset({"candidate_nll": 1.0}, "hybrid")


def test_invalid_json_schema_rejected_without_crashing():
    row = sample().model_copy(
        update={
            "instruct": "Return JSON.",
            "gold_summary": "{}",
            "output_spec": {"json_schema": {"type": "not_a_type"}},
        }
    )
    assert not validate_sample(row).accepted


def test_matched_probe_group_retains_positive_supervision():
    import torch

    encoded = [
        {
            "input_ids": torch.tensor([1, 2, 3, 4]),
            "labels": torch.tensor([-100, 2, 3, 4]),
            "attention_mask": torch.ones(4, dtype=torch.long),
        }
        for _ in range(4)
    ]
    result = matched_probe_group(copy.deepcopy(encoded), 7)
    assert sum(supervised_tokens(x) for x in result) == 7
    assert all(supervised_tokens(x) > 0 for x in result)


def test_download_helpers_have_no_import_side_effects(monkeypatch):
    import huggingface_hub

    monkeypatch.setattr(
        huggingface_hub,
        "snapshot_download",
        lambda **kwargs: pytest.fail("download on import"),
    )
    importlib.import_module("model")
    importlib.import_module("data")


def test_full_view_audit_restores_adapter_and_matches_batch_budgets(
    tmp_path, monkeypatch
):
    from test_correctness import TinyLM, Tokenizer
    from TEFLD.src import runtime
    from TEFLD.src.selector_study import full_view_audit
    from TEFLD.src.candidate_pool import candidate_id
    from TEFLD.src.helper import write_json_file
    from TEFLD.src.counterfactual_probe import fingerprint_tensors

    model = TinyLM()
    before = fingerprint_tensors(list(model.named_parameters()))
    monkeypatch.setattr(runtime, "load_runtime", lambda *a, **k: (model, Tokenizer()))
    candidates = [
        sample().model_copy(update={"instruct": f"Question {i}?"}) for i in range(16)
    ]
    observations = [
        {
            "snapshot_id": "fixture",
            "snapshot_adapter": "fixture",
            "nominal_tokens": 100,
            "actual_tokens": 100,
            "trajectory_seed": 29,
            "candidate_id": candidate_id(row),
            "features": {"adam_gain_projection": float(i)},
        }
        for i, row in enumerate(candidates)
    ]
    write_json_file(tmp_path / "audit_probes.json", observations)
    write_json_file(
        tmp_path / "audit_selection_100.json",
        {"samples": [x.model_dump(mode="json") for x in candidates]},
    )
    write_json_file(
        tmp_path / "predictor.json",
        {
            "features": ["adam_gain_projection"],
            "mean": [0.0],
            "scale": [1.0],
            "coefficients": [1.0],
            "intercept": 0.0,
        },
    )

    class Data:
        def samples(self, role):
            assert role == "curriculum_validation"
            return list(enumerate(candidates[:4]))

    result = full_view_audit(tmp_path, Data(), "fixture")
    assert len(result) == 1 and len(result[0]["matched_batch_gains"]) == 3
    assert result[0]["batch_supervised_tokens"] > 0
    assert result[0]["live_student_updates"] == 0
    assert fingerprint_tensors(list(model.named_parameters())) == before
