import random
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from TEFLD.src.benefit_features import sketch, gradient_summary, candidate_features
from TEFLD.src.benefit_predictor import BenefitPredictor, regression_metrics
from TEFLD.src.counterfactual_probe import fingerprint_tensors, virtual_update
from TEFLD.src.encoding import encode_sample
from TEFLD.src.experiment_runner import select_checkpoint
from TEFLD.src.validation_data import SplitData, score_samples
from test_correctness import TinyLM, Tokenizer, sample


@pytest.mark.parametrize("fail", [False, True])
def test_probe_restores_weights_gradients_modes_buffers_and_rng(fail):
    torch.manual_seed(10)
    np.random.seed(11)
    random.seed(12)
    model = TinyLM()
    model.register_buffer("counter", torch.tensor([3]))
    model.train()
    model.head.eval()
    for p in model.parameters():
        p.grad = torch.ones_like(p) * 0.5
    before = fingerprint_tensors(model.named_parameters())
    grads = [p.grad.clone() for p in model.parameters()]
    modes = [m.training for m in model.modules()]
    rng = torch.get_rng_state().clone()
    numpy_state = np.random.get_state()
    python_state = random.getstate()
    calls = 0

    def measure():
        nonlocal calls
        calls += 1
        if fail and calls == 2:
            model.counter.add_(7)
            raise RuntimeError("simulated audit interruption")
        return score_samples(model, Tokenizer(), [(1, sample())])["macro_nll"]

    if fail:
        with pytest.raises(RuntimeError, match="interruption"):
            virtual_update(
                model, encode_sample(sample(), Tokenizer()), measure=measure, seed=99
            )
    else:
        result = virtual_update(
            model, encode_sample(sample(), Tokenizer()), measure=measure, seed=99
        )
        assert np.isfinite(result["gain"])
    assert fingerprint_tensors(model.named_parameters()) == before
    assert model.counter.item() == 3
    assert [m.training for m in model.modules()] == modes
    for p, g in zip(model.parameters(), grads):
        torch.testing.assert_close(p.grad, g)
    assert torch.equal(torch.get_rng_state(), rng)
    assert random.getstate() == python_state
    assert np.array_equal(np.random.get_state()[1], numpy_state[1])


def test_validation_gradient_statistics_never_update_student_or_grad_buffers():
    model = TinyLM()
    model.train()
    for p in model.parameters():
        p.grad = torch.ones_like(p)
    before = fingerprint_tensors(model.named_parameters())
    result = gradient_summary(
        model,
        Tokenizer(),
        [(1, sample()), (2, sample(tag="classification"))],
        dimensions=16,
    )
    assert len(result["overall"]) == 16
    assert fingerprint_tensors(model.named_parameters()) == before
    assert all(torch.equal(p.grad, torch.ones_like(p)) for p in model.parameters())
    features = candidate_features(model, Tokenizer(), sample(), result)
    assert all(np.isfinite(v) for v in features.values())
    assert fingerprint_tensors(model.named_parameters()) == before


def test_streaming_projection_is_deterministic_and_linearly_composable():
    a = {"a": torch.arange(15, dtype=torch.float32)}
    b = {"a": torch.ones(15)}
    torch.testing.assert_close(sketch(a, 16), sketch(a, 16))
    torch.testing.assert_close(
        sketch({"a": a["a"] + b["a"]}, 16), sketch(a, 16) + sketch(b, 16)
    )


def test_frozen_predictor_json_roundtrip_and_feature_schema(tmp_path):
    rows = [
        {
            "features": {"length": float(i), "alignment": float(i % 3)},
            "gain": float(i) * 0.1,
        }
        for i in range(24)
    ]
    predictor = BenefitPredictor.fit(rows, alpha=0.01)
    path = tmp_path / "predictor.json"
    predictor.save(path)
    restored = BenefitPredictor.load(path)
    assert restored.predict(rows[3]["features"]) == predictor.predict(
        rows[3]["features"]
    )
    assert regression_metrics(rows, restored)["rmse"] < 0.01
    assert restored.predict(rows[3]["features"])["p_improve_calibrated"] is None
    with pytest.raises(ValueError):
        restored.predict({"length": 2})


def test_exact_checkpoint_selection_ignores_old_three_percent_threshold():
    values = [{"macro_nll": 2.7, "tokens": 5000}, {"macro_nll": 2.69, "tokens": 10000}]
    assert select_checkpoint(values)["tokens"] == 10000
    values.append({"macro_nll": 2.69, "tokens": 15000})
    assert select_checkpoint(values)["tokens"] == 10000


def test_final_test_and_training_seed_roles_are_guarded(tmp_path):
    import hashlib
    import json
    from TEFLD.src.helper import write_json_file

    dataset, manifest, protocol = [tmp_path / name for name in ("data.jsonl", "split.json", "protocol.json")]
    dataset.write_text("\n".join(json.dumps({"instruction": f"Question {i}?", "context": "", "response": "Yes", "category": "general_qa"}) for i in range(4)), encoding="utf-8")
    write_json_file(manifest, {"splits": {role: [i] for i, role in enumerate(("training_pool", "curriculum_validation", "checkpoint_validation", "final_test"))}})
    write_json_file(protocol, {"dataset": {"source": str(dataset), "manifest": str(manifest),
        "sha256": hashlib.sha256(dataset.read_bytes()).hexdigest(),
        "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest()},
        "tefld_controls": {"seed_example_row_ids": [0]}})
    data = SplitData(protocol)
    row_id = next(iter(data.ids["final_test"]))
    with pytest.raises(PermissionError):
        data.sample(row_id, "final_test")
    with pytest.raises(PermissionError):
        data.sample(row_id, "training_pool")
    assert all(s.verification["role"] == "training_pool" for s in data.teacher_seeds())


def test_committed_evaluation_retry_is_idempotent(tmp_path, monkeypatch):
    from TEFLD.src import helper
    from TEFLD.src.dataschema import Pipeline_State, Evaluation_Output
    from TEFLD.src.evaluator import Evaluator

    monkeypatch.setattr(helper, "SECTIONS_DIR", tmp_path)
    state = Pipeline_State(section_id="fixture", current_training_batch=[sample()])
    helper.save_pipeline_state(state, "fixture")
    evaluation = Evaluation_Output(
        sample="",
        instruct="Answer briefly.",
        gold_summary="Yes",
        student_ans="",
        token_length=20,
        loss=0.6,
        tag="general_qa",
        source="generated",
    )
    student = SimpleNamespace(
        section_id="fixture",
        pipeline=state,
        model=object(),
        evaluate_samples=lambda *a, **k: [evaluation],
    )
    evaluator = Evaluator("fixture", student=student)
    first = evaluator.run()
    second = evaluator.run()
    assert len(first[1]) == len(second[1]) == 1
    assert helper.load_pipeline_state("fixture").current_round == 1


def test_no_update_measurement_is_zero_and_preserves_mode():
    model = TinyLM()
    model.train()
    first = score_samples(model, Tokenizer(), [(1, sample())])
    second = score_samples(model, Tokenizer(), [(1, sample())])
    assert first["macro_nll"] == second["macro_nll"] and model.training
