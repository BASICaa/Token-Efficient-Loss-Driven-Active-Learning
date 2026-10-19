import copy
from types import SimpleNamespace

import pytest
import torch

from TEFLD.src.optimizer_diagnostic import (
    target_slice,
    build_groups,
    execute_groups,
    capture_rng,
    restore_rng,
    RECIPES,
)
from TEFLD.src.encoding import supervised_tokens
from test_correctness import TinyLM


class DropoutTinyLM(TinyLM):
    def forward(self, input_ids, attention_mask, labels):
        hidden = torch.nn.functional.dropout(
            self.embed(input_ids), p=0.3, training=self.training
        )
        logits = self.head(hidden)
        loss = torch.nn.functional.cross_entropy(
            logits[:, :-1].reshape(-1, 40), labels[:, 1:].reshape(-1), ignore_index=-100
        )
        return SimpleNamespace(loss=loss, logits=logits)


def encoded():
    return {
        "input_ids": torch.tensor([1, 2, 3, 4, 5, 6]),
        "labels": torch.tensor([-100, -100, 3, 4, 5, 6]),
        "attention_mask": torch.ones(6, dtype=torch.long),
    }


def records():
    return [
        {"round": i // 8, "answer_tokens": 4, "tokens_after_group": ((i // 4) + 1) * 16}
        for i in range(16)
    ]


def test_sliced_targets_cover_original_once_without_losing_context():
    original = encoded()
    parts = [
        target_slice(original, 0, 1),
        target_slice(original, 1, 2),
        target_slice(original, 3, 1),
    ]
    masks = torch.stack([p["labels"] != -100 for p in parts]).sum(0)
    torch.testing.assert_close(masks, (original["labels"] != -100).long())
    for part in parts:
        assert supervised_tokens(part) > 0
        torch.testing.assert_close(part["input_ids"], original["input_ids"])
    assert supervised_tokens(original) == 4


@pytest.mark.parametrize("start,count", [(-1, 1), (0, 0), (3, 2), (4, 1)])
def test_invalid_target_slice_rejected(start, count):
    with pytest.raises(ValueError):
        target_slice(encoded(), start, count)


def test_partitioned_target_gradients_match_full_answer_without_dropout():
    torch.manual_seed(6)
    full = TinyLM()
    partitioned = copy.deepcopy(full)
    original = encoded()
    full(**{key: value.unsqueeze(0) for key, value in original.items()}).loss.backward()
    for start, count in [(0, 1), (1, 2), (3, 1)]:
        part = target_slice(original, start, count)
        loss = partitioned(
            **{key: value.unsqueeze(0) for key, value in part.items()}
        ).loss
        (loss * count / 4).backward()
    for left, right in zip(full.parameters(), partitioned.parameters()):
        torch.testing.assert_close(left.grad, right.grad)


def test_fixed_token_groups_cover_5000_targets_exactly_and_end_on_20_updates():
    stream = [
        {"round": 0, "answer_tokens": n, "tokens_after_group": i}
        for i, n in enumerate([7] * 714 + [2])
    ]
    groups = build_groups(stream, RECIPES[1])
    assert len(groups) == 20 and all(x["targets"] == 250 for x in groups)
    seen = []
    for group in groups:
        for part in group["parts"]:
            seen.extend(
                (part["record"], i)
                for i in range(part["start"], part["start"] + part["count"])
            )
    assert len(seen) == len(set(seen)) == 5000
    assert len(build_groups(records(), RECIPES[0])) == 4


@pytest.mark.parametrize("recipe", RECIPES)
def test_optimizer_and_rng_resume_matches_uninterrupted_run(recipe):
    torch.manual_seed(6)
    initial = DropoutTinyLM()
    stream = records()
    groups = build_groups(stream, recipe, targets_per_update=8)
    examples = [encoded() for _ in stream]

    def state():
        return {"next_group": 0, "tokens": 0, "updates": [], "training_seconds": 0.0}

    clean = copy.deepcopy(initial)
    gold = state()
    execute_groups(
        clean, examples, stream, groups, recipe, {0: 2, 1: 2}, seed=42, state=gold
    )
    broken = copy.deepcopy(initial)
    partial = state()
    snapshot = {}

    def interrupted(current, opt):
        if current["next_group"] == 1:
            snapshot.update(
                weights=copy.deepcopy(broken.state_dict()),
                state=copy.deepcopy(current),
                optimizer=copy.deepcopy(opt.state_dict()),
                rng=capture_rng(),
            )
            raise RuntimeError("simulated interruption")

    with pytest.raises(RuntimeError):
        execute_groups(
            broken,
            examples,
            stream,
            groups,
            recipe,
            {0: 2, 1: 2},
            seed=42,
            state=partial,
            after_step=interrupted,
        )
    resumed = copy.deepcopy(initial)
    resumed.load_state_dict(snapshot["weights"])
    optimizer = torch.optim.AdamW(
        resumed.parameters(), lr=0.0002, betas=(0.9, 0.999), eps=1e-8, weight_decay=0
    )
    optimizer.load_state_dict(snapshot["optimizer"])
    restore_rng(snapshot["rng"])
    result = snapshot["state"]
    execute_groups(
        resumed,
        examples,
        stream,
        groups,
        recipe,
        {0: 2, 1: 2},
        seed=42,
        state=result,
        optimizer=optimizer,
    )
    assert (
        result["tokens"] == gold["tokens"] == 64
        and result["updates"] == gold["updates"]
    )
    for left, right in zip(resumed.parameters(), clean.parameters()):
        torch.testing.assert_close(left, right, rtol=0, atol=0)


def test_rng_checkpoint_payload_uses_safe_tensor_loading(tmp_path):
    from TEFLD.src.optimizer_diagnostic import (
        save_training_checkpoint,
        load_training_checkpoint,
    )
    from test_runner import SavedTiny, SavedTokenizer

    model = SavedTiny()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0002)
    state = {"tokens": 0, "next_group": 0}
    save_training_checkpoint(
        model, SavedTokenizer(), optimizer, state, tmp_path, "fixture-identity"
    )
    payload = load_training_checkpoint(tmp_path)
    assert payload["state"] == state
    restore_rng(payload["rng"])
    file = tmp_path / "training_state.pt"
    file.write_bytes(file.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="changed"):
        load_training_checkpoint(tmp_path)


def test_diagnostic_report_supports_the_new_declared_dataset_pair(tmp_path):
    from TEFLD.src.helper import write_json_file
    from TEFLD.src.optimizer_diagnostic import summarize_diagnostic

    methods = ["dolly_baseline", "tefld_dataset_loss"]
    for method in methods:
        for recipe in RECIPES:
            write_json_file(
                tmp_path / method / recipe / "state.json",
                {
                    "status": "complete",
                    "optimizer_updates": 20,
                    "forward_parts": 30,
                    "training_seconds": 1,
                    "evaluation_seconds": 2,
                    "evaluations": [
                        {
                            "nominal_tokens": 5000,
                            "actual_tokens": 5000,
                            "step": 20,
                            "evaluation": {"macro_nll": 1.5},
                        }
                    ],
                },
            )
    report = summarize_diagnostic(tmp_path, methods)
    assert list(report["results"]) == methods
    assert all(len(v) == 2 for v in report["results"].values())
    assert report["provider_calls"] == 0


def test_diagnostic_rejects_a_missing_final_budget_milestone(tmp_path, monkeypatch):
    from TEFLD.src import optimizer_diagnostic as module
    from TEFLD.src.helper import write_json_file
    from test_runner import FixtureData

    monkeypatch.setattr(module, "SplitData", FixtureData)
    source = tmp_path / "source"
    write_json_file(
        source / "state.json",
        {"config": {"seed": 42, "model_path": "fixture"}, "tokens": 77},
    )
    with pytest.raises(ValueError, match="milestones"):
        module.run_condition(
            source, tmp_path / "output", "fixture", RECIPES[0], milestones=(25, 50)
        )
    assert not (tmp_path / "output").exists()
