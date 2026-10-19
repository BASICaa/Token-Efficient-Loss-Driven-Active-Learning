import random
from pathlib import Path
from types import SimpleNamespace

import pytest

from TEFLD.src.candidate_pool import LeakageGuard, candidate_id
from TEFLD.src.dataset_curriculum import (
    dataset_protocol,
    reference_sample,
    require_reference_provenance,
    retrieve_references,
)
from test_runner import FixtureData, SavedTokenizer


def data():
    return FixtureData("fixture")


def test_dataset_profile_adds_one_fifth_arm_without_mutating_base():
    original = {"methods": [{"id": str(i)} for i in range(4)]}
    protocol = dataset_protocol(original)
    assert len(protocol["methods"]) == 5 and len(original["methods"]) == 4
    assert protocol["dataset_tefld"]["enabled_methods"] == ["tefld_dataset_loss"]
    assert protocol["methods"][-1]["development_data_affects_curriculum"] is False


def test_reference_retrieval_obeys_focus_quotas_and_preserves_answers():
    split = data()
    tokenizer, guard = SavedTokenizer(), LeakageGuard(split)
    order = sorted(split.ids["training_pool"])
    random.Random(42).shuffle(order)
    seen = set()
    batch, trace = retrieve_references(
        split, tokenizer, guard, order, seen, {"classification": 3}, ["open_qa"] * 4
    )
    assert [x.tag for x in batch] == ["classification"] * 3 + ["open_qa"] * 4
    assert len(seen) == 7 and len({candidate_id(x) for x in batch}) == 7
    assert [x["slot"] for x in trace] == ["focus"] * 3 + ["exploration"] * 4
    for sample in batch:
        require_reference_provenance(split, sample, tokenizer, guard)
    again, _ = retrieve_references(
        split, tokenizer, guard, order, seen, {"classification": 3}, []
    )
    assert not {candidate_id(x) for x in batch} & {candidate_id(x) for x in again}
    assert set(split.access) == {"training_pool"}


@pytest.mark.parametrize("row_id", [240, 248, 256, True, None])
def test_protected_or_invalid_reference_cannot_be_retrieved(row_id):
    split = data()
    with pytest.raises(PermissionError):
        reference_sample(split, row_id, SavedTokenizer(), LeakageGuard(split))
    assert split.access == []


def test_exhaustion_fails_before_admission_without_mutating_seen():
    split = data()
    seen = {
        candidate_id(split.sample(i, "training_pool"))
        for i in split.ids["training_pool"]
    }
    before = seen.copy()
    with pytest.raises(RuntimeError, match="no synthetic fallback"):
        retrieve_references(
            split,
            SavedTokenizer(),
            LeakageGuard(split),
            sorted(split.ids["training_pool"]),
            seen,
            {"classification": 1},
            [],
        )
    assert seen == before


def test_protected_row_in_retrieval_order_is_rejected():
    split = data()
    with pytest.raises(PermissionError):
        retrieve_references(
            split,
            SavedTokenizer(),
            LeakageGuard(split),
            [256],
            set(),
            {"brainstorming": 1},
            [],
        )


def test_new_teacher_study_adds_dataset_without_expanding_prompt_pilot(
    tmp_path, monkeypatch
):
    from TEFLD.src import teacher_study
    from TEFLD.src.helper import write_json_file, read_json_file
    from test_two_teacher_study import base

    protocol = tmp_path / "base.json"
    write_json_file(protocol, base())
    split = data()
    monkeypatch.setattr(
        teacher_study,
        "ExperimentRunner",
        lambda *a, **k: SimpleNamespace(
            data=split, categories=split.protocol["dataset"]["categories"]
        ),
    )
    root = tmp_path / "new_study"
    plan = teacher_study.prepare_study(
        protocol, root, tokenizer=SavedTokenizer(), seeds=(42,), dataset_feedback="loss"
    )
    assert len(plan["main_training_conditions"]) == 6
    assert plan["main_training_conditions"][-1] == "dataset-refs:tefld_dataset_loss"
    assert plan["prompt_pilot_slots"] == 32
    assert (
        read_json_file(Path(plan["profiles"]["dataset-refs"]), {})[
            "teacher_comparison"
        ]["generator_factor"]
        is None
    )
    warmup = read_json_file(root / "shared_initial_seed_42.json", [])
    assert all(x["verification"]["role"] == "training_pool" for x in warmup)
    assert not read_json_file(root / "study_budget.json", {})["approved"]
    assert (
        teacher_study.prepare_study(
            protocol,
            root,
            tokenizer=SavedTokenizer(),
            seeds=(42,),
            dataset_feedback="loss",
        )
        == plan
    )
    visited = []
    split.teacher_seeds = lambda: []

    def preview_runner(path, root):
        visited.append(Path(path).stem)
        return SimpleNamespace(
            data=split,
            categories=split.protocol["dataset"]["categories"],
            teacher=SimpleNamespace(generation_prompt=lambda *a, **k: "fixture prompt"),
        )

    monkeypatch.setattr(teacher_study, "ExperimentRunner", preview_runner)
    preview = teacher_study.preview_prompts(root, SavedTokenizer())
    assert len(preview["rows"]) == 32
    assert set(visited) == teacher_study.GENERATOR_ARMS
    assert preview["provider_calls"] == 0
    with pytest.raises(ValueError, match="fresh"):
        teacher_study.prepare_study(
            protocol,
            root,
            tokenizer=SavedTokenizer(),
            seeds=(42,),
            dataset_feedback="validation",
        )
