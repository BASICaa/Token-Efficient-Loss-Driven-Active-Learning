from types import SimpleNamespace
from unittest.mock import Mock

from TEFLD.src.candidate_pool import CandidatePool, Teacher, similar_question
from TEFLD.src.dataschema import Training_Sample
from TEFLD.src.helper import read_json_file
from test_correctness import Tokenizer


def sample(question="Why do rainbows sometimes appear after rain?"):
    return Training_Sample(
        instruct=question,
        sample="",
        gold_summary="Sunlight is refracted and reflected by water droplets.",
        tag="general_qa",
        source="generated",
    )


def pool(tmp_path, generated):
    teacher = SimpleNamespace(
        identity="fixture",
        generate=Mock(side_effect=generated),
        verify=Mock(return_value={"status": "accepted_probabilistic_review"}),
    )
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
    return CandidatePool(data, Tokenizer(), teacher, tmp_path / "quarantine.json")


def test_retry_receives_duplicate_question_and_rejection_feedback(tmp_path):
    first = sample()
    novel = sample("How do plants benefit from pollinating insects?")
    candidate_pool = pool(tmp_path, [first, first, novel])
    seen = set()
    candidate_pool.create("general_qa", "slot:0", seen)
    accepted = candidate_pool.create("general_qa", "slot:1", seen)
    assert accepted.instruct == novel.instruct
    call = candidate_pool.teacher.generate.call_args
    assert first.instruct in call.kwargs["avoid_questions"]
    assert "Duplicate generated training question." in call.kwargs["rejection_feedback"]
    assert candidate_pool.teacher.verify.call_count == 2


def test_interrupted_recipe_reuses_accepted_admission_without_api(tmp_path):
    candidate_pool = pool(tmp_path, [sample()])
    original = candidate_pool.create("general_qa", "slot:0", set())
    resumed = pool(tmp_path, [])
    result = resumed.create("general_qa", "slot:0", set())
    assert result == original
    resumed.teacher.generate.assert_not_called()
    resumed.teacher.verify.assert_not_called()
    assert (
        read_json_file(tmp_path / "candidate_admissions.json", {})["slot:0"][
            "teacher_identity"
        ]
        == "fixture"
    )


def test_semantic_template_contexts_and_near_duplicate_questions():
    left = sample(
        "Why does a metal spoon feel colder than a wooden spoon when both are at room temperature?"
    )
    right = sample(
        "Why does a metal spoon feel colder than a wooden spoon when both sit at room temperature?"
    )
    assert similar_question(left, right)
    assert not similar_question(left, sample("Why do seasons change during the year?"))
    assert not similar_question(
        left.model_copy(update={"sample": "context A"}),
        left.model_copy(update={"sample": "context B"}),
    )


def test_generation_topic_and_avoidance_are_in_prompt_and_cache_identity(tmp_path):
    teacher = Teacher(tmp_path)
    teacher.ask = Mock(return_value=sample().model_dump(mode="json"))
    teacher.generate(
        "general_qa",
        [],
        "slot:0:0",
        avoid_questions=["OLD QUESTION"],
        rejection_feedback=["duplicate"],
    )
    first = teacher.ask.call_args.args[0]
    teacher.generate("general_qa", [], "slot:0:1", avoid_questions=["OLD QUESTION"])
    second = teacher.ask.call_args.args[0]
    assert (
        "assigned_topic" in first and "OLD QUESTION" in first and "duplicate" in first
    )
    assert first != second
