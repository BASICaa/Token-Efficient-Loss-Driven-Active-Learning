import unittest

from TEFLD.src.dataschema import CommandQuery, Instructor_Slot, Pipeline_State, User_Example
from TEFLD.src.difficulty import (
    difficulty_miss_reason,
    observed_difficulty,
    score_generated_difficulty,
    shift_difficulty,
)
from TEFLD.src.diversity_planner import DiversityPlanner


class DifficultyHelpersTest(unittest.TestCase):
    def test_observed_difficulty_detects_direct_easy_sample(self):
        score = score_generated_difficulty(
            sample="Ottawa is the capital of Canada.",
            instruct="What is the capital of Canada?",
            output="Ottawa.",
        )

        self.assertEqual(observed_difficulty(score), "easy")

    def test_hard_request_with_direct_sample_is_a_miss(self):
        score = score_generated_difficulty(
            sample="Ottawa is the capital of Canada.",
            instruct="What is the capital of Canada?",
            output="Ottawa.",
        )

        reason = difficulty_miss_reason("hard", observed_difficulty(score), score)

        self.assertIsNotNone(reason)
        self.assertIn("requested hard", reason)

    def test_shift_difficulty_clamps_to_bounds(self):
        self.assertEqual(shift_difficulty("easy", -1), "easy")
        self.assertEqual(shift_difficulty("medium", 1), "medium_hard")
        self.assertEqual(shift_difficulty("hard", 1), "hard")

    def test_planner_blueprint_includes_difficulty_budget(self):
        pipeline = Pipeline_State(
            section_id="",
            user_examples=[
                User_Example(
                    sample="What is the capital city of Canada?",
                    output="Ottawa.",
                )
            ],
        )
        slot = Instructor_Slot(
            slot_id=1,
            command=CommandQuery.GENERATE,
            tag="wide_variety",
        )

        blueprint = DiversityPlanner(pipeline=pipeline, ledger=[]).plan(
            slot=slot,
            generation_index=0,
            round_tag_counts={},
        )

        self.assertIn("reasoning", blueprint.difficulty_budget)
        self.assertIn("Difficulty budget:", blueprint.as_constraints())


if __name__ == "__main__":
    unittest.main()
