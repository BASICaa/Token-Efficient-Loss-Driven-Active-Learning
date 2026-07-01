import unittest

from TEFLD.src.dataschema import Pipeline_State, User_Example, learning_record
from TEFLD.src.diversity_planner import (
    ASSISTANT_QA_SHAPE_ID,
    STRUCTURED_TEXT_SHAPE_ID,
    DiversityPlanner,
)


def structured_record(ledger_id: int) -> learning_record:
    return learning_record(
        ledger_id=ledger_id,
        sample="Items: apple, hammer, train station.",
        instruct="Classify each item as food, tool, or place. Return a JSON object.",
        gold_summary='{"apple":"food","hammer":"tool","train station":"place"}',
        token_length=40,
        loss=0.5,
        tag="structured_text_task",
        round_id=ledger_id // 10,
        source="generated",
    )


class DiversityPlannerTaskShapeProfileTest(unittest.TestCase):
    def test_task_shape_profile_prefers_recent_ledger_when_available(self):
        pipeline = Pipeline_State(
            section_id="section_test",
            user_examples=[
                User_Example(
                    sample="What is the capital city of Canada?",
                    output="The capital city of Canada is Ottawa.",
                )
            ],
        )
        ledger = [structured_record(index) for index in range(1, 11)]

        profile = DiversityPlanner(pipeline=pipeline, ledger=ledger).task_shape_profile()

        self.assertEqual(profile["shape_id"], STRUCTURED_TEXT_SHAPE_ID)
        self.assertIn("recent_learning_ledger", profile["source"])
        self.assertIn("original_user_examples", profile["source"])
        self.assertEqual(profile["user_example_shape_id"], ASSISTANT_QA_SHAPE_ID)

    def test_task_shape_profile_falls_back_to_user_examples_without_ledger(self):
        pipeline = Pipeline_State(
            section_id="section_test",
            user_examples=[
                User_Example(
                    sample="What gas do plants release during photosynthesis?",
                    output="Plants release oxygen during photosynthesis.",
                )
            ],
        )

        profile = DiversityPlanner(pipeline=pipeline, ledger=[]).task_shape_profile()

        self.assertEqual(profile["shape_id"], ASSISTANT_QA_SHAPE_ID)
        self.assertEqual(profile["source"], ["original_user_examples"])


if __name__ == "__main__":
    unittest.main()
