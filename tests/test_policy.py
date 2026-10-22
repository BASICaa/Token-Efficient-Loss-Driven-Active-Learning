import unittest

from TEFLD.src.dataschema import Pipeline_State
from TEFLD.src.policy import PolicyMaker, compose_generation_tag_plan


class PolicyHelpersTest(unittest.TestCase):
    def test_compose_generation_tag_plan_keeps_wide_variety_budget(self):
        plan = compose_generation_tag_plan(["math_qa", "science_qa"], 10)

        self.assertEqual(plan[:4], ["wide_variety"] * 4)
        self.assertEqual(
            plan[4:],
            ["math_qa", "science_qa", "math_qa", "science_qa", "math_qa", "science_qa"],
        )

    def test_percentile_uses_ordered_fraction_index(self):
        self.assertEqual(PolicyMaker.percentile([10, 1, 5, 20], 0.75), 10)

    def test_recent_validation_improvement_ignores_tiny_denominator(self):
        policy = PolicyMaker.__new__(PolicyMaker)
        policy.pipeline = Pipeline_State(
            round_health_history=[
                {"decision_loss": 1e-8},
                {"decision_loss": 0.5},
            ]
        )

        self.assertIsNone(policy.recent_validation_improvement())


if __name__ == "__main__":
    unittest.main()
