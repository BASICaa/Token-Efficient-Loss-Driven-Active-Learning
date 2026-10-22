import unittest

from TEFLD.src.dataschema import failure_vault_item
from TEFLD.src.evaluator import Evaluator


class EvaluatorVaultStatusTest(unittest.TestCase):
    def test_vault_item_status_marks_cooling_down(self):
        item = failure_vault_item(
            ledger_id=1,
            tag="assistant_qa",
            token_length=20,
            loss=1.2,
            recycle_count=1,
            last_recycled_round=4,
        )

        self.assertEqual(Evaluator.vault_item_status(item, current_round=5), "cooling_down")

    def test_vault_item_status_marks_exhausted_before_cooldown(self):
        item = failure_vault_item(
            ledger_id=1,
            tag="assistant_qa",
            token_length=20,
            loss=1.2,
            recycle_count=2,
            last_recycled_round=5,
        )

        self.assertEqual(Evaluator.vault_item_status(item, current_round=5), "exhausted")


if __name__ == "__main__":
    unittest.main()
