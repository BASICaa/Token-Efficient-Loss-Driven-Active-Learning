import unittest

from TEFLD.src.orchestrator import Orchestrator


class OrchestratorExampleParsingTest(unittest.TestCase):
    def test_parse_plain_text_examples(self):
        examples = Orchestrator.parse_text_examples(
            "Sample:\nWhat is 2+2?\n\nOutput:\n4\n",
            "plain",
        )

        self.assertEqual(len(examples), 1)
        self.assertEqual(examples[0].sample, "What is 2+2?")
        self.assertEqual(examples[0].output, "4")
        self.assertEqual(examples[0].mode, "plain")

    def test_parse_contextual_text_examples(self):
        examples = Orchestrator.parse_text_examples(
            (
                "Instruction:\nWhen did the library open?\n\n"
                "Context:\nThe library opened in 1984 and expanded in 2001.\n\n"
                "Output:\nThe library opened in 1984.\n"
            ),
            "contextual",
        )

        self.assertEqual(len(examples), 1)
        self.assertEqual(examples[0].instruct, "When did the library open?")
        self.assertEqual(examples[0].context, "The library opened in 1984 and expanded in 2001.")
        self.assertEqual(examples[0].mode, "contextual")

    def test_example_from_json_accepts_contextual_shape(self):
        example = Orchestrator.example_from_json(
            {
                "question": "Which train leaves first?",
                "context": "Train A leaves at 08:15. Train B leaves at 09:10.",
                "output": "Train A leaves first.",
            },
            "auto",
        )

        self.assertIsNotNone(example)
        self.assertEqual(example.mode, "contextual")
        self.assertEqual(example.output, "Train A leaves first.")


if __name__ == "__main__":
    unittest.main()
