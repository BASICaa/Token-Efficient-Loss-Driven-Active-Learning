import sys
import types
import unittest

from TEFLD.src.dataschema import Training_Sample
from TEFLD.src.student import TrainyModel


class FakeTensor:
    def __init__(self, values):
        self.values = list(values)

    def squeeze(self, _dim):
        return self

    def clone(self):
        return FakeTensor(self.values)

    def to(self, dtype=None):
        self.dtype = dtype
        return self

    def __eq__(self, other):
        return [value == other for value in self.values]

    def __setitem__(self, key, value):
        if isinstance(key, slice):
            indexes = range(*key.indices(len(self.values)))
            for index in indexes:
                self.values[index] = value
            return

        if isinstance(key, list):
            for index, should_set in enumerate(key):
                if should_set:
                    self.values[index] = value
            return

        self.values[key] = value


class FakeTokenizer:
    eos_token = "<eos>"

    def __init__(self):
        self.calls = []

    def __call__(
        self,
        text,
        *,
        add_special_tokens=True,
        max_length=None,
        padding=None,
        truncation=False,
        return_tensors=None,
    ):
        self.calls.append(
            {
                "text": text,
                "add_special_tokens": add_special_tokens,
                "return_tensors": return_tensors,
            }
        )
        ids = [ord(char) % 97 + 3 for char in text]
        if add_special_tokens:
            ids = [1] + ids

        if max_length is not None:
            ids = ids[:max_length]
            attention = [1] * len(ids)
            if padding == "max_length":
                pad_count = max_length - len(ids)
                ids = ids + [0] * pad_count
                attention = attention + [0] * pad_count
        else:
            attention = [1] * len(ids)

        if return_tensors == "pt":
            return {
                "input_ids": FakeTensor(ids),
                "attention_mask": FakeTensor(attention),
            }
        return {"input_ids": ids, "attention_mask": attention}


class StudentMaskingTest(unittest.TestCase):
    def test_prompt_and_padding_tokens_are_masked_with_aligned_special_tokens(self):
        fake_torch = types.SimpleNamespace(long="long")
        previous_torch = sys.modules.get("torch")
        sys.modules["torch"] = fake_torch
        try:
            model = TrainyModel.__new__(TrainyModel)
            model.base_model_location = "mock-causal-lm"
            tokenizer = FakeTokenizer()
            sample = Training_Sample(
                sample="2 + 2",
                instruct="Answer the math question.",
                gold_summary="4",
                source="generated",
                tag="math_qa",
            )

            encoded = model.encode_training_sample(sample, tokenizer, max_length=128)
        finally:
            if previous_torch is None:
                sys.modules.pop("torch", None)
            else:
                sys.modules["torch"] = previous_torch

        prompt_call, full_call = tokenizer.calls[:2]
        self.assertFalse(prompt_call["add_special_tokens"])
        self.assertFalse(full_call["add_special_tokens"])

        prompt_length = len(prompt_call["text"])
        full_length = len(full_call["text"])
        labels = encoded["labels"].values

        self.assertTrue(all(value == -100 for value in labels[:prompt_length]))
        self.assertTrue(any(value != -100 for value in labels[prompt_length:full_length]))
        self.assertTrue(all(value == -100 for value in labels[full_length:]))


if __name__ == "__main__":
    unittest.main()
