"""Frozen split loader and standalone development scoring; no section globals."""

from __future__ import annotations

from collections import defaultdict
import hashlib
import json
from pathlib import Path

from .dataschema import Training_Sample
from .encoding import encode_sample, supervised_tokens, render_prompt
from .training_safety import require_finite


class SplitData:
    def __init__(self, protocol_path):
        self.protocol_path = Path(protocol_path).resolve()
        self.protocol = json.loads(self.protocol_path.read_text(encoding="utf-8"))
        dataset = self.protocol["dataset"]
        self.manifest_path = Path(dataset["manifest"])
        self.dataset_path = Path(dataset["source"])
        for path, expected in [
            (self.manifest_path, dataset["manifest_sha256"]),
            (self.dataset_path, dataset["sha256"]),
        ]:
            if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
                raise ValueError(f"Frozen data fingerprint changed: {path.name}")
        self.manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        self.rows = [
            json.loads(line)
            for line in self.dataset_path.read_text(encoding="utf-8-sig").splitlines()
            if line.strip()
        ]
        self.ids = {name: set(ids) for name, ids in self.manifest["splits"].items()}
        roles = list(self.ids)
        for i, name in enumerate(roles):
            for other in roles[i + 1 :]:
                if self.ids[name] & self.ids[other]:
                    raise ValueError("Dataset partitions overlap.")
        self.final_test_enabled = False

    def sample(self, row_id: int, role: str) -> Training_Sample:
        if role == "final_test" and not self.final_test_enabled:
            raise PermissionError(
                "Final test is locked until all planned runs and checkpoint selections finish."
            )
        if row_id not in self.ids[role]:
            raise PermissionError(f"Row {row_id} is not in {role}.")
        row = self.rows[row_id]
        return Training_Sample(
            sample=row["context"],
            instruct=row["instruction"],
            gold_summary=row["response"],
            source="generated",
            tag=row["category"],
            verification={
                "status": "dataset_reference",
                "row_id": row_id,
                "role": role,
            },
        )

    def samples(self, role: str, per_category: int | None = None):
        counts = defaultdict(int)
        selected = []
        for row_id in sorted(self.ids[role]):
            category = self.rows[row_id]["category"]
            if per_category is None or counts[category] < per_category:
                selected.append((row_id, self.sample(row_id, role)))
                counts[category] += 1
        return selected

    def teacher_seeds(self):
        return [
            self.sample(i, "training_pool")
            for i in self.protocol["tefld_controls"]["seed_example_row_ids"]
        ]


def score_samples(model, tokenizer, samples, max_length=512):
    """Return macro example NLL and target-weighted NLL; restore module modes."""
    import torch

    modes = [(module, module.training) for module in model.modules()]
    model.eval()
    device = next(model.parameters()).device
    rows = []
    try:
        with torch.no_grad():
            for row_id, sample in samples:
                encoded = encode_sample(sample, tokenizer, max_length)
                count = supervised_tokens(encoded)
                output = model(
                    **{k: v.unsqueeze(0).to(device) for k, v in encoded.items()}
                )
                loss = float(require_finite(output.loss).item())
                rows.append(
                    {
                        "row_id": row_id,
                        "category": sample.tag,
                        "loss": loss,
                        "answer_tokens": count,
                    }
                )
    finally:
        for module, mode in modes:
            module.training = mode
    if not rows:
        raise ValueError("No examples to score.")
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["category"]].append(row["loss"])
    categories = {k: sum(v) / len(v) for k, v in grouped.items()}
    return {
        "macro_nll": sum(categories.values()) / len(categories),
        "token_weighted_nll": sum(x["loss"] * x["answer_tokens"] for x in rows)
        / sum(x["answer_tokens"] for x in rows),
        "categories": categories,
        "rows": rows,
    }


def generate_answers(model, tokenizer, samples, max_new_tokens=512):
    import torch

    modes = [(module, module.training) for module in model.modules()]
    model.eval()
    device = next(model.parameters()).device
    rows = []
    try:
        with torch.no_grad():
            for row_id, sample in samples:
                inputs = tokenizer(
                    render_prompt(sample, tokenizer),
                    add_special_tokens=False,
                    return_tensors="pt",
                )
                inputs = {k: v.to(device) for k, v in inputs.items()}
                output = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    use_cache=True,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=getattr(model.generation_config, "eos_token_id", None)
                    or tokenizer.eos_token_id,
                )
                answer = tokenizer.decode(
                    output[0][inputs["input_ids"].shape[-1] :], skip_special_tokens=True
                ).strip()
                rows.append(
                    {"row_id": row_id, "category": sample.tag, "answer": answer}
                )
    finally:
        for module, mode in modes:
            module.training = mode
    return rows
