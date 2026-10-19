"""One answer-only encoding contract for training, scoring and generation."""

from __future__ import annotations

from typing import Any


def user_content(sample) -> str:
    text = sample.instruct.strip()
    if sample.sample.strip():
        text += "\n\nInput:\n" + sample.sample.strip()
    return text


def render_prompt(sample, tokenizer: Any) -> str:
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": user_content(sample)}],
            tokenize=False,
            add_generation_prompt=True,
        )
    parts = ["### Instruction:", sample.instruct.strip()]
    if sample.sample.strip():
        parts += ["", "### Input:", sample.sample.strip()]
    parts += ["", "### Output:"]
    return (getattr(tokenizer, "bos_token", None) or "") + "\n".join(parts) + "\n"


def encode_sample(sample, tokenizer: Any, max_length: int = 512) -> dict:
    import torch

    if not sample.instruct.strip() or not (sample.gold_summary or "").strip():
        raise ValueError("Nonempty instruction and gold answer are required.")
    prompt = render_prompt(sample, tokenizer)
    answer = sample.gold_summary.strip()
    if getattr(tokenizer, "chat_template", None):
        full = tokenizer.apply_chat_template(
            [
                {"role": "user", "content": user_content(sample)},
                {"role": "assistant", "content": answer},
            ],
            tokenize=False,
            add_generation_prompt=False,
        )
        if not full.startswith(prompt):
            raise ValueError("Chat template does not preserve the generation prefix.")
        suffix = full[len(prompt) :]
    else:
        suffix = answer + (getattr(tokenizer, "eos_token", None) or "")
    prefix_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    target_ids = tokenizer(suffix, add_special_tokens=False)["input_ids"]
    if not prefix_ids or not target_ids:
        raise ValueError(
            "Encoding requires a prompt and positive shifted answer supervision."
        )
    if len(prefix_ids) + len(target_ids) > max_length:
        raise ValueError(
            f"Complete prompt/answer exceeds {max_length} tokens; truncation is forbidden."
        )
    ids = torch.tensor(prefix_ids + target_ids, dtype=torch.long)
    labels = ids.clone()
    labels[: len(prefix_ids)] = -100
    if int((labels[1:] != -100).sum()) == 0:
        raise ValueError("No shifted supervised answer tokens.")
    return {"input_ids": ids, "attention_mask": torch.ones_like(ids), "labels": labels}


def collate_answer_only(examples: list[dict], pad_token_id: int) -> dict:
    import torch

    if not examples:
        raise ValueError("Cannot collate an empty batch.")
    width = max(len(x["input_ids"]) for x in examples)
    batch = {}
    for name, fill in [
        ("input_ids", pad_token_id),
        ("attention_mask", 0),
        ("labels", -100),
    ]:
        batch[name] = torch.stack(
            [
                torch.nn.functional.pad(x[name], (0, width - len(x[name])), value=fill)
                for x in examples
            ]
        )
    return batch


def supervised_tokens(encoded: dict) -> int:
    return int((encoded["labels"][..., 1:] != -100).sum().item())
