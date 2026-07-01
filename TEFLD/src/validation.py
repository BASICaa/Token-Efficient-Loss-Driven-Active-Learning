from __future__ import annotations

import json
from collections import defaultdict
from statistics import mean
from typing import Any

from .dataschema import Evaluation_Output, Training_Sample
from .diversity_planner import infer_task_shape_profile
from .evaluator import finite_loss_value
from .helper import (
    append_section_debug_log,
    call_client,
    create_client,
    get_learning_tag_counts,
    get_section_file_path,
    get_user_examples,
    read_json_file,
    write_json_file,
)
from .instructor import (
    Instructor,
    format_known_categories,
    format_user_example_for_prompt,
)
from .student import TrainyModel


SHARED_VALIDATION_FILENAME = "shared_validation.json"
DEFAULT_SHARED_VALIDATION_SIZE = 10


def get_shared_validation_path(section_id: str):
    return get_section_file_path(section_id, SHARED_VALIDATION_FILENAME)


def load_shared_validation_samples(section_id: str) -> list[Training_Sample]:
    raw_data = read_json_file(get_shared_validation_path(section_id), [])
    if isinstance(raw_data, dict):
        raw_samples = raw_data.get("samples") or raw_data.get("items") or []
    else:
        raw_samples = raw_data

    if not isinstance(raw_samples, list):
        raise ValueError(f"{SHARED_VALIDATION_FILENAME} must contain a JSON array.")

    return [Training_Sample.model_validate(row) for row in raw_samples]


def save_shared_validation_samples(
    section_id: str,
    samples: list[Training_Sample],
    *,
    metadata: dict[str, Any] | None = None,
) -> None:
    write_json_file(
        get_shared_validation_path(section_id),
        {
            "metadata": metadata or {},
            "samples": [sample.model_dump(mode="json") for sample in samples],
        },
    )


def build_shared_validation_prompt(
    *,
    examples_block: str,
    known_categories: str,
    validation_size: int,
) -> str:
    return (
        f"You create a fixed shared-validation set with {validation_size} samples "
        "for a small active-learning experiment.\n\n"
        "User examples that define the task family:\n"
        f"{examples_block}\n\n"
        "Known categories from prior generated data, if any:\n"
        f"{known_categories}\n\n"
        "Purpose:\n"
        "- This set is used only for checkpoint selection and rollback decisions.\n"
        "- It must not be copied into the training batch, ledger, or failure vault.\n"
        "- It should be broad and holdout-like, not a near-duplicate of the "
        "current TEFLD-generated training batches.\n"
        "- Cover a diverse mix of realistic user requests, reasoning patterns, "
        "input shapes, and output shapes that match the same broad distribution "
        "as a final holdout set.\n\n"
        "Return only one valid JSON array. Each item must include exactly these "
        "keys:\n"
        "sample\n"
        "instruct\n"
        "gold_summary\n"
        "tag\n\n"
        "Rules:\n"
        "- sample is the input/context shown to the student.\n"
        "- instruct is the exact user-facing task or question.\n"
        "- gold_summary is the expected answer/output.\n"
        "- tag is a concise snake_case category.\n"
        "- Reuse exact known category names when they fit.\n"
        "- Do not return markdown fences, explanations, or extra text."
    )


def generate_shared_validation_samples(
    *,
    section_id: str,
    model_id: str,
    validation_size: int = DEFAULT_SHARED_VALIDATION_SIZE,
) -> list[Training_Sample]:
    examples = get_user_examples(section_id)
    if not examples:
        raise ValueError("Cannot build shared validation without user examples.")

    examples_block = "\n\n---\n\n".join(
        format_user_example_for_prompt(example) for example in examples
    )
    prompt = build_shared_validation_prompt(
        examples_block=examples_block,
        known_categories=format_known_categories(get_learning_tag_counts(section_id)),
        validation_size=validation_size,
    )
    client = create_client()
    response = call_client(client=client, prompt=prompt, model_id=model_id)

    parser = Instructor(model_id=model_id)
    raw_payload = parser.parse_generated_json(response)
    raw_samples = (
        raw_payload.get("samples") or raw_payload.get("items")
        if isinstance(raw_payload, dict)
        else raw_payload
    )
    if not isinstance(raw_samples, list):
        raise ValueError("Shared validation response must be a JSON array.")
    if len(raw_samples) != validation_size:
        raise ValueError(
            "Shared validation response must contain exactly "
            f"{validation_size} samples, got {len(raw_samples)}."
        )

    samples = [
        parser.training_sample_from_generated_object(
            raw_sample=raw_sample,
            instruction="fixed shared validation generation",
            tag=str(raw_sample.get("tag") or "shared_validation")
            if isinstance(raw_sample, dict)
            else "shared_validation",
            tag_counts=get_learning_tag_counts(section_id),
        )
        for raw_sample in raw_samples
    ]
    save_shared_validation_samples(
        section_id,
        samples,
        metadata={
            "model_id": model_id,
            "validation_size": validation_size,
            "source": "generated_fixed_shared_validation",
        },
    )
    append_section_debug_log(
        section_id,
        "shared_validation_ready",
        {
            "sample_count": len(samples),
            "path": str(get_shared_validation_path(section_id)),
            "source": "generated_fixed_shared_validation",
        },
    )
    return samples


def ensure_shared_validation_samples(
    *,
    section_id: str,
    model_id: str,
    validation_size: int = DEFAULT_SHARED_VALIDATION_SIZE,
) -> list[Training_Sample]:
    samples = load_shared_validation_samples(section_id)
    if samples:
        if len(samples) != validation_size:
            raise ValueError(
                f"{SHARED_VALIDATION_FILENAME} must contain exactly "
                f"{validation_size} samples, got {len(samples)}."
            )
        return samples

    return generate_shared_validation_samples(
        section_id=section_id,
        model_id=model_id,
        validation_size=validation_size,
    )


def evaluate_shared_validation_samples(
    student: TrainyModel,
    samples: list[Training_Sample],
) -> list[Evaluation_Output]:
    return [
        student.evaluate_sample(sample, generate_answer=False)
        for sample in samples
    ]


def average_validation_loss(evaluations: list[Evaluation_Output]) -> float | None:
    losses = [
        loss
        for loss in (finite_loss_value(evaluation.loss) for evaluation in evaluations)
        if loss is not None
    ]
    return mean(losses) if losses else None


def build_validation_weakness_profile(
    *,
    evaluations: list[Evaluation_Output],
    round_id: int,
    source: str,
) -> dict[str, Any]:
    finite_evaluations = [
        evaluation
        for evaluation in evaluations
        if finite_loss_value(evaluation.loss) is not None
    ]
    avg_loss = average_validation_loss(finite_evaluations)
    groups: dict[str, list[Evaluation_Output]] = defaultdict(list)
    for evaluation in finite_evaluations:
        groups[evaluation.tag or "untagged"].append(evaluation)

    top_categories: list[dict[str, Any]] = []
    total = len(finite_evaluations)
    for tag, rows in groups.items():
        losses = [
            float(finite_loss_value(row.loss) or 0.0)
            for row in rows
        ]
        shape_profile = infer_task_shape_profile(
            rows,
            source="shared_validation_loss",
        )
        top_categories.append(
            {
                "tag": tag,
                "shape_id": shape_profile.get("shape_id"),
                "avg_loss": mean(losses),
                "max_loss": max(losses),
                "count": len(rows),
                "share": (len(rows) / total) if total else 0.0,
                "input_shape": shape_profile.get("input_shape"),
                "output_shape": shape_profile.get("output_shape"),
                "reasoning_pattern": shape_profile.get("reasoning_pattern"),
                "confidence": shape_profile.get("confidence"),
            }
        )

    top_categories = sorted(
        top_categories,
        key=lambda row: (-float(row["avg_loss"]), -int(row["count"]), row["tag"]),
    )
    for index, category in enumerate(top_categories, start=1):
        category["rank"] = index

    task_shape_profile = infer_task_shape_profile(
        finite_evaluations,
        source="shared_validation_loss",
    )
    return {
        "round_id": round_id,
        "source": source,
        "avg_loss": avg_loss,
        "sample_count": len(finite_evaluations),
        "task_shape_profile": task_shape_profile,
        "top_categories": top_categories[:5],
        "raw_losses": [
            {
                "index": index,
                "tag": evaluation.tag,
                "loss": finite_loss_value(evaluation.loss),
            }
            for index, evaluation in enumerate(finite_evaluations, start=1)
        ],
    }


def profile_to_json(profile: dict[str, Any]) -> str:
    return json.dumps(profile, indent=2, ensure_ascii=False)
