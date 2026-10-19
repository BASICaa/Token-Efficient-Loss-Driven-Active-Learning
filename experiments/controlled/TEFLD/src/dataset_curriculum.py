"""TEFLD retrieval from immutable training references; no generation or relabeling."""

from __future__ import annotations

import copy

from .candidate_pool import candidate_id
from .encoding import encode_sample
from .validation import validate_sample

DATASET_METHODS = {
    "tefld_dataset_loss": "training_ledger_loss",
    "tefld_dataset_validation": "curriculum_validation_loss",
}
POLICY_VERSION = "dataset-tefld-v1"


def dataset_protocol(base, feedback="loss"):
    if feedback not in {"loss", "validation", "both"}:
        raise ValueError("Unknown dataset feedback variant.")
    result = copy.deepcopy(base)
    methods = [
        m for m in DATASET_METHODS if feedback == "both" or m.endswith("_" + feedback)
    ]
    result["dataset_tefld"] = {
        "version": POLICY_VERSION,
        "source_role": "training_pool",
        "enabled_methods": methods,
        "match_baseline_reference_gates": True,
        "initial_batch": "same_seeded_reference_batch_as_dolly_baseline",
        "new_examples": "seeded_unused_reference_retrieval_within_weak_category_quotas",
        "replay": "same_loss_threshold_0_35_max_3_slots_use_limit_2_cooldown_as_synthetic_tefld",
        "exploration_slots": 4,
        "round_slots": 10,
        "original_reference_answers_unchanged": True,
        "synthetic_generation": False,
        "semantic_verifier_requests": False,
        "teacher_api_requests": False,
        "exhaustion": "fail_before_training_round_without_synthetic_fallback",
        "primary_ready": False,
        "remaining_gate": "Freeze a common optimizer recipe after the optimizer diagnostic and equal development tuning.",
    }
    result["methods"] = [
        m for m in result.get("methods", []) if m["id"] not in DATASET_METHODS
    ]
    result["methods"] += [
        {
            "id": method,
            "training_source": "original_training_pool_references_plus_training_replay",
            "feedback_signal": DATASET_METHODS[method],
            "development_data_affects_curriculum": method.endswith("_validation"),
            "synthetic_generation": False,
            "teacher_model": None,
            "primary_contrast": "dolly_baseline",
        }
        for method in methods
    ]
    result["status"] = "dataset_tefld_control_prepared_optimizer_decision_pending"
    result["design_date"] = "2026-10-19"
    return result


def require_dataset_policy(protocol):
    policy = protocol.get("dataset_tefld", {})
    if (
        policy.get("version") != POLICY_VERSION
        or policy.get("source_role") != "training_pool"
        or policy.get("match_baseline_reference_gates") is not True
        or policy.get("round_slots") != 10
        or policy.get("exploration_slots") != 4
        or policy.get("original_reference_answers_unchanged") is not True
        or policy.get("synthetic_generation") is not False
        or policy.get("teacher_api_requests") is not False
        or policy.get("semantic_verifier_requests") is not False
    ):
        raise ValueError("Use the dataset-TEFLD protocol with matched reference gates.")
    enabled = policy.get("enabled_methods", [])
    if (
        not enabled
        or len(set(enabled)) != len(enabled)
        or any(m not in DATASET_METHODS for m in enabled)
    ):
        raise ValueError("Declare distinct dataset feedback methods.")
    return policy


def reference_sample(data, row_id, tokenizer, guard):
    if type(row_id) is not int or row_id not in data.ids["training_pool"]:
        raise PermissionError("Dataset TEFLD can retrieve only training_pool rows.")
    sample = data.sample(row_id, "training_pool")
    row = data.rows[row_id]
    if (sample.instruct, sample.sample, sample.gold_summary, sample.tag) != (
        row["instruction"],
        row["context"],
        row["response"],
        row["category"],
    ):
        raise ValueError("Dataset reference content changed during retrieval.")
    validate_sample(sample).require()
    guard.require(sample)
    encode_sample(sample, tokenizer, data.protocol["common_training"]["max_length"])
    sample.verification.update(
        status="dataset_reference", row_id=row_id, role="training_pool"
    )
    return sample


def require_reference_provenance(data, sample, tokenizer, guard):
    metadata = sample.verification
    if (
        metadata.get("role") != "training_pool"
        or metadata.get("status") != "dataset_reference"
    ):
        raise ValueError("Missing training-only dataset provenance.")
    canonical = reference_sample(data, metadata.get("row_id"), tokenizer, guard)
    if (sample.instruct, sample.sample, sample.gold_summary, sample.tag) != (
        canonical.instruct,
        canonical.sample,
        canonical.gold_summary,
        canonical.tag,
    ):
        raise ValueError("Saved dataset example differs from its original reference.")


def retrieve_references(data, tokenizer, guard, order, seen, quotas, exploration):
    """Use the seeded per-category order, without replacing previously admitted prompts."""
    selected, trace = [], []
    if not set(order) <= data.ids["training_pool"]:
        raise PermissionError("Dataset retrieval order contains a protected row.")
    available = set(seen)
    for category, kind in [
        (category, "focus")
        for category, count in sorted(quotas.items())
        for _ in range(count)
    ] + [(category, "exploration") for category in exploration]:
        for row_id in order:
            if data.rows[row_id]["category"] != category:
                continue
            sample = data.sample(row_id, "training_pool")
            identity = candidate_id(sample)
            if identity in available:
                continue
            try:
                sample = reference_sample(data, row_id, tokenizer, guard)
            except ValueError:
                continue
            selected.append(sample)
            trace.append(
                {
                    "row_id": row_id,
                    "candidate_id": identity,
                    "category": category,
                    "slot": kind,
                }
            )
            available.add(identity)
            break
        else:
            raise RuntimeError(
                f"Dataset-only {kind} quota exhausted for {category}; no synthetic fallback; round not trained."
            )
    seen.update(available)
    return selected, trace
