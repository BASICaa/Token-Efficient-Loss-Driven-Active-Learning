"""Study summaries with paired seeds and passage-group-aware uncertainty."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import random
import statistics

from .candidate_pool import normalize
from .helper import read_json_file, write_json_file


def paired_interval(left, right, data, repetitions=1000, seed=20261018):
    seeds = sorted(set(left) & set(right))
    if not seeds:
        raise ValueError("No paired seeds.")
    common = set.intersection(*(set(left[s]) & set(right[s]) for s in seeds))
    groups = defaultdict(list)
    for row_id in sorted(common):
        row = data.rows[row_id]
        context = normalize(row["context"])
        key = context if len(context) >= 80 else f"row:{row_id}"
        groups[key].append(row_id)
    if not groups:
        raise ValueError("No paired evaluation groups.")
    rng = random.Random(seed)
    keys = list(groups)
    draws = []
    for _ in range(repetitions):
        picked_seeds = rng.choices(seeds, k=len(seeds))
        picked_groups = rng.choices(keys, k=len(keys))
        categories = defaultdict(list)
        for s in picked_seeds:
            for group in picked_groups:
                for i in groups[group]:
                    categories[data.rows[i]["category"]].append(
                        left[s][i] - right[s][i]
                    )
        draws.append(statistics.mean(statistics.mean(v) for v in categories.values()))
    draws.sort()
    return {
        "lower": draws[int(0.025 * (len(draws) - 1))],
        "upper": draws[int(0.975 * (len(draws) - 1))],
        "paired_seeds": len(seeds),
        "passage_groups": len(groups),
        "scope": "Bootstrap over observed seeds and passage groups; few seeds limit population inference.",
    }


def summarize_study(directory, data):
    directory = Path(directory)
    frozen = read_json_file(directory / "frozen_study.json", {})
    if not frozen.get("test_scored"):
        raise ValueError(
            "Final scoring is incomplete; no completed-study report can be produced."
        )
    results = read_json_file(directory / "final_test_results.json", [])
    if not results:
        raise ValueError(
            "No final evaluation results; the report cannot invent a completed comparison."
        )
    expected = {
        (m, s, k)
        for m in frozen["methods"]
        for s in frozen["seeds"]
        for k in ["final_checkpoint", "selected_checkpoint"]
    }
    actual = [(r["method"], r["seed"], r["checkpoint_kind"]) for r in results]
    if set(actual) != expected or len(actual) != len(expected):
        raise ValueError("Missing or duplicate final results.")
    summaries = {}
    paired = defaultdict(dict)
    for kind in ["final_checkpoint", "selected_checkpoint"]:
        by_method = defaultdict(list)
        for row in results:
            if row["checkpoint_kind"] == kind:
                by_method[row["method"]].append(row)
        summaries[kind] = {}
        for method, rows in by_method.items():
            values = [r["evaluation"]["macro_nll"] for r in rows]
            summaries[kind][method] = {
                "mean_macro_nll": statistics.mean(values),
                "std_across_seeds": (
                    statistics.stdev(values) if len(values) > 1 else None
                ),
                "seeds": len(values),
                "actual_tokens": [r["tokens"] for r in rows],
            }
            if kind == "final_checkpoint":
                paired[method] = {
                    r["seed"]: {x["row_id"]: x["loss"] for x in r["evaluation"]["rows"]}
                    for r in rows
                }
    contrasts = {}
    for left, right in [
        ("tefld_validation", "tefld_loss"),
        ("tefld_validation", "dolly_baseline"),
        ("tefld_loss", "dolly_baseline"),
        ("tefld_benefit", "tefld_validation"),
        ("tefld_dataset_loss", "dolly_baseline"),
        ("tefld_dataset_validation", "dolly_baseline"),
        ("tefld_dataset_validation", "tefld_dataset_loss"),
        ("tefld_loss", "tefld_dataset_loss"),
        ("tefld_validation", "tefld_dataset_validation"),
    ]:
        if left in paired and right in paired:
            contrasts[left + " minus " + right] = paired_interval(
                paired[left], paired[right], data
            )
    usage = read_json_file(directory / "teacher_cache/usage.json", {"requests": []})
    report = {
        "status": "completed_NLL_comparison",
        "teacher_known_cost_toman": sum(
            x.get("cost_toman", 0) for x in usage["requests"]
        ),
        "teacher_unknown_usage_reservations_toman": sum(
            x.get("reserved_cost_toman", 0)
            for x in usage["requests"]
            if "cost_toman" not in x
        ),
        "summaries": summaries,
        "paired_intervals": contrasts,
        "teacher_known_cost_usd": sum(x.get("cost_usd", 0) for x in usage["requests"]),
        "teacher_unknown_usage_reservations_usd": sum(
            x.get("reserved_cost_usd", 0)
            for x in usage["requests"]
            if "cost_usd" not in x
        ),
        "teacher_requests": len(usage["requests"]),
        "answer_quality_status": "requires_generated_answers_and_blinded_review; NLL alone does not establish answer quality",
        "auxiliary_compute_status": "include auxiliary/selector state and probe logs before claiming efficiency",
    }
    write_json_file(directory / "study_report.json", report)
    return report


def export_blinded_review(directory, data):
    directory = Path(directory)
    frozen = read_json_file(directory / "frozen_study.json", {})
    if not frozen.get("test_scored"):
        raise PermissionError("Review export requires completed frozen final scoring.")
    answers = read_json_file(directory / "generated_answers.json", [])
    if not answers:
        raise ValueError(
            "Run final evaluation with generated answers before review export."
        )
    expected = {
        (m, s, k)
        for m in frozen["methods"]
        for s in frozen["seeds"]
        for k in ["final_checkpoint", "selected_checkpoint"]
    }
    actual = [(r["method"], r["seed"], r["checkpoint_kind"]) for r in answers]
    if set(actual) != expected or len(actual) != len(expected):
        raise ValueError("Generated answer conditions are incomplete or duplicated.")
    conditions = sorted(
        {(r["method"], r["seed"], r["checkpoint_kind"]) for r in answers}
    )
    rng = random.Random(20261018)
    codes = [f"R{i:03d}" for i in range(len(conditions))]
    rng.shuffle(codes)
    mapping = dict(zip(conditions, codes))
    selected = set(
        data.protocol["metrics"]["generated_answers"]["blinded_review_row_ids"]
    )
    review = []
    for condition in answers:
        ids = [row["row_id"] for row in condition["rows"]]
        if len(ids) != len(set(ids)) or not selected.issubset(ids):
            raise ValueError("Generated review rows are missing or duplicated.")
        for row in condition["rows"]:
            if row["row_id"] not in selected:
                continue
            reference = data.rows[row["row_id"]]
            review.append(
                {
                    "run_code": mapping[
                        (
                            condition["method"],
                            condition["seed"],
                            condition["checkpoint_kind"],
                        )
                    ],
                    "row_id": row["row_id"],
                    "category": row["category"],
                    "instruction": reference["instruction"],
                    "context": reference["context"],
                    "reference_answer": reference["response"],
                    "student_answer": row["answer"],
                    "reviewer_1": {
                        "correctness": None,
                        "instruction_following": None,
                        "completeness": None,
                    },
                    "reviewer_2": {
                        "correctness": None,
                        "instruction_following": None,
                        "completeness": None,
                    },
                }
            )
    rng.shuffle(review)
    write_json_file(directory / "blinded_review.json", review)
    write_json_file(
        directory / "blinded_review_key.json",
        [
            {
                "run_code": mapping[c],
                "method": c[0],
                "seed": c[1],
                "checkpoint_kind": c[2],
            }
            for c in conditions
        ],
    )
    return {"items": len(review), "status": "awaiting_two_independent_reviewers"}
