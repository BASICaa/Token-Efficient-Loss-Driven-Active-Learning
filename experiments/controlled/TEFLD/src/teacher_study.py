"""Offline setup and bounded prompt pilots for two generators and one fixed verifier."""

from __future__ import annotations

import copy
import hashlib
from pathlib import Path
import random

from .candidate_pool import CandidatePool, LeakageGuard, candidate_id
from .dataschema import Training_Sample
from .dataset_curriculum import dataset_protocol, reference_sample
from .encoding import encode_sample
from .experiment_runner import ExperimentRunner
from .helper import read_json_file, write_json_file, stable_json_hash
from .teacher_accounting import summarize_teacher_costs
from .transactions import section_lock
from .validation import validate_sample

MODELS = {"sol61": "openai/gpt-6.1-sol", "sol6": "openai/gpt-6-sol"}
GENERATOR_ARMS = {
    teacher + "-" + variant for teacher in MODELS for variant in ("full", "compact")
}


def comparison_protocol(base, teacher, variant, *, baseline=False):
    if teacher not in MODELS or variant not in {"full", "compact"}:
        raise ValueError("Unknown teacher study condition.")
    result = copy.deepcopy(base)
    controls = result["tefld_controls"]
    common_verifier = copy.deepcopy(controls["provider"])
    common_verifier["model_id"] = "openai/gpt-6.1-sol"
    controls["provider"]["model_id"] = MODELS[teacher]
    controls["teacher_model_id"] = MODELS[teacher]
    controls["teacher_family"] = MODELS[teacher].removeprefix("openai/")
    controls["generation_prompt_variant"] = variant
    controls["verifier"] = {
        "model_id": "openai/gpt-6.1-sol",
        "provider": common_verifier,
    }
    result["teacher_comparison"] = {
        "arm": "dolly" if baseline else teacher + "-" + variant,
        "generator_factor": None if baseline else MODELS[teacher],
        "prompt_variant": variant,
        "warmup_source": "shared_category_balanced_training_only_dolly_references",
        "shared_verifier": "openai/gpt-6.1-sol",
        "shared_budget_required": True,
        "primary_ready": False,
        "purpose": "development only; optimizer-control decision pending before primary freeze",
    }
    result["status"] = "prepared_offline_live_teacher_and_prompt_comparison_pending"
    # Remove inherited live notes and obsolete per-study recommendations.
    controls["hibana_notes"] = {
        "wire_ids": "Provider-prefixed IDs require live acceptance; Sol 6 has not been called.",
        "temperature": "1.0 requested, no reasoning override; supported parameters must pass a live development check.",
        "pricing": "Both catalogue models use the same checked rates; compare actual usage and accepted-candidate cost.",
    }
    return result


def prepare_study(
    base_path, root, *, tokenizer, seeds=(42, 123, 2026), dataset_feedback=None
):
    base_path = Path(base_path).resolve()
    root = Path(root).resolve()
    base = read_json_file(base_path, {})
    if dataset_feedback is not None:
        base = dataset_protocol(base, dataset_feedback)
    with section_lock(root):
        plan_path = root / "study_plan.json"
        identity = stable_json_hash(
            {
                "base": base,
                "seeds": list(seeds),
                "version": (
                    "teacher-study-v2-dataset"
                    if dataset_feedback is not None
                    else "teacher-study-v1"
                ),
            }
        )
        previous = read_json_file(plan_path, {})
        if previous and previous.get("identity") != identity:
            raise ValueError("Study setup changed; use a fresh study directory.")
        profiles = {}
        for teacher in MODELS:
            for variant in ("full", "compact"):
                key = teacher + "-" + variant
                profile = comparison_protocol(base, teacher, variant)
                path = root / "protocols" / (key + ".json")
                if path.exists() and read_json_file(path, {}) != profile:
                    raise ValueError(
                        "Prepared profile changed; use a fresh study directory."
                    )
                write_json_file(path, profile)
                profiles[key] = str(path)
        baseline_path = root / "protocols/dolly.json"
        baseline = comparison_protocol(base, "sol61", "compact", baseline=True)
        if baseline_path.exists() and read_json_file(baseline_path, {}) != baseline:
            raise ValueError("Prepared baseline changed; use a fresh study directory.")
        write_json_file(baseline_path, baseline)
        profiles["dolly"] = str(baseline_path)
        if dataset_feedback is not None:
            profile = comparison_protocol(base, "sol61", "compact", baseline=True)
            profile["teacher_comparison"]["arm"] = "dataset-refs"
            profile["teacher_comparison"][
                "purpose"
            ] = "Teacher-free adaptive dataset selection; common optimizer decision pending."
            path = root / "protocols/dataset-refs.json"
            if path.exists() and read_json_file(path, {}) != profile:
                raise ValueError(
                    "Prepared dataset profile changed; use a fresh study directory."
                )
            write_json_file(path, profile)
            profiles["dataset-refs"] = str(path)
        # Read only training-pool rows; no teacher request or held-out text export.
        runner = ExperimentRunner(profiles["sol61-compact"], root)
        guard = LeakageGuard(runner.data)
        for seed in seeds:
            warmup_path = root / f"shared_initial_seed_{seed}.json"
            batch = []
            groups = set()
            rng = random.Random(seed)
            categories = runner.categories + [
                rng.choice(runner.categories) for _ in range(2)
            ]
            for category in categories:
                ids = sorted(
                    i
                    for i in runner.data.ids["training_pool"]
                    if runner.data.rows[i]["category"] == category
                )
                rng.shuffle(ids)
                for row_id in ids:
                    sample = runner.data.sample(row_id, "training_pool")
                    context = " ".join(sample.sample.casefold().split())
                    group = context if len(context) >= 80 else candidate_id(sample)
                    if group in groups:
                        continue
                    try:
                        if dataset_feedback is not None:
                            sample = reference_sample(
                                runner.data, row_id, tokenizer, guard
                            )
                        validate_sample(sample).require()
                        guard.require(sample)
                        encode_sample(sample, tokenizer)
                    except ValueError:
                        continue
                    batch.append(sample.model_dump(mode="json"))
                    groups.add(group)
                    break
                else:
                    raise RuntimeError("Cannot construct diverse training-only warmup.")
            old = read_json_file(warmup_path, None)
            if old is not None and old != batch:
                raise ValueError("Shared warmup changed; preserve the prepared study.")
            write_json_file(warmup_path, batch)
        budget_path = root / "study_budget.json"
        if not budget_path.exists():
            write_json_file(
                budget_path,
                {
                    "approved": False,
                    "currency": "toman",
                    "max_spend": 0,
                    "scope": "One total cap across both generators, fixed verifier, retries, rejected samples, prompt pilots and audits in this study.",
                    "note": "No new paid budget authorized; do not reuse the previous study cap as fresh credit.",
                },
            )
        plan = {
            "identity": identity,
            "profiles": profiles,
            "seeds": list(seeds),
            "main_training_conditions": [
                "sol61-compact:tefld_validation",
                "sol61-compact:tefld_loss",
                "sol6-compact:tefld_validation",
                "sol6-compact:tefld_loss",
                "dolly:dolly_baseline",
            ],
            "prompt_pilot_slots": 32,
            "max_generation_attempts_per_slot": 2,
            "maximum_generation_requests": 64,
            "fixed_verifier": "openai/gpt-6.1-sol",
            "full_primary_ready": False,
            "remaining_gate": "Explain/decide optimizer control, then freeze fresh scientific protocol; live prompt quality and Sol 6 compatibility untested.",
            "profile_sha256": {
                key: hashlib.sha256(Path(path).read_bytes()).hexdigest()
                for key, path in profiles.items()
            },
            "warmup_sha256": {
                str(seed): hashlib.sha256(
                    (root / f"shared_initial_seed_{seed}.json").read_bytes()
                ).hexdigest()
                for seed in seeds
            },
        }
        if dataset_feedback is not None:
            plan["dataset_feedback"] = dataset_feedback
            plan["remaining_gate"] = (
                "Freeze common optimizer controls and a fresh scientific protocol before primary training; the previous separate prompt pilot is preserved in its own study."
            )
            plan["main_training_conditions"] += [
                "dataset-refs:" + method
                for method in base["dataset_tefld"]["enabled_methods"]
            ]
        write_json_file(plan_path, plan)
        return plan


def verify_plan(root):
    root = Path(root)
    plan = read_json_file(root / "study_plan.json", {})
    if not plan:
        raise ValueError("Prepare the study first.")
    for key, path in plan["profiles"].items():
        if (
            hashlib.sha256(Path(path).read_bytes()).hexdigest()
            != plan["profile_sha256"][key]
        ):
            raise ValueError("Prepared teacher protocol changed.")
    for seed, expected in plan["warmup_sha256"].items():
        if (
            hashlib.sha256(
                (root / f"shared_initial_seed_{seed}.json").read_bytes()
            ).hexdigest()
            != expected
        ):
            raise ValueError("Prepared common warmup changed.")
    return plan


def import_verifier_audit(root, path):
    root = Path(root)
    path = Path(path).resolve()
    plan = verify_plan(root)
    runner = ExperimentRunner(plan["profiles"]["sol61-compact"], root)
    audit = read_json_file(path, {})
    results = audit.get("results", [])
    if (
        audit.get("status") != "passed"
        or audit.get("model") != runner.verifier.model
        or audit.get("teacher_identity") != runner.verifier.identity
        or audit.get("valid_cases") != 8
        or audit.get("invalid_cases") != 8
        or len(results) != 16
        or audit.get("false_accepts") != 0
        or audit.get("false_rejects", 99) > 1
    ):
        raise ValueError(
            "Existing verifier audit does not meet this fixed-verifier contract."
        )
    if any(
        type(x.get("expected_accept")) is not bool
        or type(x.get("accepted")) is not bool
        for x in results
    ):
        raise ValueError("Invalid verifier fixture results.")
    if sum(x["expected_accept"] for x in results) != 8:
        raise ValueError("Invalid verifier fixture balance.")
    false_accepts = sum(not x["expected_accept"] and x["accepted"] for x in results)
    false_rejects = sum(x["expected_accept"] and not x["accepted"] for x in results)
    if false_accepts != 0 or false_rejects > 1:
        raise ValueError("Verifier fixture results do not meet the gate.")
    target = runner.verification_audit_path
    with section_lock(root):
        write_json_file(
            target,
            {
                **audit,
                "reused_from": str(path),
                "reused_source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "reuse_scope": "Same provider/model/parameters and unchanged verification prompt. Cost belongs to the original study; no new calls.",
            },
        )
    return {
        "status": "matching_verifier_audit_imported",
        "path": str(target),
        "provider_calls": 0,
    }


def preview_prompts(root, tokenizer):
    root = Path(root)
    plan = verify_plan(root)
    if not plan:
        raise ValueError("Prepare the study first.")
    rows = []
    for key, path in plan["profiles"].items():
        if key not in GENERATOR_ARMS:
            continue
        runner = ExperimentRunner(path, root)
        for category in runner.categories:
            prompt = runner.teacher.generation_prompt(
                category,
                runner.data.teacher_seeds(),
                "prompt-study:" + category + ":0:0",
                avoid_questions=["Why does a metal spoon feel colder than wood?"],
            )
            # Student tokens are a local size proxy; provider billing uses its own tokenizer.
            count = len(tokenizer(prompt, add_special_tokens=False)["input_ids"])
            rows.append(
                {
                    "arm": key,
                    "category": category,
                    "utf8_bytes": len(prompt.encode("utf-8")),
                    "local_student_token_proxy": count,
                    "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                }
            )
    pairs = []
    for teacher in MODELS:
        full = sum(
            x["local_student_token_proxy"]
            for x in rows
            if x["arm"] == teacher + "-full"
        )
        compact = sum(
            x["local_student_token_proxy"]
            for x in rows
            if x["arm"] == teacher + "-compact"
        )
        pairs.append(
            {
                "teacher": teacher,
                "full_local_token_proxy": full,
                "compact_local_token_proxy": compact,
                "proxy_reduction_percent": 100 * (full - compact) / full,
            }
        )
    result = {
        "status": "offline_prompt_size_preview",
        "rows": rows,
        "comparisons": pairs,
        "provider_calls": 0,
        "scope": "Input size proxy only; quality, billed-token and cost savings remain untested.",
    }
    write_json_file(root / "prompt_preview.json", result)
    return result


def prompt_pilot(root, tokenizer, *, allow_paid_api=False):
    root = Path(root)
    with section_lock(root):
        plan = verify_plan(root)
        if not plan:
            raise ValueError("Prepare the study first.")
        results_path = root / "prompt_pilot_results.json"
        previous = read_json_file(
            results_path, {"plan_identity": plan["identity"], "slots": []}
        )
        if previous.get("plan_identity") != plan["identity"]:
            raise ValueError("Prompt pilot plan changed.")
        slots = previous["slots"]
        for key, path in plan["profiles"].items():
            if key not in GENERATOR_ARMS:
                continue
            runner = ExperimentRunner(path, root, allow_paid_api=allow_paid_api)
            audit = read_json_file(runner.verification_audit_path, {})
            if (
                audit.get("status") != "passed"
                or audit.get("teacher_identity") != runner.verifier.identity
            ):
                raise PermissionError(
                    "Pass the fixed-verifier regression audit before prompt comparisons."
                )
            directory = runner.output / "prompt_pilot"
            pool = CandidatePool(
                runner.data, tokenizer, runner.teacher, directory / "quarantine.json"
            )
            seen = set()
            pool.remember(
                [
                    Training_Sample.model_validate(x["sample"])
                    for x in read_json_file(pool.admissions_path, {}).values()
                ]
            )
            for category in runner.categories:
                existing = [
                    x for x in slots if x["arm"] == key and x["category"] == category
                ]
                if existing:
                    if existing[0].get("candidate_id"):
                        seen.add(existing[0]["candidate_id"])
                    continue
                with runner.teacher.account_to(
                    arm=key,
                    phase="prompt_pilot",
                    method="fixed_category",
                    round=category,
                ):
                    try:
                        sample = pool.create(
                            category,
                            "prompt-study:" + category + ":0",
                            seen,
                            max_attempts=2,
                        )
                        result = {
                            "arm": key,
                            "category": category,
                            "accepted": True,
                            "candidate_id": candidate_id(sample),
                            "sample": sample.model_dump(mode="json"),
                        }
                    except RuntimeError as exc:
                        if "Valid candidate quota could not be filled" not in str(exc):
                            raise
                        result = {
                            "arm": key,
                            "category": category,
                            "accepted": False,
                            "reason": str(exc),
                        }
                slots.append(result)
                write_json_file(
                    results_path, {"plan_identity": plan["identity"], "slots": slots}
                )
        report = {
            "status": "completed_prompt_pilot",
            "slots": len(slots),
            "accepted_by_arm": {
                arm: sum(x["accepted"] for x in slots if x["arm"] == arm)
                for arm in plan["profiles"]
                if arm in GENERATOR_ARMS
            },
            "quality_conclusion": "Acceptance is probabilistic review, not downstream learning benefit; blinded accepted/rejected review and training comparisons remain necessary.",
            "cost": summarize_teacher_costs(root),
        }
        write_json_file(root / "prompt_pilot_report.json", report)
        return report
