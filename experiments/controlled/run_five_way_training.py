"""Prepare/run five fresh 100-unique-example pilots with the original 64-row evaluation.

Importing this module does not load models or make provider calls.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parent
ARMS = {
    "dolly_baseline": ("dolly_baseline", None),
    "tefld_dataset_loss": ("tefld_dataset_loss", None),
    "sol61_validation": ("tefld_validation", "openai/gpt-6.1-sol"),
    "sol61_loss": ("tefld_loss", "openai/gpt-6.1-sol"),
    "luna_validation": ("tefld_validation", "gpt-6-luna"),
}
TARGET = 100
# A safety ceiling, not the training stopping rule. At most two replay rounds
# per unique example, three epochs per round, <=511 shifted targets/example.
SAFETY_TOKENS = 500000


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def profile(base, arm):
    from TEFLD.src.dataset_curriculum import dataset_protocol

    method, teacher = ARMS[arm]
    result = dataset_protocol(base, "loss")
    controls = result["tefld_controls"]
    verifier = copy.deepcopy(controls["provider"])
    controls["verifier"] = {
        "model_id": "openai/gpt-6.1-sol",
        "provider": verifier,
    }
    controls["generation_prompt_variant"] = "full"
    if teacher == "gpt-6-luna":
        controls.update(
            teacher_model_id=teacher,
            teacher_family=teacher,
            teacher_temperature=1.0,
            teacher_reasoning_effort="low",
        )
        controls["provider"] = {
            **copy.deepcopy(verifier),
            "model_id": teacher,
            "input_per_million": 37000,
            "output_per_million": 184000,
            "pricing_source": "https://hibanacloud.ir/docs/models",
            "pricing_checked": "2026-10-19",
            "request_options": {"temperature": 1.0, "reasoning_effort": "low"},
        }
        controls["teacher_pricing"] = {
            "currency": "toman",
            "input_per_million": 37000,
            "output_per_million": 184000,
            "checked": "2026-10-19",
            "source": "https://hibanacloud.ir/docs/models",
        }
    controls["teacher_request_ceiling"] = 400
    controls["hibana_notes"] = {
        "pricing": "Retained October 19 provider profiles; review before a later run.",
        "warmup": "Ten shared original Dolly references, counted within each arm's 100.",
    }
    result["teacher_comparison"] = {
        "arm": arm,
        "generator_factor": teacher,
        "shared_verifier": "openai/gpt-6.1-sol",
        "shared_budget_required": True,
        "primary_ready": False,
        "purpose": "Fresh development comparison matched by unique training examples.",
    }
    result["example_count_control"] = {
        "unique_training_examples": TARGET,
        "replay_counts_as_new": False,
        "epochs_per_round": 3,
        "checkpoint": "final_100_unique_examples_only",
        "evaluation": "original_checkpoint_validation_8_per_category",
        "tokens_and_updates_matched": False,
    }
    result["common_training"]["epochs_per_round"] = 3
    result["status"] = "five_way_100_training_examples_development"
    result["research_question"] = (
        "Compare five curricula after 100 unique training examples each; "
        "report answer-token exposures, replay, optimizer updates and provider cost."
    )
    return result


def require_fixed(path, value):
    from TEFLD.src.helper import write_json_file

    if path.exists() and read(path) != value:
        raise ValueError(f"Prepared identity changed; use a fresh --output: {path}")
    if not path.exists():
        write_json_file(path, value)


def require_study_plan(path, plan):
    """Adopt only the known launcher-only spend-limit repair, preserving its audit."""
    from TEFLD.src.helper import write_json_file

    prior = read(path) if path.exists() else None
    original_launcher = (
        "60efd378b42359ecfb2822cf3bb0e5bb770cc01eab439be59219f78b95cc7689"
    )
    if prior and prior != plan and prior.get("launcher_sha256") == original_launcher:
        old_inputs = {k: v for k, v in prior.items() if k != "launcher_sha256"}
        new_inputs = {k: v for k, v in plan.items() if k != "launcher_sha256"}
        if old_inputs == new_inputs:
            require_fixed(
                path.with_name("study_plan_before_spend_limit_fix.json"), prior
            )
            require_fixed(
                path.with_name("launcher_spend_limit_fix.json"),
                {
                    "previous_launcher_sha256": original_launcher,
                    "new_launcher_sha256": plan["launcher_sha256"],
                    "reason": "Bind operational spend limits to each client's existing teacher identity; validate before loading models.",
                    "core_training_code_changed": False,
                    "protocols_changed": False,
                    "existing_checkpoints_reused": True,
                },
            )
            write_json_file(path, plan)
            return
    require_fixed(path, plan)


def configure_client_limits(runner, budget=None):
    """Operational ceilings retain provider, audit and request-cache identities."""
    from TEFLD.src.helper import write_json_file

    for client, ceiling in ((runner.teacher, 400), (runner.verifier, 1200)):
        client.max_requests = ceiling
        if budget is not None:
            write_json_file(
                client.directory / "spend_limit.json",
                {
                    "teacher_identity": client.identity,
                    "currency": client.provider["currency"],
                    "max_spend": budget,
                },
            )
        # Fail locally before model loading rather than disguising a config
        # problem as four rejected generation candidates.
        client.spend_ceiling()


def prepare(root):
    """Reuse only frozen input artifacts; never reuse trained adapters."""
    from TEFLD.src.candidate_pool import candidate_id
    from TEFLD.src.dataschema import Training_Sample
    from TEFLD.src.experiment_runner import ExperimentRunner
    from TEFLD.src.helper import stable_json_hash
    from TEFLD.src.runtime import file_fingerprint
    from TEFLD.src.validation_data import SplitData

    base_path = ROOT / "experiment_design_2026-10-18/experiment_protocol_hibana.json"
    data = SplitData(base_path)
    warmup_source = (
        ROOT
        / "experiment_runs_two_teachers_dataset_extension/shared_initial_seed_42.json"
    )
    prior_plan = read(warmup_source.parent / "study_plan.json")
    if prior_plan["warmup_sha256"]["42"] != file_fingerprint(warmup_source):
        raise ValueError("Prepared training-only warmup changed.")
    warmup = read(warmup_source)
    if (
        len(warmup) != 10
        or len({candidate_id(Training_Sample.model_validate(x)) for x in warmup}) != 10
    ):
        raise ValueError("Ten distinct shared warmup examples are required.")
    for item in warmup:
        metadata = item["verification"]
        sample = data.sample(metadata["row_id"], "training_pool")
        if metadata["role"] != "training_pool" or any(
            item[key] != getattr(sample, key)
            for key in ("sample", "instruct", "gold_summary", "tag")
        ):
            raise ValueError("Warmup must retain original training references.")
    evaluation_ids = [i for i, _ in data.samples("checkpoint_validation", 8)]
    historical = (
        "experiment_runs_dataset_tefld/pilot/dolly_baseline_seed_42",
        "experiment_runs_dataset_tefld/pilot/tefld_dataset_loss_seed_42",
        "experiment_runs_hibana/pilot/tefld_validation_seed_42",
        "experiment_runs_hibana/pilot/tefld_loss_seed_42",
    )
    for source in historical:
        state = read(ROOT / source / "state.json")
        scores = read(Path(state["final_checkpoint"]["path"]) / "validation.json")
        if [row["row_id"] for row in scores["rows"]] != evaluation_ids or len(
            evaluation_ids
        ) != 64:
            raise ValueError("The original 64 evaluation rows no longer match.")
    profiles = {arm: profile(data.protocol, arm) for arm in ARMS}
    audit_source = ROOT / "experiment_runs_hibana/verification_audit.json"
    audit = read(audit_source)
    cases = audit.get("results", [])
    if (
        audit.get("status") != "passed"
        or len(cases) != 16
        or any(
            type(c.get("expected_accept")) is not bool
            or type(c.get("accepted")) is not bool
            for c in cases
        )
        or sum(c["expected_accept"] for c in cases) != 8
        or any(c["accepted"] != c["expected_accept"] for c in cases)
    ):
        raise ValueError("The fixed verifier needs a passing audit.")
    plan = {
        "version": "five-way-100-training-v1",
        "seed": 42,
        "unique_training_examples_per_arm": TARGET,
        "evaluation_row_ids": evaluation_ids,
        "evaluation_examples": 64,
        "evaluation_per_category": 8,
        "evaluation_metric": "reference_answer_macro_category_NLL",
        "curriculum_validation_per_category": 8,
        "warmup_sha256": {"42": file_fingerprint(warmup_source)},
        "warmup_source": str(warmup_source),
        "warmup_counts_within_100": True,
        "profiles_sha256": {arm: stable_json_hash(p) for arm, p in profiles.items()},
        "dataset_sha256": data.protocol["dataset"]["sha256"],
        "split_sha256": data.protocol["dataset"]["manifest_sha256"],
        "core_code_sha256": stable_json_hash(
            {
                p.name: file_fingerprint(p)
                for p in sorted((ROOT / "TEFLD/src").glob("*.py"))
            }
        ),
        "launcher_sha256": file_fingerprint(Path(__file__)),
        "audit_sha256": file_fingerprint(audit_source),
        "safety_token_ceiling": SAFETY_TOKENS,
        "maximum_requests": {"generation_per_arm": 400, "shared_verification": 1200},
        "token_matched": False,
        "optimizer_recipe": "original_four_examples_reset_each_round",
        "final_test_scored": False,
    }
    require_study_plan(root / "study_plan.json", plan)
    warmup_path = root / "shared_initial_seed_42.json"
    if warmup_path.exists() and warmup_path.read_bytes() != warmup_source.read_bytes():
        raise ValueError("Shared warmup changed.")
    warmup_path.write_bytes(warmup_source.read_bytes())
    for arm, settings in profiles.items():
        path = root / "protocols" / (arm + ".json")
        require_fixed(path, settings)
        runner = ExperimentRunner(path, root, allow_paid_api=False)
        if (
            audit.get("model") != runner.verifier.model
            or audit.get("teacher_identity") != runner.verifier.identity
        ):
            raise ValueError("Fixed verifier identity does not match its audit.")
    require_fixed(root / "verification_audit.json", audit)
    if not (root / "study_budget.json").exists():
        require_fixed(
            root / "study_budget.json",
            {
                "approved": False,
                "currency": "toman",
                "max_spend": 0,
                "scope": "New total cap across all three synthetic curricula and fixed verification.",
            },
        )
    return plan


def run_path(root, arm):
    return root / "arms" / arm / "pilot" / (ARMS[arm][0] + "_seed_42")


def status(root):
    result = {}
    for arm in ARMS:
        path = run_path(root, arm) / "state.json"
        state = read(path) if path.exists() else {}
        result[arm] = {
            "status": state.get("status", "not_started"),
            "unique_examples": len(state.get("ledger", [])),
            "answer_token_exposures": state.get("tokens", 0),
        }
    return result


def summarize(root, plan):
    from TEFLD.src.experiment_runner import require_adapter
    from TEFLD.src.helper import write_json_file
    from TEFLD.src.teacher_accounting import summarize_teacher_costs

    results = {}
    expected_contract = None
    for arm in ARMS:
        path = run_path(root, arm) / "state.json"
        if not path.exists() or read(path).get("status") != "complete":
            return
        state = read(path)
        if len({entry["id"] for entry in state["ledger"]}) != TARGET:
            raise ValueError(
                "Every reported arm must have exactly 100 unique trained examples."
            )
        checkpoint = Path(state["final_checkpoint"]["path"])
        require_adapter(checkpoint)
        scores = read(checkpoint / "validation.json")
        if [r["row_id"] for r in scores["rows"]] != plan["evaluation_row_ids"]:
            raise ValueError("Evaluation rows changed.")
        contract = [
            (r["row_id"], r["category"], r["answer_tokens"]) for r in scores["rows"]
        ]
        if expected_contract is not None and contract != expected_contract:
            raise ValueError("Evaluation answer supervision differs between arms.")
        expected_contract = contract
        results[arm] = {
            "unique_training_examples": TARGET,
            "evaluation_examples": 64,
            "macro_nll": scores["macro_nll"],
            "token_weighted_nll": scores["token_weighted_nll"],
            "answer_token_exposures": state["tokens"],
            "optimizer_updates": len(
                {(r["round"], r["tokens_after_group"]) for r in state["losses"]}
            ),
            "training_presentations": len(state["losses"]),
            "replay_batch_slots": sum(
                p["source"] == "recycled" for p in state["presentations"]
            ),
            "untrained_admissions": len(state.get("untrained_final_admissions", [])),
            "checkpoint": str(checkpoint),
            "base_identity": state["config"]["model_sha256"],
        }
    if len({r["base_identity"] for r in results.values()}) != 1:
        raise ValueError("Base model identities differ.")
    write_json_file(
        root / "comparison.json",
        {
            "status": "complete",
            "results": results,
            "ranking": sorted(results, key=lambda arm: results[arm]["macro_nll"]),
            "teacher_costs": summarize_teacher_costs(root),
            "token_matched": False,
            "final_test_scored": False,
        },
    )
    lines = [
        "# Five methods, 100 unique training examples each",
        "",
        "Fresh seed-42 adapters; the original 64 development evaluation examples.",
        "",
        "| Method | Unique training examples | Answer-token exposures | Updates | Macro NLL |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for arm, row in results.items():
        lines.append(
            f"| {arm} | 100 | {row['answer_token_exposures']} | {row['optimizer_updates']} | {row['macro_nll']:.6f} |"
        )
    lines += [
        "",
        "Three epochs per admitted batch; adaptive replay retained. Token exposure and optimizer updates differ. The final 100-example checkpoint is used for every arm. Development NLL is not accuracy; these are single-seed exploratory results. Final holdout unscored.",
    ]
    (root / "COMPARISON.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "run", "status"))
    parser.add_argument(
        "--output", type=Path, default=ROOT / "experiment_runs_five_way_100_training"
    )
    parser.add_argument("--methods", nargs="+", choices=tuple(ARMS), default=list(ARMS))
    parser.add_argument("--allow-paid-api", action="store_true")
    parser.add_argument("--teacher-budget-toman", type=float)
    args = parser.parse_args()
    if len(args.methods) != len(set(args.methods)):
        parser.error("Choose each method once.")
    if args.teacher_budget_toman is not None and (
        not math.isfinite(args.teacher_budget_toman) or args.teacher_budget_toman <= 0
    ):
        parser.error("Use a finite positive total study budget.")
    if args.command != "run" and (
        args.allow_paid_api or args.teacher_budget_toman is not None
    ):
        parser.error("Paid options are only used by the run command.")
    if args.teacher_budget_toman is not None and not args.allow_paid_api:
        parser.error("A teacher budget also requires --allow-paid-api.")
    root = args.output.resolve()
    if args.command == "status":
        print(json.dumps(status(root), indent=2))
        return
    from TEFLD.src.experiment_runner import ExperimentRunner
    from TEFLD.src.helper import write_json_file
    from TEFLD.src.teacher_accounting import study_spend
    from TEFLD.src.transactions import section_lock

    with section_lock(root):
        plan = prepare(root)
        if args.command == "prepare":
            print(
                json.dumps(
                    {
                        "status": "prepared",
                        "output": str(root),
                        "training_examples_per_arm": 100,
                        "evaluation_examples": 64,
                        "provider_calls": 0,
                    },
                    indent=2,
                )
            )
            return
        current = status(root)
        needs_paid = any(
            ARMS[arm][1] is not None and current[arm]["status"] != "complete"
            for arm in args.methods
        )
        if needs_paid:
            if not args.allow_paid_api or args.teacher_budget_toman is None:
                parser.error(
                    "Synthetic training requires --allow-paid-api and an explicit --teacher-budget-toman total cap."
                )
            if args.teacher_budget_toman < study_spend(root).get("toman", 0):
                parser.error("The total cap is below recorded study spending.")
            write_json_file(
                root / "study_budget.json",
                {
                    "approved": True,
                    "currency": "toman",
                    "max_spend": args.teacher_budget_toman,
                    "scope": "Explicit user-run total cap across this entire five-arm study; not an additional allowance on resume.",
                },
            )
        last_request = [0.0]
        for arm in args.methods:
            if current[arm]["status"] == "complete":
                continue
            runner = ExperimentRunner(
                root / "protocols" / (arm + ".json"),
                root,
                allow_paid_api=args.allow_paid_api,
            )
            configure_client_limits(
                runner, args.teacher_budget_toman if needs_paid else None
            )
            for client in (runner.teacher, runner.verifier):
                original_ask = client.ask

                def paced_ask(prompt, kind, _ask=original_ask):
                    delay = 13.0 - (time.monotonic() - last_request[0])
                    if delay > 0:
                        time.sleep(delay)
                    last_request[0] = time.monotonic()
                    return _ask(prompt, kind)

                client.ask = paced_ask
            runner.run(
                ARMS[arm][0], 42, phase="pilot", budget=SAFETY_TOKENS, per_category=8
            )
        summarize(root, plan)
        print(json.dumps(status(root), indent=2))


if __name__ == "__main__":
    main()
