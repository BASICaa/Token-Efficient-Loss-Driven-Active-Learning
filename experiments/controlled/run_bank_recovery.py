"""Fork a saved-bank pilot; replenish required categories with verified generation.

The original study and its bank stay immutable. No provider calls or training
occur during prepare. Run requires explicit paid enablement and a shared cap.
"""

import argparse
from collections import Counter
import copy
import json
import math
from pathlib import Path
import random
import shutil
import time

from TEFLD.src.candidate_pool import candidate_id
from TEFLD.src.dataschema import Training_Sample
from TEFLD.src.experiment_runner import ExperimentRunner, require_adapter
from TEFLD.src.helper import read_json_file, stable_json_hash, write_json_file
from TEFLD.src.runtime import file_fingerprint
from TEFLD.src.teacher_accounting import study_spend
from TEFLD.src.transactions import section_lock

POLICY = "required-category-replenishment-v1"
CHECKS = ("correct", "supported", "format_valid", "unambiguous")


def verified_bank(path):
    bank = read_json_file(Path(path), [])
    parsed = [Training_Sample.model_validate(x) for x in bank]
    if not parsed or len({candidate_id(x) for x in parsed}) != len(parsed):
        raise ValueError("Bank must contain unique verified candidates.")
    for sample in parsed:
        if sample.verification.get("status") != "accepted_probabilistic_review" or any(
            sample.verification.get("result", {}).get(key) is not True for key in CHECKS
        ):
            raise ValueError("Bank lacks passing verification.")
    return bank


def required_categories(state, categories, rng, replay):
    """Mirror the frozen engine's quota draw without consuming its RNG state."""
    draw = random.Random()
    draw.setstate(rng.getstate())
    round_id = state["round"]
    eligible = [
        x
        for x in state["ledger"]
        if x.get("loss_round") == round_id - 1
        and x["loss"] >= 0.35
        and x["replay_uses"] < 2
        and (x["last_replay_round"] is None or round_id - x["last_replay_round"] > 1)
    ]
    recycled = min(3, len(eligible)) if replay else 0
    weakness = state["weakness"]
    fallback = sum(weakness.values()) / len(weakness) if weakness else 1.0
    weights = [max(0.00001, weakness.get(c, fallback)) for c in categories]
    focus = Counter(draw.choices(categories, weights=weights, k=6 - recycled))
    exploration = [draw.choice(categories) for _ in range(4)]
    return focus + Counter(exploration)


class RecoveryRunner(ExperimentRunner):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.base_code_hash = self.code_hash
        self.code_hash = stable_json_hash(
            {
                "base_code_sha256": self.base_code_hash,
                "recovery_sha256": file_fingerprint(Path(__file__)),
                "policy": POLICY,
            }
        )

    def _batch(
        self,
        state,
        method,
        seed,
        round_id,
        rng,
        pool,
        model,
        tokenizer,
        replay,
        bank,
        predictor,
        baseline_order,
    ):
        if not bank or method not in {"tefld_loss", "tefld_validation"}:
            raise ValueError("Recovery supports only the two saved-bank TEFLD methods.")
        # Durable admissions survive failure after a paid call, before training.
        augmented = list(bank)
        known = {candidate_id(Training_Sample.model_validate(x)) for x in augmented}
        for item in read_json_file(pool.admissions_path, {}).values():
            sample = Training_Sample.model_validate(item["sample"])
            if item.get("teacher_identity") != self.teacher.identity:
                raise ValueError("Recovery admission teacher changed.")
            if sample.verification.get(
                "status"
            ) != "accepted_probabilistic_review" or any(
                sample.verification.get("result", {}).get(key) is not True
                for key in CHECKS
            ):
                raise ValueError("Recovery admission lacks verification.")
            identity = candidate_id(sample)
            if identity not in known:
                augmented.append(sample.model_dump(mode="json"))
                known.add(identity)
        available = Counter(
            x["tag"]
            for x in augmented
            if candidate_id(Training_Sample.model_validate(x)) not in state["seen"]
        )
        required = required_categories(state, self.categories, rng, replay)
        pool.remember([Training_Sample.model_validate(x) for x in augmented])
        seen = set(state["seen"]) | known
        for category, count in sorted(required.items()):
            for slot in range(max(0, count - available[category])):
                # Include durable admission count to avoid nonce collisions after
                # a partially filled recovery round is restarted.
                nonce = f"recovery:{method}:{seed}:{round_id}:{category}:{available[category] + slot}"
                sample = pool.create(category, nonce, seen, state["weakness"])
                augmented.append(sample.model_dump(mode="json"))
        # Selection, replay, RNG, optimizer and round commit remain the original
        # engine's responsibility. No category substitution or quota renormalization.
        return super()._batch(
            state,
            method,
            seed,
            round_id,
            rng,
            pool,
            model,
            tokenizer,
            replay,
            augmented,
            predictor,
            baseline_order,
        )


def prepare(source, output, bank_path, budget_root, generation_protocol=None):
    source, output, budget_root = map(
        lambda p: Path(p).resolve(), (source, output, budget_root)
    )
    bank_path = Path(bank_path).resolve()
    if source == output or source in output.parents or output in source.parents:
        raise ValueError("Recovery output must be separate from its source.")
    if (
        budget_root not in output.parents
        or budget_root == source
        or budget_root in source.parents
    ):
        raise ValueError(
            "Use a separate shared recovery budget directory containing its outputs."
        )
    if output.exists():
        raise ValueError(
            "Recovery output exists; run it directly instead of preparing again."
        )
    bank = verified_bank(bank_path)
    protocol = read_json_file(source / "protocol.json", {})
    plan = read_json_file(source / "study_plan.json", {})
    settings = protocol["controlled_entry"]
    if settings["mode"] not in {"api-loss", "api-validation"}:
        raise ValueError("Only API-derived saved-bank studies can be recovered.")
    original = ExperimentRunner(source / "protocol.json", source)
    if (
        plan.get("core_code_sha256") != original.code_hash
        or plan.get("protocol_sha256") != original.protocol_hash
    ):
        raise ValueError("Source code/protocol changed; cannot adopt this recovery.")
    source_dir = original.run_directory("pilot", settings["method"], settings["seed"])
    state_path = source_dir / "state.json"
    state = read_json_file(state_path, {})
    if state and (state.get("status") == "complete" or state.get("batch")):
        raise ValueError(
            "Recover only unfinished pilots at a committed round boundary."
        )
    if state and (
        state["config"]["bank_sha256"] != stable_json_hash({"bank": bank})
        or state["config"]["code_sha256"] != original.code_hash
        or state["config"]["protocol_sha256"] != original.protocol_hash
    ):
        raise ValueError("Source run does not match its saved bank and protocol.")
    if state.get("last_adapter"):
        require_adapter(state["last_adapter"])
    if generation_protocol:
        controls = read_json_file(Path(generation_protocol), {})["tefld_controls"]
        # Generation routing/pricing only; retain the original fixed verifier.
        for key in (
            "teacher_model_id",
            "provider",
            "teacher_temperature",
            "generation_prompt_variant",
        ):
            if key in controls:
                protocol["tefld_controls"][key] = copy.deepcopy(controls[key])
    write_json_file(output / "protocol.json", protocol)
    runner = RecoveryRunner(output / "protocol.json", output)
    manifest = read_json_file(bank_path.with_suffix(".manifest.json"), {})
    if (
        manifest.get("bank_sha256") != stable_json_hash({"bank": bank})
        or manifest.get("teacher_identity") != runner.teacher.identity
    ):
        raise ValueError(
            "Generation profile must match the exported bank's teacher identity."
        )
    audit = read_json_file(source / "verification_audit.json", {})
    if (
        audit.get("status") != "passed"
        or audit.get("teacher_identity") != runner.verifier.identity
    ):
        raise ValueError("An identity-matched passing verifier audit is required.")
    write_json_file(output / "verification_audit.json", audit)
    if (original.output / "model_fingerprint.json").exists():
        shutil.copy2(
            original.output / "model_fingerprint.json",
            runner.output / "model_fingerprint.json",
        )
    for path in source.glob("shared_initial_seed_*.json"):
        shutil.copy2(path, output / path.name)
    plan.update(core_code_sha256=runner.code_hash, protocol_sha256=runner.protocol_hash)
    write_json_file(output / "study_plan.json", plan)
    if state:
        state = copy.deepcopy(state)
        state["config"].update(
            code_sha256=runner.code_hash, protocol_sha256=runner.protocol_hash
        )
        state["recovery"] = {
            "policy": POLICY,
            "source": str(state_path),
            "source_sha256": file_fingerprint(state_path),
            "inherited_unique_examples": len(state["ledger"]),
            "inherited_tokens": state["tokens"],
        }
        write_json_file(
            runner.run_directory("pilot", settings["method"], settings["seed"])
            / "state.json",
            state,
        )
    recovery = {
        "policy": POLICY,
        "source": str(source),
        "source_state_sha256": file_fingerprint(state_path) if state else None,
        "bank": str(bank_path),
        "bank_sha256": stable_json_hash({"bank": bank}),
        "budget_root": str(budget_root),
        "code_sha256": runner.code_hash,
        "protocol_sha256": runner.protocol_hash,
        "scope": "Amended development pilot: saved bank plus verified category replenishment; not fixed-bank-only.",
        "inherited_unique_examples": len(state.get("ledger", [])),
        "provider_calls_during_prepare": 0,
    }
    write_json_file(output / "recovery_plan.json", recovery)
    return recovery


def run(output, *, allow_paid, budget):
    output = Path(output).resolve()
    if not allow_paid or budget is None or not math.isfinite(budget) or budget <= 0:
        raise ValueError(
            "Recovery requires --allow-paid-api and a positive total --teacher-budget-toman."
        )
    recovery = read_json_file(output / "recovery_plan.json", {})
    budget_root = Path(recovery["budget_root"])
    # A shared lock covers reservations and training for every recovery arm.
    with section_lock(budget_root), section_lock(output):
        runner = RecoveryRunner(output / "protocol.json", output, allow_paid_api=True)
        if (
            runner.code_hash != recovery["code_sha256"]
            or runner.protocol_hash != recovery["protocol_sha256"]
        ):
            raise ValueError("Recovery code or protocol changed.")
        bank = verified_bank(recovery["bank"])
        if stable_json_hash({"bank": bank}) != recovery["bank_sha256"]:
            raise ValueError("Original saved bank changed.")
        if budget < study_spend(budget_root).get("toman", 0):
            raise ValueError("Cap is below already recorded recovery spending.")
        write_json_file(
            budget_root / "study_budget.json",
            {"approved": True, "currency": "toman", "max_spend": budget},
        )
        audit = read_json_file(runner.verification_audit_path, {})
        if (
            audit.get("status") != "passed"
            or audit.get("teacher_identity") != runner.verifier.identity
        ):
            raise ValueError("Verifier audit changed.")
        for client in (runner.teacher, runner.verifier):
            client.study_budget_root = budget_root
        from run_five_way_training import configure_client_limits

        configure_client_limits(runner, budget)
        last = [0.0]
        for client in (runner.teacher, runner.verifier):
            ask = client.ask

            def paced(prompt, kind, _ask=ask):
                delay = 13 - (time.monotonic() - last[0])
                if delay > 0:
                    time.sleep(delay)
                last[0] = time.monotonic()
                return _ask(prompt, kind)

            client.ask = paced
        settings = runner.protocol["controlled_entry"]
        result = runner.run(
            settings["method"],
            settings["seed"],
            phase="pilot",
            budget=settings["safety_tokens"],
            per_category=settings["per_category"],
            candidate_bank=bank,
        )
        directory = runner.run_directory("pilot", settings["method"], settings["seed"])
        admissions = read_json_file(directory / "candidate_admissions.json", {})
        summary = {
            "scope": recovery["scope"],
            "status": result["status"],
            "unique_examples": len(result["ledger"]),
            "new_verified_candidates": len(admissions),
            "inherited_unique_examples": recovery["inherited_unique_examples"],
            "new_verified_categories": dict(
                Counter(x["sample"]["tag"] for x in admissions.values())
            ),
            "new_trained_candidates": len(
                {
                    candidate_id(Training_Sample.model_validate(x["sample"]))
                    for x in admissions.values()
                }
                & {x["id"] for x in result["ledger"]}
            ),
            "this_arm_recovery_spend_toman": study_spend(output).get("toman", 0),
            "shared_recovery_spend_toman": study_spend(budget_root).get("toman", 0),
            "final_checkpoint": result["final_checkpoint"],
            "final_test_scored": False,
        }
        write_json_file(output / "recovery_result.json", summary)
        return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "run"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--candidate-bank", type=Path)
    parser.add_argument("--budget-root", type=Path)
    parser.add_argument("--generation-protocol", type=Path)
    parser.add_argument("--allow-paid-api", action="store_true")
    parser.add_argument("--teacher-budget-toman", type=float)
    args = parser.parse_args()
    if args.command == "prepare":
        if not args.source or not args.candidate_bank or not args.budget_root:
            parser.error(
                "prepare requires --source, --candidate-bank and --budget-root"
            )
        if args.allow_paid_api or args.teacher_budget_toman is not None:
            parser.error("prepare does not accept paid flags")
        with section_lock(args.source):
            result = prepare(
                args.source,
                args.output,
                args.candidate_bank,
                args.budget_root,
                args.generation_protocol,
            )
    else:
        result = run(
            args.output,
            allow_paid=args.allow_paid_api,
            budget=args.teacher_budget_toman,
        )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
