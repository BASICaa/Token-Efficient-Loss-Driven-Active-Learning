"""Offline common-bank studies and grouped counterfactual label collection."""

from __future__ import annotations

from collections import defaultdict
import hashlib
from pathlib import Path
import random
import time

from .benefit_features import candidate_features, gradient_summary
from .benefit_predictor import BenefitPredictor, regression_metrics
from .candidate_pool import LeakageGuard, candidate_id, normalize
from .counterfactual_probe import virtual_update
from .dataschema import Training_Sample
from .encoding import encode_sample
from .helper import read_json_file, stable_json_hash, write_json_file
from .validation import validate_sample
from .validation_data import score_samples


def group_id(sample):
    context = normalize(sample.sample)
    return (
        stable_json_hash({"context": context})
        if len(context) >= 80
        else candidate_id(sample)
    )


def prepare_bank(
    data, tokenizer, path, *, per_category=64, seed=20261018, exclude_groups=None
):
    guard = LeakageGuard(data)
    excluded = set(exclude_groups or [])
    selected = []
    rng = random.Random(seed)
    for category in data.protocol["dataset"]["categories"]:
        ids = sorted(
            i for i in data.ids["training_pool"] if data.rows[i]["category"] == category
        )
        rng.shuffle(ids)
        count = 0
        for row_id in ids:
            sample = data.sample(row_id, "training_pool")
            if group_id(sample) in excluded:
                continue
            try:
                validate_sample(sample).require()
                guard.require(sample)
                encode_sample(sample, tokenizer)
            except ValueError:
                continue
            selected.append(sample.model_dump(mode="json"))
            excluded.add(group_id(sample))
            count += 1
            if count == per_category:
                break
        if count != per_category:
            raise ValueError(f"Cannot fill verified bank for {category}.")
    write_json_file(Path(path), selected)
    write_json_file(
        Path(path).with_suffix(".manifest.json"),
        {
            "status": "offline_development_bank",
            "source_role": "training_pool",
            "samples": len(selected),
            "per_category": per_category,
            "seed": seed,
            "bank_sha256": stable_json_hash({"bank": selected}),
            "group_ids": sorted(
                {group_id(Training_Sample.model_validate(x)) for x in selected}
            ),
            "scope": "Dolly reference targets; not newly generated synthetic data.",
        },
    )
    return selected


def view_samples(data, name):
    path = data.protocol_path.parent / "benefit_curriculum_views.json"
    views = read_json_file(path, {})
    expected = data.protocol["benefit_prediction"]["probe"].get(
        "view_membership_sha256"
    )
    if expected and hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        raise ValueError("Curriculum subview changed.")
    return [(i, data.sample(i, "curriculum_validation")) for i in views["views"][name]]


class ProbeCollector:
    """Checkpoint/candidate groups stay intact across fit/calibration/audit roles."""

    def __init__(
        self,
        runner,
        directory,
        role,
        snapshots,
        *,
        per_snapshot=16,
        probe_bank=None,
        with_jev=False,
    ):
        if role not in {"fit", "calibration", "audit"}:
            raise ValueError("Unknown calibration role.")
        self.runner = runner
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.role = role
        self.snapshots = snapshots
        self.per_snapshot = per_snapshot
        self.bank = probe_bank
        self.with_jev = with_jev
        if not 1 <= per_snapshot <= 16:
            raise ValueError("Probe snapshots require 1–16 candidates.")
        if (
            role == "audit"
            and read_json_file(self.directory / "predictor_fit_report.json", {}).get(
                "status"
            )
            != "frozen_for_locked_audit"
        ):
            raise PermissionError(
                "Freeze the predictor before collecting locked audit labels."
            )
        self.path = self.directory / f"{role}_probes.json"
        self.rows = read_json_file(self.path, [])
        self.group_map_path = self.directory / "candidate_group_roles.json"
        groups = read_json_file(self.group_map_path, {})
        for previous_role in ("fit", "calibration", "audit"):
            for row in read_json_file(
                self.directory / f"{previous_role}_probes.json", []
            ):
                if (
                    row["candidate_group"] in groups
                    and groups[row["candidate_group"]] != previous_role
                ):
                    raise ValueError("Persisted candidate-group leakage.")
                groups[row["candidate_group"]] = previous_role
        write_json_file(self.group_map_path, groups)

    def __call__(self, model, tokenizer, state):
        if state["round"] < 1:
            return
        for nominal in self.snapshots:
            threshold = 0 if nominal == "post_warmup" else int(nominal)
            if state["tokens"] < threshold:
                continue
            existing = [x for x in self.rows if x["nominal_tokens"] == nominal]
            if len(existing) >= self.per_snapshot:
                continue
            snapshot = stable_json_hash(
                {
                    "adapter": state["last_adapter"],
                    "tokens": state["tokens"],
                    "seed": state["config"]["seed"],
                }
            )
            if any(
                x["snapshot_id"] == snapshot and x["nominal_tokens"] != nominal
                for x in self.rows
            ):
                raise RuntimeError(
                    "Multiple nominal probes map to one actual checkpoint; adjust trajectory checkpoints before collection."
                )
            if existing and existing[0]["snapshot_id"] != snapshot:
                raise RuntimeError(
                    "Resume the saved adapter for an interrupted probe snapshot before advancing training."
                )
            feature_view = view_samples(self.runner.data, "gradient_features")
            outcome_view = view_samples(self.runner.data, "probe_outcomes")
            summary = gradient_summary(
                model,
                tokenizer,
                feature_view,
                dimensions=self.runner.projection_dimensions,
                include_subspace=self.runner.gradient_geometry == "subspace",
            )
            plan_path = self.directory / f"{self.role}_selection_{nominal}.json"
            plan = read_json_file(plan_path, {})
            if plan:
                if plan["snapshot_id"] != snapshot:
                    raise RuntimeError("Probe selection snapshot changed.")
                selected = [Training_Sample.model_validate(x) for x in plan["samples"]]
            else:
                trained = {x["id"] for x in state["ledger"]}
                trained |= {
                    candidate_id(Training_Sample.model_validate(x))
                    for x in state.get("batch", [])
                }
                groups = read_json_file(self.group_map_path, {})
                used = {x["candidate_id"] for x in self.rows}
                used_groups = {x["candidate_group"] for x in self.rows}
                if not self.bank:
                    raise RuntimeError(
                        "Supply a verified training-only probe bank; no implicit paid generation."
                    )
                rng = random.Random(state["config"]["seed"] + int(state["tokens"]))
                by_category = defaultdict(list)
                for raw in self.bank:
                    sample = Training_Sample.model_validate(raw)
                    identity = candidate_id(sample)
                    group = group_id(sample)
                    if (
                        identity in trained
                        or identity in used
                        or group in used_groups
                        or groups.get(group, self.role) != self.role
                    ):
                        continue
                    by_category[sample.tag].append(sample)
                selected = []
                selected_groups = set()
                for category in self.runner.categories:
                    options = by_category[category]
                    rng.shuffle(options)
                    chosen = []
                    for sample in options:
                        group = group_id(sample)
                        if group in selected_groups:
                            continue
                        chosen.append(sample)
                        selected_groups.add(group)
                        if len(chosen) == 2:
                            break
                    if len(chosen) < 2:
                        raise RuntimeError(
                            "Insufficient disjoint untrained probe candidates for "
                            + category
                        )
                    selected.extend(chosen)
                selected = selected[: self.per_snapshot]
                write_json_file(
                    plan_path,
                    {
                        "snapshot_id": snapshot,
                        "samples": [x.model_dump(mode="json") for x in selected],
                    },
                )
            groups = read_json_file(self.group_map_path, {})
            baseline = score_samples(model, tokenizer, outcome_view)["macro_nll"]
            repeat = score_samples(model, tokenizer, outcome_view)["macro_nll"]
            for sample in selected[len(existing) :]:
                attempts_path = self.directory / "probe_attempts.json"
                attempts = read_json_file(attempts_path, [])
                total = sum(
                    len(read_json_file(self.directory / f"{role}_probes.json", []))
                    for role in ("fit", "calibration", "audit")
                )
                if len(attempts) >= 512 or total >= 256:
                    raise RuntimeError("Probe study attempt/success budget exhausted.")
                attempts.append(
                    {
                        "role": self.role,
                        "snapshot_id": snapshot,
                        "candidate_id": candidate_id(sample),
                    }
                )
                write_json_file(attempts_path, attempts)
                started = time.perf_counter()
                features = candidate_features(
                    model,
                    tokenizer,
                    sample,
                    summary,
                    progress=state["tokens"] / state["config"]["budget"],
                    seed=state["config"]["seed"],
                )
                if self.with_jev:
                    features.update(
                        self.runner.jev.features(
                            sample, state["weakness"], candidate_id(sample)
                        )
                    )
                encoded = encode_sample(sample, tokenizer)
                label = virtual_update(
                    model,
                    encoded,
                    measure=lambda: score_samples(model, tokenizer, outcome_view)[
                        "macro_nll"
                    ],
                    seed=state["config"]["seed"],
                )
                identity = candidate_id(sample)
                group = group_id(sample)
                groups[group] = self.role
                self.rows.append(
                    {
                        "role": self.role,
                        "trajectory_seed": state["config"]["seed"],
                        "jev_backend": (
                            self.runner.jev.backend if self.with_jev else None
                        ),
                        "snapshot_id": snapshot,
                        "snapshot_adapter": state["last_adapter"],
                        "nominal_tokens": nominal,
                        "actual_tokens": state["tokens"],
                        "candidate_id": identity,
                        "candidate_group": group,
                        "features": features,
                        "gain": label["gain"],
                        "state_sha256": label["state_sha256"],
                        "no_update_noise": abs(baseline - repeat),
                        "seconds": time.perf_counter() - started,
                        "label_virtual_optimizer_steps": 1,
                        "feature_virtual_optimizer_steps": 1,
                        "probe_answer_tokens": int(features["target_tokens"]),
                        "source_status": (
                            "dataset_reference_bank"
                            if sample.verification.get("status") == "dataset_reference"
                            else "verified_synthetic_bank"
                        ),
                        "protocol_sha256": self.runner.protocol_hash,
                        "code_sha256": self.runner.code_hash,
                    }
                )
                write_json_file(self.path, self.rows)
                write_json_file(self.group_map_path, groups)


def fit_grouped_predictor(directory, feature_mode="gradient"):
    directory = Path(directory)
    if read_json_file(directory / "audit_probes.json", []):
        raise PermissionError(
            "Locked audit has been opened; use a fresh calibration study to refit."
        )
    fit = read_json_file(directory / "fit_probes.json", [])
    calibration = read_json_file(directory / "calibration_probes.json", [])
    if not fit or not calibration:
        raise ValueError("Fit and calibration observations are required.")
    for key in ("code_sha256", "protocol_sha256", "source_status"):
        if len({r.get(key) for r in fit + calibration}) != 1:
            raise ValueError("Calibration observation identities differ: " + key)
    if feature_mode in {"jev", "hybrid"} and any(
        r.get("jev_backend") != "typesafe" for r in fit + calibration
    ):
        raise ValueError(
            "Only provenance-verified TypeSafe observations qualify for Jev predictor fitting."
        )
    if {r["candidate_group"] for r in fit} & {
        r["candidate_group"] for r in calibration
    }:
        raise ValueError("Candidate-group leakage.")
    if {r["trajectory_seed"] for r in fit} & {
        r["trajectory_seed"] for r in calibration
    }:
        raise ValueError("Trajectory leakage.")
    snapshots = sorted({r["actual_tokens"] for r in fit})
    if len(snapshots) < 3:
        raise ValueError("At least three chronological fit checkpoints required.")
    cutoff = snapshots[-2]
    early = [r for r in fit if r["actual_tokens"] < cutoff]
    later = [r for r in fit if r["actual_tokens"] >= cutoff]
    if {r["candidate_group"] for r in early} & {r["candidate_group"] for r in later}:
        raise ValueError("Inner chronological validation has shared candidate groups.")
    candidates = []
    for alpha in [0.1, 1.0, 10.0, 100.0]:
        predictor = BenefitPredictor.fit(early, alpha, feature_mode)
        candidates.append((regression_metrics(later, predictor)["rmse"], alpha))
    _, alpha = min(candidates)
    predictor = BenefitPredictor.fit(fit, alpha, feature_mode)
    predictor.calibrate(
        calibration,
        delta_min=max(1e-5, 3 * max(r["no_update_noise"] for r in calibration)),
    )
    predictor.save(directory / "predictor.json")
    report = {
        "status": "frozen_for_locked_audit",
        "alpha": alpha,
        "fit_rows": len(fit),
        "calibration_rows": len(calibration),
        "calibration_regression": regression_metrics(calibration, predictor),
        "probability_status": predictor.payload["probability_status"],
        "deployment_promotion": "not_established_until_locked_audit_and_common_bank_training_comparisons",
    }
    write_json_file(directory / "predictor_fit_report.json", report)
    return report


def audit_predictor(directory):
    directory = Path(directory)
    audit = read_json_file(directory / "audit_probes.json", [])
    previous = read_json_file(directory / "fit_probes.json", []) + read_json_file(
        directory / "calibration_probes.json", []
    )
    if not audit:
        raise ValueError("Locked audit is empty.")
    if {r["candidate_group"] for r in audit} & {r["candidate_group"] for r in previous}:
        raise ValueError("Audit candidate leakage.")
    if {r["trajectory_seed"] for r in audit} & {r["trajectory_seed"] for r in previous}:
        raise ValueError("Audit trajectory leakage.")
    predictor = BenefitPredictor.load(directory / "predictor.json")
    if predictor.payload.get("feature_mode") in {"jev", "hybrid"} and any(
        r.get("jev_backend") != "typesafe" for r in audit
    ):
        raise ValueError(
            "Locked Jev audit requires TypeSafe observations, not fixtures."
        )
    by_snapshot = defaultdict(list)
    for row in audit:
        by_snapshot[row["snapshot_id"]].append(row)
    comparisons = []
    for snapshot, rows in by_snapshot.items():
        count = max(1, len(rows) // 4)
        hybrid = sorted(
            rows, key=lambda r: -predictor.predict(r["features"])["expected_gain_nll"]
        )[:count]
        gradient = sorted(rows, key=lambda r: -r["features"]["adam_gain_projection"])[
            :count
        ]
        mean = lambda values: sum(r["gain"] for r in values) / len(values)
        comparisons.append(
            {
                "snapshot": snapshot,
                "top_quartile_gain": mean(hybrid),
                "gradient_only_gain": mean(gradient),
                "uniform_expected_gain": mean(rows),
                "observations": len(rows),
            }
        )
    report = {
        "status": "development_audit_only",
        "regression": regression_metrics(audit, predictor),
        "snapshots": comparisons,
        "promotion": False,
        "promotion_reason": "Requires batch-transfer, full-view and resource checks; ranking results alone are insufficient.",
    }
    write_json_file(directory / "predictor_audit_report.json", report)
    return report


def matched_probe_group(encoded, limit):
    from .encoding import supervised_tokens
    from .token_budget import limit_targets

    counts = [supervised_tokens(x) for x in encoded]
    if limit < len(encoded) or limit > sum(counts):
        raise ValueError("Invalid matched mini-batch target budget.")
    allocations = [1] * len(encoded)
    remaining = limit - len(encoded)
    while remaining:
        for i, count in enumerate(counts):
            if allocations[i] < count and remaining:
                allocations[i] += 1
                remaining -= 1
    return [limit_targets(x, n) for x, n in zip(encoded, allocations)]


def full_view_audit(directory, data, model_path):
    """Full curriculum measurements and matched virtual mini-batch comparisons."""
    from .encoding import supervised_tokens
    from .runtime import load_runtime, cleanup

    directory = Path(directory)
    rows = read_json_file(directory / "audit_probes.json", [])
    predictor = BenefitPredictor.load(directory / "predictor.json")
    by_snapshot = defaultdict(list)
    for row in rows:
        by_snapshot[row["snapshot_id"]].append(row)
    if not by_snapshot:
        raise ValueError("No locked audit snapshots.")
    report = []
    for snapshot, observations in by_snapshot.items():
        if len(observations) < 16:
            raise ValueError(
                "Complete sixteen-candidate snapshots before full-view audit."
            )
        nominal = observations[0]["nominal_tokens"]
        selection = read_json_file(directory / f"audit_selection_{nominal}.json", {})
        samples = {
            candidate_id(
                Training_Sample.model_validate(x)
            ): Training_Sample.model_validate(x)
            for x in selection["samples"]
        }
        model, tokenizer = load_runtime(
            model_path, adapter=observations[0]["snapshot_adapter"]
        )
        started = time.perf_counter()
        try:
            view = data.samples("curriculum_validation")
            baseline = score_samples(model, tokenizer, view)["macro_nll"]
            predicted = sorted(
                observations,
                key=lambda r: -predictor.predict(r["features"])["expected_gain_nll"],
            )
            gradient = sorted(
                observations, key=lambda r: -r["features"]["adam_gain_projection"]
            )
            rng = random.Random(
                observations[0]["trajectory_seed"]
                + int(observations[0]["actual_tokens"])
            )
            selected = predicted[:2] + rng.sample(predicted[2:], 2)

            def measured(encoded):
                calls = 0

                def measure():
                    nonlocal calls
                    calls += 1
                    return (
                        baseline
                        if calls == 1
                        else score_samples(model, tokenizer, view)["macro_nll"]
                    )

                return virtual_update(
                    model,
                    encoded,
                    measure=measure,
                    seed=observations[0]["trajectory_seed"],
                )["gain"]

            candidates = [
                {
                    "candidate_id": r["candidate_id"],
                    "full_curriculum_gain": measured(
                        encode_sample(samples[r["candidate_id"]], tokenizer)
                    ),
                }
                for r in selected
            ]
            options = {
                "learned_head": predicted[:4],
                "gradient_raw": gradient[:4],
                "uniform": rng.sample(observations, 4),
            }
            encoded = {
                name: [
                    encode_sample(samples[r["candidate_id"]], tokenizer) for r in subset
                ]
                for name, subset in options.items()
            }
            budget = min(
                128,
                *(
                    sum(supervised_tokens(x) for x in batch)
                    for batch in encoded.values()
                ),
            )
            batches = {
                name: measured(matched_probe_group(batch, budget))
                for name, batch in encoded.items()
            }
            report.append(
                {
                    "snapshot_id": snapshot,
                    "candidate_probes": candidates,
                    "matched_batch_gains": batches,
                    "batch_supervised_tokens": budget,
                    "curriculum_rows": len(view),
                    "seconds": time.perf_counter() - started,
                    "post_update_validation_forwards": len(view)
                    * (len(candidates) + len(batches)),
                    "live_student_updates": 0,
                }
            )
            write_json_file(directory / "full_view_audit.json", report)
        finally:
            model = tokenizer = None
            cleanup()
    return report
