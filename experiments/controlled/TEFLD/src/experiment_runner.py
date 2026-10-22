"""Isolated, resumable, token-matched TEFLD/Dolly experiments.

This runner never selects or writes a legacy active section. Protected test
scoring is a separate command after a frozen study completion check.
"""

from __future__ import annotations

from collections import Counter
from contextlib import contextmanager
import math
from pathlib import Path
import random
import time

from .candidate_pool import CandidatePool, Teacher, candidate_id
from .dataschema import Training_Sample
from .dataset_curriculum import (
    DATASET_METHODS,
    require_dataset_policy,
    reference_sample,
    retrieve_references,
    require_reference_provenance,
)
from .encoding import encode_sample, supervised_tokens
from .example_budget import example_target, cap_new_examples
from .helper import read_json_file, write_json_file, stable_json_hash
from .runtime import cleanup, file_fingerprint, load_runtime, set_seed
from .token_budget import TokenBudget, limit_targets, optimizer_group
from .transactions import section_lock
from .validation import validate_sample
from .validation_data import SplitData, score_samples, generate_answers

METHODS = {
    "tefld_validation",
    "tefld_loss",
    "dolly_baseline",
    "tefld_gradient",
    "tefld_benefit",
    "tefld_jev",
    "tefld_uniform",
} | set(DATASET_METHODS)


def select_checkpoint(candidates):
    if not candidates:
        raise ValueError("No checkpoint selection candidates.")
    if any(not math.isfinite(x["macro_nll"]) for x in candidates):
        raise ValueError("Nonfinite checkpoint score.")
    return min(candidates, key=lambda x: (x["macro_nll"], x["tokens"]))


def encode_batch(samples, tokenizer, max_length):
    return [encode_sample(x, tokenizer, max_length) for x in samples]


def save_adapter(model, tokenizer, path, metadata):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    marker = path / "completion.json"
    write_json_file(marker, {"complete": False})
    model.save_pretrained(path, safe_serialization=True)
    tokenizer.save_pretrained(path)
    weights = path / "adapter_model.safetensors"
    if not weights.exists():
        raise FileNotFoundError("Adapter weights were not saved.")
    write_json_file(
        marker,
        {**metadata, "complete": True, "weights_sha256": file_fingerprint(weights)},
    )


def require_adapter(path):
    path = Path(path)
    metadata = read_json_file(path / "completion.json", {})
    if not metadata.get("complete") or metadata.get(
        "weights_sha256"
    ) != file_fingerprint(path / "adapter_model.safetensors"):
        raise ValueError("Incomplete or changed experiment adapter.")
    return metadata


class ExperimentRunner:
    def __init__(
        self,
        protocol,
        output,
        *,
        model_path=None,
        allow_paid_api=False,
        gradient_geometry="raw",
        projection_dimensions=1024,
    ):
        if gradient_geometry not in {"raw", "subspace"}:
            raise ValueError("Unknown gradient geometry.")
        self.gradient_geometry = gradient_geometry
        self.projection_dimensions = projection_dimensions
        self.data = SplitData(protocol)
        self.protocol = self.data.protocol
        from .experiment_settings import TrainingSettings

        self.training_settings = TrainingSettings.from_protocol(self.protocol)
        self.study_root = Path(output).resolve()
        comparison = self.protocol.get("teacher_comparison", {})
        self.arm = comparison.get("arm")
        if self.arm and (
            not isinstance(self.arm, str)
            or not self.arm.replace("-", "").replace("_", "").isalnum()
        ):
            raise ValueError("Invalid teacher arm directory name.")
        self.output = (
            self.study_root / "arms" / self.arm if self.arm else self.study_root
        )
        self.output.mkdir(parents=True, exist_ok=True)
        self.verifier = None
        verifier_settings = self.protocol["tefld_controls"].get("verifier")
        if verifier_settings:
            self.verifier = Teacher(
                self.study_root / "verifier_cache",
                verifier_settings["model_id"],
                allow_network=allow_paid_api,
                provider=verifier_settings["provider"],
                study_budget_root=self.study_root if comparison else None,
            )
        self.model_path = model_path or self.protocol["common_training"]["model_path"]
        self.teacher = Teacher(
            self.output / "teacher_cache",
            self.protocol["tefld_controls"]["teacher_model_id"],
            self.protocol["tefld_controls"]["teacher_temperature"],
            allow_network=allow_paid_api,
            provider=self.protocol["tefld_controls"].get("provider"),
            prompt_variant=self.protocol["tefld_controls"].get(
                "generation_prompt_variant", "full"
            ),
            verifier=self.verifier,
            study_budget_root=self.study_root if comparison else None,
        )
        self.verifier = self.verifier or self.teacher
        self.verification_audit_path = (
            self.study_root if comparison else self.output
        ) / "verification_audit.json"
        source_dir = Path(__file__).resolve().parent
        self.code_hash = stable_json_hash(
            {p.name: file_fingerprint(p) for p in sorted(source_dir.glob("*.py"))}
        )
        self.protocol_hash = file_fingerprint(self.data.protocol_path)
        self.categories = self.protocol["dataset"]["categories"]
        self.compute_times = {
            "optimizer_wall_seconds": 0.0,
            "evaluation_wall_seconds": 0.0,
        }
        from .jev_scorer import JevScorer

        self.jev = JevScorer(self.output / "jev_cache")

    def model_identity(self):
        path = Path(self.model_path)
        files = sorted(
            p
            for p in path.iterdir()
            if p.suffix == ".safetensors"
            or p.name
            in {
                "config.json",
                "tokenizer.json",
                "tokenizer_config.json",
                "chat_template.jinja",
                "generation_config.json",
                "model.safetensors.index.json",
            }
        )
        if not any(p.suffix == ".safetensors" for p in files):
            raise FileNotFoundError("Local model weight files are required.")
        stats = {
            p.name: {"bytes": p.stat().st_size, "mtime_ns": p.stat().st_mtime_ns}
            for p in files
        }
        manifest_path = self.output / "model_fingerprint.json"
        cached = read_json_file(manifest_path, {})
        if (
            cached.get("model_path") == str(path.resolve())
            and cached.get("file_stats") == stats
        ):
            return cached["identity_sha256"]
        print("Fingerprinting local model files before training.", flush=True)
        hashes = {p.name: file_fingerprint(p) for p in files}
        identity = stable_json_hash({"files": hashes})
        write_json_file(
            manifest_path,
            {
                "model_path": str(path.resolve()),
                "file_stats": stats,
                "sha256": hashes,
                "identity_sha256": identity,
            },
        )
        return identity

    def _score_samples(self, *args, **kwargs):
        started = time.perf_counter()
        try:
            return score_samples(*args, **kwargs)
        finally:
            self.compute_times["evaluation_wall_seconds"] += (
                time.perf_counter() - started
            )

    def _optimizer_group(self, *args, **kwargs):
        started = time.perf_counter()
        try:
            return optimizer_group(*args, **kwargs)
        finally:
            self.compute_times["optimizer_wall_seconds"] += (
                time.perf_counter() - started
            )

    def run_directory(self, phase, method, seed):
        if method not in METHODS or phase not in {
            "pilot",
            "primary",
            "auxiliary",
            "selector",
            "ablation",
        }:
            raise ValueError("Unknown method or phase.")
        return self.output / phase / f"{method}_seed_{seed}"

    @contextmanager
    def _network_policy(self, method):
        clients = [self.teacher, self.verifier, self.jev]
        previous = [(client, client.allow_network) for client in clients]
        if method in DATASET_METHODS or method == "dolly_baseline":
            for client in clients:
                client.allow_network = False
        try:
            yield
        finally:
            for client, enabled in previous:
                client.allow_network = enabled

    def run(
        self,
        method,
        seed,
        *,
        phase="pilot",
        budget=5000,
        per_category=8,
        replay=True,
        candidate_bank=None,
        predictor=None,
        probe_callback=None,
        resume_pilot_after_fix=False,
    ):
        directory = self.run_directory(phase, method, seed)
        with section_lock(self.study_root), section_lock(self.output), section_lock(
            directory
        ), self._network_policy(method), self.teacher.account_to(
            arm=self.arm or "legacy", phase=phase, method=method, seed=seed
        ):
            return self._run(
                directory,
                method,
                seed,
                phase,
                budget,
                per_category,
                replay,
                candidate_bank,
                predictor,
                probe_callback,
                resume_pilot_after_fix,
            )

    def _run(
        self,
        directory,
        method,
        seed,
        phase,
        budget,
        per_category,
        replay,
        candidate_bank,
        predictor,
        probe_callback,
        resume_pilot_after_fix=False,
    ):
        import torch

        unique_target = example_target(self.protocol, phase, method)
        if method in DATASET_METHODS:
            policy = require_dataset_policy(self.protocol)
            if method not in policy.get("enabled_methods", []):
                raise ValueError(
                    "Dataset feedback method is not declared in this protocol."
                )
            if (
                candidate_bank is not None
                or predictor is not None
                or probe_callback is not None
            ):
                raise ValueError(
                    "Dataset TEFLD uses the original training pool, without a synthetic bank, predictor or external probe callback."
                )
        state_path = directory / "state.json"
        config = {
            "method": method,
            "seed": seed,
            "phase": phase,
            "budget": budget,
            "per_category": per_category,
            "replay": replay,
            "protocol_sha256": self.protocol_hash,
            "code_sha256": self.code_hash,
            "model_path": str(self.model_path),
            "model_sha256": self.model_identity(),
            "gradient_geometry": self.gradient_geometry,
            "projection_dimensions": self.projection_dimensions,
            "source_mode": (
                "adaptive_dolly_training_pool"
                if method in DATASET_METHODS
                else (
                    "fixed_dolly_bank_development"
                    if candidate_bank
                    else (
                        "direct_dolly"
                        if method == "dolly_baseline"
                        else "live_verified_synthetic"
                    )
                )
            ),
            "bank_sha256": (
                stable_json_hash({"bank": candidate_bank}) if candidate_bank else None
            ),
            "predictor_sha256": (
                stable_json_hash(predictor.payload) if predictor else None
            ),
        }
        if unique_target is not None:
            config["unique_training_examples"] = unique_target
        state = read_json_file(state_path, {})
        if state and state["config"] != config:
            old = state["config"]
            if not (
                resume_pilot_after_fix
                and phase == "pilot"
                and state.get("status") != "complete"
                and not state.get("batch")
                and {k: v for k, v in old.items() if k != "code_sha256"}
                == {k: v for k, v in config.items() if k != "code_sha256"}
            ):
                raise ValueError(
                    "Run identity changed; use a fresh output directory, or explicitly adopt a code fix for an interrupted pilot with --resume-pilot-after-fix."
                )
            if state.get("last_adapter"):
                require_adapter(state["last_adapter"])
            backup = directory / f"state_before_code_fix_{old['code_sha256'][:12]}.json"
            if not backup.exists():
                write_json_file(backup, state)
            state.setdefault("pilot_code_amendments", []).append(
                {
                    "previous_code_sha256": old["code_sha256"],
                    "new_code_sha256": config["code_sha256"],
                    "tokens": state["tokens"],
                    "round": state["round"],
                    "scope": "amended development pilot; not a single-revision primary experiment",
                }
            )
            state["config"] = config
            write_json_file(state_path, state)
        if state.get("status") == "complete":
            return state
        if phase == "primary" and candidate_bank:
            raise ValueError(
                "Common-bank runs are selector ablations, not primary synthetic-data comparisons."
            )
        if phase == "primary":
            if self.protocol.get("dataset_tefld", {}).get("primary_ready") is False:
                raise PermissionError(
                    "Resolve the common optimizer protocol before primary dataset comparisons."
                )
            if (
                self.protocol.get("teacher_comparison", {}).get("primary_ready")
                is False
            ):
                raise PermissionError(
                    "Resolve the optimizer-control protocol before primary teacher comparisons."
                )
            frozen = read_json_file(self.output / "frozen_study.json", {})
            if not frozen or frozen.get("test_scored"):
                raise PermissionError(
                    "Freeze an unscored study before primary training."
                )
            if any(
                frozen.get(k) != config[k]
                for k in ("code_sha256", "protocol_sha256", "model_sha256")
            ):
                raise PermissionError(
                    "Primary run differs from the frozen study identity."
                )
            if (
                method not in frozen["methods"]
                or seed not in frozen["seeds"]
                or budget != frozen["budget"]
                or not replay
                or per_category is not None
            ):
                raise PermissionError(
                    "Primary run settings differ from the frozen study."
                )
            if method == "tefld_benefit" and config["predictor_sha256"] != frozen.get(
                "predictor_sha256"
            ):
                raise PermissionError(
                    "Primary benefit predictor differs from the frozen head."
                )
        if method == "tefld_benefit" and predictor is None:
            raise ValueError(
                "Hybrid arm requires a frozen validated predictor; no silent gradient-only substitution."
            )
        if method == "tefld_jev" and predictor is None:
            raise ValueError(
                "Jev-only control requires a separately fitted Jev feature head."
            )
        if (
            method in {"tefld_jev", "tefld_benefit"}
            and predictor.payload.get("feature_mode")
            != {"tefld_jev": "jev", "tefld_benefit": "hybrid"}[method]
        ):
            raise ValueError("Predictor mode does not match the experimental arm.")
        if (
            method != "dolly_baseline"
            and method not in DATASET_METHODS
            and not candidate_bank
        ):
            audit = read_json_file(self.verification_audit_path, {})
            if (
                audit.get("status") != "passed"
                or audit.get("model") != self.verifier.model
                or audit.get("teacher_identity") != self.verifier.identity
            ):
                raise RuntimeError(
                    "Run audit-verifier successfully before live synthetic generation."
                )
        if not state:
            state = {
                "config": config,
                "status": "ready",
                "tokens": 0,
                "round": 0,
                "ledger": [],
                "presentations": [],
                "batch": [],
                "seen": [],
                "baseline_cursor": 0,
                "candidates": [],
                "losses": [],
                "weakness": {},
                "rounds": [],
            }
            write_json_file(state_path, state)
        if state.get("last_adapter"):
            require_adapter(state["last_adapter"])
        self.compute_times = state.get(
            "compute_accounting",
            {"optimizer_wall_seconds": 0.0, "evaluation_wall_seconds": 0.0},
        )
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        model, tokenizer = load_runtime(
            self.model_path,
            seed=seed,
            adapter=state.get("last_adapter"),
            training_config=self.training_settings.model_dump(),
        )
        started = time.perf_counter()
        initial_tokens = state["tokens"]
        pool = CandidatePool(
            self.data, tokenizer, self.teacher, directory / "quarantine.json"
        )
        pool.remember(
            [Training_Sample.model_validate(x["sample"]) for x in state["ledger"]]
        )
        pool.remember(
            [
                Training_Sample.model_validate(x["sample"])
                for x in read_json_file(pool.admissions_path, {}).values()
            ]
        )
        budget_state = TokenBudget(budget, state["tokens"])
        milestones = [
            m
            for m in self.protocol["checkpoint_selection"][
                "nominal_supervised_token_milestones"
            ]
            if m <= budget
        ]
        if unique_target is not None:
            # Count-matched runs score one final checkpoint, with the usual rows.
            milestones = []
        elif budget not in milestones:
            milestones.append(budget)
        completed_milestones = {x["nominal_tokens"] for x in state["candidates"]}
        baseline_order = self._baseline_order(seed)
        try:
            if probe_callback and state["round"] > 0:
                probe_callback(model, tokenizer, state)
            while budget_state.remaining and (
                unique_target is None or len(state["ledger"]) < unique_target
            ):
                round_id = state["round"]
                self.teacher.owner["round"] = round_id
                set_seed(seed + round_id)
                rng = random.Random(seed * 1000003 + round_id)
                if not state["batch"]:
                    batch = self._batch(
                        state,
                        method,
                        seed,
                        round_id,
                        rng,
                        pool,
                        model,
                        tokenizer,
                        replay,
                        candidate_bank,
                        predictor,
                        baseline_order,
                    )
                    if unique_target is not None:
                        batch, omitted = cap_new_examples(
                            batch,
                            state["ledger"],
                            unique_target,
                            random.Random(seed * 1000037 + round_id),
                        )
                        state.setdefault("untrained_final_admissions", []).extend(
                            omitted
                        )
                    # Store admissions before training so restart never regenerates them.
                    state["batch"] = [x.model_dump(mode="json") for x in batch]
                    write_json_file(state_path, state)
                write_json_file(
                    directory / f"batch_{round_id:04d}.json", state["batch"]
                )
                batch = [Training_Sample.model_validate(x) for x in state["batch"]]
                if method in DATASET_METHODS or (
                    method == "dolly_baseline" and self.protocol.get("dataset_tefld")
                ):
                    for sample in batch:
                        require_reference_provenance(
                            self.data, sample, tokenizer, pool.guard
                        )
                pool.remember(batch)
                encoded = encode_batch(
                    batch, tokenizer, self.protocol["common_training"]["max_length"]
                )
                if unique_target is not None:
                    expected = (
                        sum(supervised_tokens(x) for x in encoded)
                        * self.protocol["common_training"]["epochs_per_round"]
                    )
                    if expected > budget_state.remaining:
                        raise RuntimeError(
                            "Safety token ceiling would truncate a count-matched round; "
                            "no partial-answer training is allowed."
                        )
                set_seed(seed + round_id)
                order_rng = random.Random(seed * 1000033 + round_id)
                presentations = []
                for epoch in range(
                    self.protocol["common_training"]["epochs_per_round"]
                ):
                    order = list(range(len(batch)))
                    order_rng.shuffle(order)
                    presentations.extend((epoch, i) for i in order)
                optimizer = torch.optim.AdamW(
                    [p for p in model.parameters() if p.requires_grad],
                    lr=self.training_settings.learning_rate,
                    betas=(0.9, 0.999),
                    eps=1e-8,
                    weight_decay=0,
                )
                steps = math.ceil(
                    len(presentations)
                    / self.training_settings.gradient_accumulation_steps
                )
                scheduler = torch.optim.lr_scheduler.LambdaLR(
                    optimizer, lambda step: max(0.0, 1 - step / max(1, steps))
                )
                group = []
                group_meta = []
                used = Counter()
                round_started = time.perf_counter()
                model.train()
                for epoch, index in presentations:
                    available = budget_state.remaining - sum(
                        supervised_tokens(x) for x in group
                    )
                    if available < 1:
                        break
                    item = limit_targets(
                        encoded[index],
                        min(available, supervised_tokens(encoded[index])),
                    )
                    group.append(item)
                    group_meta.append((epoch, index))
                    if (
                        len(group) == self.training_settings.gradient_accumulation_steps
                        or sum(supervised_tokens(x) for x in group)
                        == budget_state.remaining
                    ):
                        count, losses = self._optimizer_group(model, optimizer, group)
                        scheduler.step()
                        budget_state.commit(count)
                        for item, (epoch, index), loss in zip(
                            group, group_meta, losses
                        ):
                            used[index] += 1
                            state["losses"].append(
                                {
                                    "round": round_id,
                                    "epoch": epoch,
                                    "batch_index": index,
                                    "source": batch[index].source,
                                    "candidate_id": candidate_id(batch[index]),
                                    "loss": loss,
                                    "answer_tokens": supervised_tokens(item),
                                    "tokens_after_group": budget_state.consumed,
                                }
                            )
                        group = []
                        group_meta = []
                        self._milestones(
                            model,
                            tokenizer,
                            state,
                            directory,
                            budget_state.consumed,
                            milestones,
                            completed_milestones,
                            per_category,
                        )
                        if not budget_state.remaining:
                            break
                if group:
                    count, losses = self._optimizer_group(model, optimizer, group)
                    scheduler.step()
                    budget_state.commit(count)
                    for item, (epoch, index), loss in zip(group, group_meta, losses):
                        used[index] += 1
                        state["losses"].append(
                            {
                                "round": round_id,
                                "epoch": epoch,
                                "batch_index": index,
                                "source": batch[index].source,
                                "candidate_id": candidate_id(batch[index]),
                                "loss": loss,
                                "answer_tokens": supervised_tokens(item),
                                "tokens_after_group": budget_state.consumed,
                            }
                        )
                    self._milestones(
                        model,
                        tokenizer,
                        state,
                        directory,
                        budget_state.consumed,
                        milestones,
                        completed_milestones,
                        per_category,
                    )
                if not used:
                    raise RuntimeError("Round made no progress.")
                post = self._score_samples(
                    model,
                    tokenizer,
                    [(i, batch[i]) for i in used],
                    self.protocol["common_training"]["max_length"],
                )
                for row in post["rows"]:
                    i = row["row_id"]
                    sample = batch[i]
                    identity = candidate_id(sample)
                    entry = next(
                        (x for x in state["ledger"] if x["id"] == identity), None
                    )
                    if entry is None:
                        entry = {
                            "id": identity,
                            "sample": sample.model_dump(mode="json"),
                            "loss": row["loss"],
                            "first_round": round_id,
                            "replay_uses": 0,
                            "last_replay_round": None,
                        }
                        state["ledger"].append(entry)
                    entry.update(loss=row["loss"], loss_round=round_id)
                    if sample.source == "recycled":
                        entry["replay_uses"] += 1
                        entry["last_replay_round"] = round_id
                    state["presentations"].append(
                        {
                            "round": round_id,
                            "id": identity,
                            "source": sample.source,
                            "epoch_presentations": used[i],
                        }
                    )
                if method != "dolly_baseline":
                    probe = []
                    for category in self.categories:
                        probe.extend(
                            [
                                x
                                for x in reversed(state["ledger"])
                                if x["sample"]["tag"] == category
                            ][:16]
                        )
                    fresh = self._score_samples(
                        model,
                        tokenizer,
                        [
                            (i, Training_Sample.model_validate(x["sample"]))
                            for i, x in enumerate(probe)
                        ],
                    )
                    for row in fresh["rows"]:
                        probe[row["row_id"]].update(
                            loss=row["loss"], loss_round=round_id
                        )
                    fresh["training_candidate_ids"] = [x["id"] for x in probe]
                    fresh["student_tokens"] = budget_state.consumed
                    write_json_file(
                        directory / f"ledger_probe_{round_id:04d}.json", fresh
                    )
                    if method in {"tefld_loss", "tefld_dataset_loss"}:
                        state["weakness"] = fresh["categories"]
                if method in {
                    "tefld_validation",
                    "tefld_gradient",
                    "tefld_benefit",
                    "tefld_jev",
                    "tefld_dataset_validation",
                }:
                    evaluation = self._score_samples(
                        model,
                        tokenizer,
                        self.data.samples("curriculum_validation", per_category),
                    )
                    state["weakness"] = evaluation["categories"]
                    write_json_file(
                        directory / f"curriculum_{round_id:04d}.json", evaluation
                    )
                adapter = directory / f"round_{round_id:04d}"
                save_adapter(
                    model,
                    tokenizer,
                    adapter,
                    {
                        "config_sha256": stable_json_hash(config),
                        "tokens": budget_state.consumed,
                        "round": round_id,
                    },
                )
                state.update(
                    tokens=budget_state.consumed,
                    round=round_id + 1,
                    batch=[],
                    last_adapter=str(adapter),
                    status="running",
                )
                state["rounds"].append(
                    {
                        "round": round_id,
                        "tokens": budget_state.consumed,
                        "round_training_and_evaluation_seconds": time.perf_counter()
                        - round_started,
                        "examples": len(used),
                        "partial_final_round": sum(used.values()) < len(presentations),
                        "unused_candidate_ids": [
                            candidate_id(batch[i])
                            for i in range(len(batch))
                            if not used[i]
                        ],
                    }
                )
                # Whole run state is one atomic file; adapter marker is durable first.
                state["compute_accounting"] = {
                    **self.compute_times,
                    "scope": "Cumulative committed training/evaluation wall time; excludes teacher network requests, model loading and failed uncommitted rounds. Not GPU-kernel time.",
                }
                write_json_file(state_path, state)
                from .round_logging import write_round_events

                write_round_events(directory, state)
                print(
                    (
                        f"{phase} {method} seed={seed} round={round_id} "
                        f'unique_examples={len(state["ledger"])}/{unique_target} '
                        f'tokens={state["tokens"]}'
                        if unique_target is not None
                        else f'{phase} {method} seed={seed} round={round_id} tokens={state["tokens"]}/{budget}'
                    ),
                    flush=True,
                )
                if probe_callback:
                    probe_callback(model, tokenizer, state)
            if unique_target is not None:
                if len(state["ledger"]) != unique_target:
                    raise RuntimeError("Unique-example target was not reached.")
                self._milestones(
                    model,
                    tokenizer,
                    state,
                    directory,
                    budget_state.consumed,
                    [budget_state.consumed],
                    completed_milestones,
                    per_category,
                )
                state["compute_accounting"].update(self.compute_times)
                state["example_count_control"] = {
                    "unique_trained": len(state["ledger"]),
                    "answer_token_exposures": budget_state.consumed,
                    "token_matched": False,
                    "full_round_epochs": self.protocol["common_training"][
                        "epochs_per_round"
                    ],
                    "untrained_admissions": len(
                        state.get("untrained_final_admissions", [])
                    ),
                }
            state.update(
                status="complete",
                selected_checkpoint=select_checkpoint(state["candidates"]),
                final_checkpoint=state["candidates"][-1],
                seconds_this_session=time.perf_counter() - started,
                tokens_this_session=budget_state.consumed - initial_tokens,
                peak_gpu_bytes=(
                    torch.cuda.max_memory_allocated()
                    if torch.cuda.is_available()
                    else None
                ),
            )
            if method in DATASET_METHODS:
                state["dataset_control"] = {
                    "feedback_source": DATASET_METHODS[method],
                    "source_role": "training_pool",
                    "original_answers_unchanged": True,
                    "provider_calls": 0,
                    "synthetic_generation": False,
                }
            write_json_file(state_path, state)
            return state
        finally:
            del model, tokenizer
            cleanup()

    def _milestones(
        self,
        model,
        tokenizer,
        state,
        directory,
        tokens,
        milestones,
        completed,
        per_category,
    ):
        for nominal in milestones:
            if nominal in completed or tokens < nominal:
                continue
            path = directory / f"checkpoint_{nominal:06d}"
            save_adapter(
                model,
                tokenizer,
                path,
                {
                    "tokens": tokens,
                    "nominal_tokens": nominal,
                    "protocol_sha256": self.protocol_hash,
                    "code_sha256": self.code_hash,
                },
            )
            evaluation = self._score_samples(
                model,
                tokenizer,
                self.data.samples("checkpoint_validation", per_category),
            )
            write_json_file(path / "validation.json", evaluation)
            state["candidates"].append(
                {
                    "nominal_tokens": nominal,
                    "tokens": tokens,
                    "path": str(path),
                    "macro_nll": evaluation["macro_nll"],
                }
            )
            completed.add(nominal)

    def _baseline_order(self, seed):
        rng = random.Random(seed)
        groups = {}
        for category in self.categories:
            ids = sorted(
                i
                for i in self.data.ids["training_pool"]
                if self.data.rows[i]["category"] == category
            )
            rng.shuffle(ids)
            groups[category] = ids
        order = []
        while any(groups.values()):
            categories = self.categories.copy()
            rng.shuffle(categories)
            for category in categories:
                if groups[category]:
                    order.append(groups[category].pop())
        return order

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
        if (
            round_id == 0
            and self.protocol.get("dataset_tefld")
            and self.arm
            and (method == "dolly_baseline" or method in DATASET_METHODS)
        ):
            path = self.study_root / f"shared_initial_seed_{seed}.json"
            plan = read_json_file(self.study_root / "study_plan.json", {})
            if not path.exists() or plan.get("warmup_sha256", {}).get(
                str(seed)
            ) != file_fingerprint(path):
                raise ValueError(
                    "Shared dataset-study warmup identity changed or was not prepared."
                )
            batch = [
                Training_Sample.model_validate(x) for x in read_json_file(path, [])
            ]
            if len(batch) != 10 or len({candidate_id(x) for x in batch}) != 10:
                raise ValueError(
                    "Shared dataset warmup must contain ten distinct references."
                )
            for sample in batch:
                require_reference_provenance(self.data, sample, tokenizer, pool.guard)
            state["seen"] = sorted(
                set(state["seen"]) | {candidate_id(x) for x in batch}
            )
            state.setdefault("dataset_selection_traces", []).append(
                {
                    "round": 0,
                    "signal": "shared_teacher_study_reference_warmup",
                    "row_ids": [x.verification["row_id"] for x in batch],
                }
            )
            return batch
        if method == "dolly_baseline" or (method in DATASET_METHODS and round_id == 0):
            selected = []
            matched = bool(self.protocol.get("dataset_tefld"))
            seen = set(state["seen"])
            while len(selected) < 10:
                if state["baseline_cursor"] >= len(baseline_order):
                    raise RuntimeError("Training pool exhausted.")
                row_id = baseline_order[state["baseline_cursor"]]
                state["baseline_cursor"] += 1
                sample = self.data.sample(row_id, "training_pool")
                if matched and candidate_id(sample) in seen:
                    continue
                try:
                    if matched:
                        require_dataset_policy(self.protocol)
                        sample = reference_sample(
                            self.data, row_id, tokenizer, pool.guard
                        )
                    else:
                        validate_sample(sample).require()
                except ValueError as error:
                    path = pool.quarantine
                    write_json_file(
                        path,
                        read_json_file(path, [])
                        + [{"row_id": row_id, "reasons": [str(error)]}],
                    )
                    continue
                selected.append(sample)
                seen.add(candidate_id(sample))
            if matched:
                state["seen"] = sorted(seen)
                state.setdefault("dataset_selection_traces", []).append(
                    {
                        "round": round_id,
                        "signal": (
                            "shared_seeded_dolly_warmup"
                            if round_id == 0
                            else "baseline_shuffle"
                        ),
                        "row_ids": [x.verification["row_id"] for x in selected],
                    }
                )
            return selected
        if round_id == 0 and not bank:
            comparison = self.protocol.get("teacher_comparison", {})
            if comparison:
                path = self.study_root / f"shared_initial_seed_{seed}.json"
                if not path.exists():
                    raise RuntimeError(
                        "Prepare the common training-only warmup before a teacher comparison."
                    )
                plan = read_json_file(self.study_root / "study_plan.json", {})
                import hashlib

                if (
                    plan.get("warmup_sha256", {}).get(str(seed))
                    != hashlib.sha256(path.read_bytes()).hexdigest()
                ):
                    raise ValueError(
                        "Shared teacher-study warmup identity changed or was not prepared."
                    )
            else:
                path = self.output / f"shared_initial_seed_{seed}.json"
            cached = read_json_file(path, [])
            if cached:
                batch = [Training_Sample.model_validate(x) for x in cached]
                for x in batch:
                    validate_sample(x).require()
                    pool.guard.require(x)
                    encode_sample(x, tokenizer)
                state["seen"] = sorted(
                    set(state["seen"]) | {candidate_id(x) for x in batch}
                )
                return batch
            seen = set(state["seen"])
            categories = self.categories + [
                rng.choice(self.categories) for _ in range(2)
            ]
            batch = [
                pool.create(c, f"initial:{seed}:{i}", seen)
                for i, c in enumerate(categories)
            ]
            write_json_file(path, [x.model_dump(mode="json") for x in batch])
            state["seen"] = sorted(seen)
            return batch
        eligible = [
            x
            for x in state["ledger"]
            if x.get("loss_round") == round_id - 1
            and x["loss"] >= 0.35
            and x["replay_uses"] < 2
            and (
                x["last_replay_round"] is None or round_id - x["last_replay_round"] > 1
            )
        ]
        eligible = sorted(eligible, key=lambda x: (-x["loss"], x["id"]))[:10]
        selected_replays = eligible[:3] if replay else []
        recycled = []
        for entry in selected_replays:
            sample = Training_Sample.model_validate(entry["sample"]).model_copy(
                deep=True
            )
            sample.source = "recycled"
            sample.source_ledger_id = state["ledger"].index(entry) + 1
            validate_sample(sample).require()
            recycled.append(sample)
        focus = 10 - len(recycled) - 4
        fallback = (
            sum(state["weakness"].values()) / len(state["weakness"])
            if state["weakness"]
            else 1.0
        )
        weights = [
            max(0.00001, state["weakness"].get(c, fallback)) for c in self.categories
        ]
        quotas = Counter(rng.choices(self.categories, weights=weights, k=focus))
        exploration = [rng.choice(self.categories) for _ in range(4)]
        if method in DATASET_METHODS:
            seen = set(state["seen"])
            new, trace = retrieve_references(
                self.data,
                tokenizer,
                pool.guard,
                baseline_order,
                seen,
                quotas,
                exploration,
            )
            state["seen"] = sorted(seen)
            state.setdefault("dataset_selection_traces", []).append(
                {
                    "round": round_id,
                    "signal": DATASET_METHODS[method],
                    "weakness": dict(state["weakness"]),
                    "focus_quotas": dict(quotas),
                    "exploration_categories": exploration,
                    "replay_candidate_ids": [candidate_id(x) for x in recycled],
                    "new_references": trace,
                }
            )
            return recycled + new
        if method in {"tefld_gradient", "tefld_benefit"} and round_id > 0:
            from .benefit_features import candidate_features, gradient_summary

            views = read_json_file(
                self.data.protocol_path.parent / "benefit_curriculum_views.json", {}
            )
            view = [
                (i, self.data.sample(i, "curriculum_validation"))
                for i in views["views"]["gradient_features"]
            ]
            summary = gradient_summary(
                model,
                tokenizer,
                view,
                dimensions=self.projection_dimensions,
                include_subspace=self.gradient_geometry == "subspace",
            )

            def selector(candidates, count):
                values = []
                selection_started = time.perf_counter()
                for sample in candidates:
                    features = candidate_features(
                        model,
                        tokenizer,
                        sample,
                        summary,
                        progress=state["tokens"] / max(1, state["config"]["budget"]),
                        seed=seed + round_id,
                    )
                    if method == "tefld_benefit":
                        features.update(
                            self.jev.features(
                                sample, state["weakness"], candidate_id(sample)
                            )
                        )
                        gain = predictor.predict(features)["expected_gain_nll"]
                    elif predictor:
                        gain = predictor.predict(features)["expected_gain_nll"]
                    else:
                        gain = (
                            features["subspace_adam_gain"]
                            if self.gradient_geometry == "subspace"
                            else features["adam_gain_projection"]
                        )
                    values.append((gain, candidate_id(sample), sample))
                trace = state.setdefault("selection_traces", [])
                ranked = sorted(values, key=lambda x: (-x[0], x[1]))
                trace.append(
                    {
                        "round": round_id,
                        "method": method,
                        "scores": [
                            {"candidate_id": x[1], "gain": x[0], "selected": i < count}
                            for i, x in enumerate(ranked)
                        ],
                        "seconds": time.perf_counter() - selection_started,
                    }
                )
                return [
                    x[2] for x in sorted(values, key=lambda x: (-x[0], x[1]))[:count]
                ]

        elif method == "tefld_jev" and round_id > 0:

            def selector(candidates, count):
                values = []
                for sample in candidates:
                    encoded = encode_sample(sample, tokenizer)
                    features = {
                        "candidate_nll": score_samples(model, tokenizer, [(0, sample)])[
                            "macro_nll"
                        ],
                        "target_tokens": float(supervised_tokens(encoded)),
                        "sequence_tokens": float(len(encoded["input_ids"])),
                        "progress": state["tokens"] / max(1, state["config"]["budget"]),
                    }
                    features.update(
                        {"family_" + c: float(sample.tag == c) for c in self.categories}
                    )
                    features.update(
                        self.jev.features(
                            sample, state["weakness"], candidate_id(sample)
                        )
                    )
                    values.append(
                        (
                            predictor.predict(features)["expected_gain_nll"],
                            candidate_id(sample),
                            sample,
                        )
                    )
                return [
                    x[2] for x in sorted(values, key=lambda x: (-x[0], x[1]))[:count]
                ]

        else:
            selector = lambda candidates, count: rng.sample(candidates, count)
        if bank:
            # Fixed-bank variants never invoke the teacher, including exploration.
            candidates = [
                Training_Sample.model_validate(x)
                for x in bank
                if candidate_id(Training_Sample.model_validate(x)) not in state["seen"]
            ]
            for sample in candidates:
                validate_sample(sample).require()
                pool.guard.require(sample)
                encode_sample(sample, tokenizer)
            new = []
            for category, count in sorted(quotas.items()):
                options = [x for x in candidates if x.tag == category and x not in new][
                    : 2 * count
                ]
                if len(options) < count:
                    raise RuntimeError("Common bank cannot satisfy category quota.")
                new += selector(options, count)
            for category in exploration:
                options = [x for x in candidates if x.tag == category and x not in new]
                if not options:
                    raise RuntimeError("Common bank cannot satisfy exploration quota.")
                new.append(rng.choice(options))
            state["seen"] = sorted(set(state["seen"]) | {candidate_id(x) for x in new})
        else:
            seen = set(state["seen"])
            new = pool.recipe(
                quotas,
                exploration,
                f"{method}:{seed}:{round_id}",
                seen,
                state["weakness"],
                selector,
            )
            state["seen"] = sorted(seen)
        return recycled + new

    def freeze(self, methods, seeds, budget=30000, predictor=None):
        if self.protocol.get("dataset_tefld", {}).get("primary_ready") is False:
            raise PermissionError(
                "The dataset comparison is development-only until the common optimizer recipe is frozen."
            )
        if self.protocol.get("teacher_comparison", {}).get("primary_ready") is False:
            raise PermissionError(
                "The prepared two-teacher study is development-only until optimizer controls and live checks are resolved."
            )
        if any(
            m
            not in {"tefld_validation", "tefld_loss", "dolly_baseline", "tefld_benefit"}
            | set(DATASET_METHODS)
            for m in methods
        ):
            raise ValueError("Declare exploratory variants separately.")
        if any(method in DATASET_METHODS for method in methods):
            policy = require_dataset_policy(self.protocol)
            if any(
                m in DATASET_METHODS and m not in policy.get("enabled_methods", [])
                for m in methods
            ):
                raise ValueError(
                    "Dataset feedback method is not declared in this protocol."
                )
        if (
            not methods
            or len(set(methods)) != len(methods)
            or not seeds
            or len(set(seeds)) != len(seeds)
            or budget <= 0
        ):
            raise ValueError(
                "Nonempty unique methods/seeds and a positive budget are required."
            )
        if "tefld_benefit" in methods:
            if predictor is None or predictor.payload.get("feature_mode") != "hybrid":
                raise PermissionError(
                    "The hybrid arm needs a frozen hybrid predictor and promotion evidence."
                )
            promotion = read_json_file(
                self.output / "benefit_calibration" / "promotion.json", {}
            )
            if (
                promotion.get("promotion") is not True
                or promotion.get("code_sha256") != self.code_hash
                or promotion.get("protocol_sha256") != self.protocol_hash
                or promotion.get("predictor_sha256")
                != stable_json_hash(predictor.payload)
            ):
                raise PermissionError(
                    "The hybrid arm needs documented, identity-matched promotion evidence."
                )
        path = self.output / "frozen_study.json"
        payload = {
            "methods": methods,
            "seeds": seeds,
            "budget": budget,
            "protocol_sha256": self.protocol_hash,
            "code_sha256": self.code_hash,
            "model_sha256": self.model_identity(),
            "test_scored": False,
            "predictor_sha256": (
                stable_json_hash(predictor.payload) if predictor else None
            ),
            "benefit_status": (
                "deferred_exploratory"
                if "tefld_benefit" not in methods
                else "requires_documented_promotion"
            ),
        }
        existing = read_json_file(path, {})
        if existing and existing != payload:
            raise ValueError(
                "Study already frozen differently; use a new study/test contract."
            )
        write_json_file(path, payload)

    def evaluate_final(self, *, include_generated=False):
        frozen = read_json_file(self.output / "frozen_study.json", {})
        if not frozen:
            raise PermissionError("Freeze the study before test scoring.")
        if (
            frozen["protocol_sha256"] != self.protocol_hash
            or frozen["code_sha256"] != self.code_hash
        ):
            raise PermissionError("Frozen experiment identity changed.")
        states = []
        for method in frozen["methods"]:
            for seed in frozen["seeds"]:
                state = read_json_file(
                    self.run_directory("primary", method, seed) / "state.json", {}
                )
                if (
                    state.get("status") != "complete"
                    or state.get("tokens") != frozen["budget"]
                ):
                    raise PermissionError(
                        f"Final test remains locked: {method}/{seed} incomplete."
                    )
                if (
                    state["config"]["protocol_sha256"] != self.protocol_hash
                    or state["config"]["code_sha256"] != self.code_hash
                ):
                    raise PermissionError(
                        "Completed run identity differs from frozen study."
                    )
                if state["config"]["model_sha256"] != frozen["model_sha256"]:
                    raise PermissionError("Completed run used a different base model.")
                if method == "tefld_benefit" and state["config"][
                    "predictor_sha256"
                ] != frozen.get("predictor_sha256"):
                    raise PermissionError(
                        "Completed run used a different benefit head."
                    )
                states.append(state)
        if frozen.get("model_sha256") != self.model_identity():
            raise PermissionError("Frozen model files changed.")
        self.data.final_test_enabled = True
        results = []
        answer_results = []
        try:
            for state in states:
                for kind in ["final_checkpoint", "selected_checkpoint"]:
                    checkpoint = state[kind]
                    require_adapter(checkpoint["path"])
                    model, tokenizer = load_runtime(
                        self.model_path,
                        seed=state["config"]["seed"],
                        adapter=checkpoint["path"],
                        train=False,
                    )
                    try:
                        evaluation = self._score_samples(
                            model, tokenizer, self.data.samples("final_test")
                        )
                        results.append(
                            {
                                "method": state["config"]["method"],
                                "seed": state["config"]["seed"],
                                "checkpoint_kind": kind,
                                "tokens": checkpoint["tokens"],
                                "evaluation": evaluation,
                            }
                        )
                        write_json_file(
                            self.output / "final_test_results.json", results
                        )
                        if include_generated:
                            answers = generate_answers(
                                model,
                                tokenizer,
                                self.data.samples("final_test"),
                                max_new_tokens=self.protocol["metrics"][
                                    "generated_answers"
                                ]["max_new_tokens"],
                            )
                            answer_results.append(
                                {
                                    "method": state["config"]["method"],
                                    "seed": state["config"]["seed"],
                                    "checkpoint_kind": kind,
                                    "rows": answers,
                                }
                            )
                            write_json_file(
                                self.output / "generated_answers.json", answer_results
                            )
                    finally:
                        del model, tokenizer
                        cleanup()
            model, tokenizer = load_runtime(self.model_path, train=False)
            try:
                write_json_file(
                    self.output / "base_reference_test.json",
                    self._score_samples(
                        model, tokenizer, self.data.samples("final_test")
                    ),
                )
            finally:
                del model, tokenizer
                cleanup()
            frozen["test_scored"] = True
            write_json_file(self.output / "frozen_study.json", frozen)
            return results
        finally:
            self.data.final_test_enabled = False
