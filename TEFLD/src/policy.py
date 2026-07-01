from __future__ import annotations

from collections import Counter, defaultdict
import json
import logging
from math import ceil
from statistics import mean
from typing import Any

from .dataschema import (
    CommandQuery,
    Failure_VaultDB,
    Instructor_Recipe,
    Instructor_Slot,
    Pipeline_State,
    learning_record,
    validate_instructor_recipe_size,
)
from .diversity_planner import (
    ASSISTANT_QA_SHAPE_ID,
    CONTEXTUAL_QA_SHAPE_ID,
    VISUAL_GENERATION_SHAPE_ID,
    DiversityPlanner,
    base_generation_tag,
    hard_generation_tag,
    normalize_tag,
    validation_generation_tag,
)
from .helper import (
    append_section_debug_log,
    call_client,
    create_client,
    ensure_current_section,
    load_pipeline_state,
    load_prompt_template,
    load_section_ledger,
    load_section_vault,
    render_prompt_template,
    save_failure_vault,
    save_pipeline_recipe,
    save_pipeline_state,
)


WIDE_VARIETY_TAG = "wide_variety"
WIDE_VARIETY_FRACTION = 0.35
MAX_RECYCLE_SLOTS_PER_ROUND = 3
MAX_RECYCLE_COUNT_PER_ITEM = 2
RECYCLE_COOLDOWN_ROUNDS = 1
HARD_GENERATION_SCORE_THRESHOLD = 0.65
HARD_GENERATION_MIN_ROUND = 3
HARD_GENERATION_HISTORY_WINDOW = 3
HARD_GENERATION_MIN_HITS = 2
HARD_GENERATION_COOLDOWN_ROUNDS = 1
HARD_GENERATION_SLOTS = 3
HARD_GENERATION_MAX_RECYCLE_SLOTS = 2
HARD_VAULT_USABLE_TARGET = 3
HARD_EASY_AVG_LOSS = 0.18
HARD_EASY_P75_LOSS = 0.35
HARD_EASY_MAX_LOSS = 0.50
HARD_VAULT_EASY_LOSS = 0.35
SHAPE_UNDERREPRESENTED_SHARE = 0.15
VALIDATION_WEAKNESS_BASE_FRACTION = 0.30
VALIDATION_WEAKNESS_DEGRADED_FRACTION = 0.20
VALIDATION_WEAKNESS_IMPROVING_FRACTION = 0.45
VALIDATION_WEAKNESS_PLATEAU_FRACTION = 0.50
VALIDATION_WEAKNESS_MIN_SLOTS = 1
VALIDATION_WEAKNESS_MAX_SLOTS = 4
VALIDATION_WEAKNESS_IMPROVEMENT_DELTA = 0.03
VALIDATION_IMPROVEMENT_MIN_DENOMINATOR = 1e-6

# Empty by default on purpose: concrete learning tags should come from the data
# generation API and the loss-grouping API, not from a fixed local taxonomy.
DEFAULT_POLICY_TAGS: list[str] = []

LOGGER = logging.getLogger(__name__)


def wide_variety_slot_count(batch_size: int) -> int:
    """Return how many generated slots should stay open-ended."""
    batch_size = validate_instructor_recipe_size(batch_size)
    return max(1, ceil(batch_size * WIDE_VARIETY_FRACTION))


def focused_generation_slot_count(batch_size: int) -> int:
    """Return how many generated slots should follow discovered loss tags."""
    batch_size = validate_instructor_recipe_size(batch_size)
    return max(0, batch_size - wide_variety_slot_count(batch_size))


def is_exploration_tag(tag: str) -> bool:
    base_tag = base_generation_tag(tag).strip().lower()
    return base_tag.startswith(f"{WIDE_VARIETY_TAG}:") or base_tag == WIDE_VARIETY_TAG


def compose_generation_tag_plan(focused_tags: list[str], batch_size: int) -> list[str]:
    """
    Build a full generation plan with exploration slots first.

    Recycle slots are filled before generation slots, so putting exploration at
    the front preserves the wide-variety budget even when only part of the round
    is newly generated.
    """
    batch_size = validate_instructor_recipe_size(batch_size)
    wide_slots = wide_variety_slot_count(batch_size)
    focused_slots = batch_size - wide_slots
    cleaned_focus = [
        tag.strip()
        for tag in focused_tags
        if tag and tag.strip() and not is_exploration_tag(tag.strip())
    ]

    if not cleaned_focus:
        return [WIDE_VARIETY_TAG for _index in range(batch_size)]

    plan = [WIDE_VARIETY_TAG for _index in range(wide_slots)]
    plan.extend(
        cleaned_focus[index % len(cleaned_focus)]
        for index in range(focused_slots)
    )
    return plan[:batch_size]


class PolicyMaker:
    """
    Builds the next 10-slot instructor recipe from local learning state.

    The policy stays token-efficient by using the compact failure vault and
    section ledger instead of sending the whole dataset to an external model.
    """

    def __init__(
        self,
        section_id: str | None = None,
        *,
        batch_size: int = 10,
        max_recycle_slots: int | None = None,
        default_tags: list[str] | None = None,
        model_id: str = "gpt-5.4-mini",
        use_api_grouping: bool = True,
    ) -> None:
        batch_size = validate_instructor_recipe_size(batch_size)
        if max_recycle_slots is not None and max_recycle_slots < 0:
            raise ValueError("max_recycle_slots cannot be negative.")

        self.section_id = section_id or ensure_current_section()
        self.batch_size = batch_size
        recycle_limit = (
            min(MAX_RECYCLE_SLOTS_PER_ROUND, batch_size)
            if max_recycle_slots is None
            else max_recycle_slots
        )
        self.max_recycle_slots = min(recycle_limit, batch_size)
        self.default_tags = default_tags or list(DEFAULT_POLICY_TAGS)
        self.model_id = model_id
        self.use_api_grouping = use_api_grouping
        self.last_grouping_error: str | None = None

        self.average: float = 0.0
        self.pipeline: Pipeline_State = Pipeline_State(section_id=self.section_id)
        self.learning_data: list[learning_record] = []
        self.fail_vault: Failure_VaultDB = Failure_VaultDB()
        self.refresh()

    def refresh(self) -> None:
        self.pipeline = load_pipeline_state(self.section_id)
        self.learning_data = load_section_ledger(self.section_id)
        self.fail_vault = Failure_VaultDB.model_validate(
            load_section_vault(self.section_id) or {}
        )

    def PercentageLoss(self) -> float:
        """Update and return average loss across the current failure vault."""
        self.refresh()
        self._update_vault_analytics()
        save_failure_vault(self.fail_vault, self.section_id)
        return self.average

    def CreatingLossList(self) -> list[dict[str, int | float | str]]:
        """Return high-loss records in descending priority order."""
        self.refresh()

        if self.fail_vault.top_failures:
            rows = [
                {
                    "ledger_id": item.ledger_id,
                    "tag": item.tag,
                    "token_length": item.token_length,
                    "loss": item.loss,
                }
                for item in self.fail_vault.top_failures
            ]
        else:
            rows = [
                {
                    "ledger_id": row.ledger_id,
                    "tag": row.tag,
                    "token_length": row.token_length,
                    "loss": row.loss,
                }
                for row in self.learning_data
            ]

        return sorted(
            rows,
            key=lambda row: (-float(row["loss"]), int(row["ledger_id"])),
        )

    def build_prompt(self) -> str:
        """Render the optional LLM analysis prompt for the current loss list."""
        template = load_prompt_template("LossTagAvg.txt")
        tag_list = [
            {"tag": row["tag"], "loss": row["loss"]}
            for row in self.CreatingLossList()
        ]
        return render_prompt_template(
            template,
            {"TagList": json.dumps(tag_list, indent=2)},
        )

    def PerTagAvgLoss(self) -> list[dict[str, float | int | str]]:
        """Compute average loss per tag from the ledger, with vault fallback."""
        self.refresh()
        tag_losses: dict[str, list[float]] = defaultdict(list)
        oldest_ledger_id: dict[str, int] = {}

        for row in self.learning_data:
            if row.source == "recycled":
                continue
            if row.tag and row.tag != "good":
                tag_losses[row.tag].append(row.loss)
                oldest_ledger_id[row.tag] = min(
                    oldest_ledger_id.get(row.tag, row.ledger_id),
                    row.ledger_id,
                )

        if not tag_losses:
            for item in self.fail_vault.top_failures:
                if item.tag and item.tag != "good":
                    tag_losses[item.tag].append(item.loss)
                    oldest_ledger_id[item.tag] = min(
                        oldest_ledger_id.get(item.tag, item.ledger_id),
                        item.ledger_id,
                    )

        averages = [
            {
                "tag": tag,
                "avg_loss": mean(losses),
                "count": len(losses),
                "oldest_ledger_id": oldest_ledger_id.get(tag, 0),
                "source_tags": [tag],
            }
            for tag, losses in tag_losses.items()
            if losses
        ]

        return self._sort_metrics(averages)

    def SemanticTagAvgLoss(
        self,
        use_api: bool | None = None,
    ) -> list[dict[str, float | int | str | list[str]]]:
        """Group related tags with the API, falling back to local tag averages."""
        exact_metrics = self.PerTagAvgLoss()
        should_call_api = self.use_api_grouping if use_api is None else use_api
        self.last_grouping_error = None

        if not exact_metrics or not should_call_api:
            return exact_metrics

        try:
            client = create_client()
            response = call_client(
                client=client,
                prompt=self.build_prompt(),
                model_id=self.model_id,
            )
            grouped_metrics = self._parse_grouping_response(response, exact_metrics)
        except Exception as exc:
            self.last_grouping_error = str(exc)
            LOGGER.warning(
                "Semantic tag grouping failed; falling back to exact tag metrics: %s",
                exc,
            )
            return exact_metrics

        return grouped_metrics or exact_metrics

    def WeightTagLoss(self, use_api: bool | None = None) -> list[str]:
        """Return the tag plan for the next batch, weighted by average loss."""
        metrics = self.SemanticTagAvgLoss(use_api=use_api)
        focused_slots = focused_generation_slot_count(self.batch_size)
        if not metrics:
            return compose_generation_tag_plan(
                self._fallback_focus_tags(focused_slots),
                self.batch_size,
            )

        active_metrics = [
            metric
            for metric in metrics
            if str(metric["tag"]).strip()
            and str(metric["tag"]).strip() != "good"
            and not is_exploration_tag(str(metric["tag"]).strip())
        ][:focused_slots]
        if not active_metrics:
            return compose_generation_tag_plan(
                self._fallback_focus_tags(focused_slots),
                self.batch_size,
            )

        allocations: dict[str, int] = {
            str(row["tag"]): 1 for row in active_metrics
        }
        remaining_slots = focused_slots - len(allocations)

        total_loss = sum(float(row["avg_loss"]) for row in active_metrics)
        if remaining_slots > 0 and total_loss > 0:
            remainders: list[tuple[float, float, str]] = []
            for row in active_metrics:
                tag = str(row["tag"])
                raw_share = (float(row["avg_loss"]) / total_loss) * remaining_slots
                whole_slots = int(raw_share)
                allocations[tag] += whole_slots
                remainders.append((raw_share - whole_slots, float(row["avg_loss"]), tag))

            leftovers = focused_slots - sum(allocations.values())
            for _, _, tag in sorted(remainders, reverse=True)[:leftovers]:
                allocations[tag] += 1

        focused_plan = self._expand_allocations(
            allocations,
            active_metrics,
            target_size=focused_slots,
        )
        return compose_generation_tag_plan(focused_plan, self.batch_size)

    @staticmethod
    def percentile(values: list[float], fraction: float) -> float | None:
        if not values:
            return None
        ordered = sorted(values)
        index = min(len(ordered) - 1, max(0, int(round((len(ordered) - 1) * fraction))))
        return ordered[index]

    def current_task_shape_profile(self) -> dict[str, Any]:
        planner = DiversityPlanner(
            pipeline=self.pipeline,
            ledger=self.learning_data,
        )
        profile = planner.task_shape_profile()
        self.pipeline.task_shape_profile = profile
        return profile

    def latest_round_records(self) -> list[learning_record]:
        if not self.learning_data:
            return []
        latest_round = max(row.round_id for row in self.learning_data)
        return [row for row in self.learning_data if row.round_id == latest_round]

    def vault_starvation_score(self) -> tuple[float, dict[str, Any]]:
        status_rows = [
            item
            for item in self.fail_vault.top_failures
            if self.can_recycle_item(item)
        ]
        usable_count = len(status_rows)
        usable_losses = [float(item.loss) for item in status_rows]
        usable_avg_loss = mean(usable_losses) if usable_losses else None

        if self.pipeline.current_round == 0:
            score = 0.0
        elif usable_count == 0:
            score = 1.0
        elif usable_count < HARD_VAULT_USABLE_TARGET:
            score = 0.85
        elif usable_avg_loss is not None and usable_avg_loss < HARD_VAULT_EASY_LOSS:
            score = 0.65
        else:
            score = 0.0

        return score, {
            "usable_count": usable_count,
            "usable_avg_loss": usable_avg_loss,
            "target_usable_count": HARD_VAULT_USABLE_TARGET,
            "easy_loss_threshold": HARD_VAULT_EASY_LOSS,
        }

    def easy_batch_score(self) -> tuple[float, dict[str, Any]]:
        records = self.latest_round_records()
        generated_records = [
            row for row in records if row.source in {None, "generated"}
        ]
        losses = [float(row.loss) for row in generated_records if row.loss is not None]
        if not losses:
            return 0.0, {"record_count": 0}

        avg_loss = mean(losses)
        p75_loss = self.percentile(losses, 0.75)
        max_loss = max(losses)
        score = 0.0
        if avg_loss < HARD_EASY_AVG_LOSS:
            score += 0.45
        if p75_loss is not None and p75_loss < HARD_EASY_P75_LOSS:
            score += 0.35
        if max_loss < HARD_EASY_MAX_LOSS:
            score += 0.20

        return min(1.0, score), {
            "record_count": len(losses),
            "avg_loss": avg_loss,
            "p75_loss": p75_loss,
            "max_loss": max_loss,
            "easy_avg_threshold": HARD_EASY_AVG_LOSS,
            "easy_p75_threshold": HARD_EASY_P75_LOSS,
            "easy_max_threshold": HARD_EASY_MAX_LOSS,
        }

    def validation_plateau_score(self) -> tuple[float, dict[str, Any]]:
        history = list(self.pipeline.round_health_history or [])
        if len(history) < 2:
            return 0.0, {"history_count": len(history)}

        last_event = history[-1]
        best_round_id = self.pipeline.best_round_id
        current_round = self.pipeline.current_round
        rounds_since_best = (
            current_round - best_round_id
            if best_round_id is not None
            else None
        )
        recent_actions = [str(event.get("action")) for event in history[-3:]]
        recent_degraded = sum(
            1 for action in recent_actions if action in {"degraded", "rollback_next"}
        )

        score = 0.0
        if rounds_since_best is not None and rounds_since_best >= 3:
            score += 0.45
        if rounds_since_best is not None and rounds_since_best >= 5:
            score += 0.25
        if recent_degraded >= 1:
            score += 0.25
        if str(last_event.get("action")) == "stable" and rounds_since_best:
            score += 0.10

        return min(1.0, score), {
            "history_count": len(history),
            "best_round_id": best_round_id,
            "rounds_since_best": rounds_since_best,
            "recent_actions": recent_actions,
            "recent_degraded": recent_degraded,
            "last_decision_loss": last_event.get("decision_loss"),
            "last_action": last_event.get("action"),
        }

    @staticmethod
    def shape_tag_aliases(shape_id: str) -> set[str]:
        if shape_id == ASSISTANT_QA_SHAPE_ID:
            return {
                "assistant_qa",
                "question_answering",
                "open_ended_qa",
                "open_qa",
                "general_qa",
                "factual_qa",
            }
        if shape_id == CONTEXTUAL_QA_SHAPE_ID:
            return {
                "contextual_qa",
                "contextual_factual_qa",
                "contextual_question_answering",
                "reading_comprehension",
                "open_book_qa",
            }
        if shape_id == VISUAL_GENERATION_SHAPE_ID:
            return {
                "visual_generation",
                "image_generation",
                "image_prompting",
                "image_editing",
                "visual_asset_generation",
            }
        return {normalize_tag(shape_id)}

    def shape_gap_score(self, profile: dict[str, Any]) -> tuple[float, dict[str, Any]]:
        shape_id = str(profile.get("shape_id") or "")
        confidence = float(profile.get("confidence", 0.0) or 0.0)
        if not shape_id or confidence < 0.50:
            return 0.0, {
                "shape_id": shape_id,
                "confidence": confidence,
                "reason": "low_confidence_or_missing_shape",
            }

        aliases = self.shape_tag_aliases(shape_id)
        generated_records = [
            row
            for row in self.learning_data
            if row.source in {None, "generated"} and row.tag and row.tag != "good"
        ]
        total = len(generated_records)
        matching = sum(
            1
            for row in generated_records
            if normalize_tag(row.tag) in aliases
        )
        share = (matching / total) if total else 0.0

        if total == 0:
            score = 0.55 if shape_id in aliases or shape_id else 0.0
        elif share < SHAPE_UNDERREPRESENTED_SHARE:
            score = min(1.0, 0.55 + confidence * 0.45)
        elif share < SHAPE_UNDERREPRESENTED_SHARE * 1.75:
            score = 0.45
        else:
            score = 0.0

        return score, {
            "shape_id": shape_id,
            "confidence": confidence,
            "aliases": sorted(aliases),
            "generated_record_count": total,
            "matching_record_count": matching,
            "matching_share": share,
            "underrepresented_share_threshold": SHAPE_UNDERREPRESENTED_SHARE,
        }

    def hard_focus_tag(self, profile: dict[str, Any]) -> str:
        profile_tag = str(profile.get("recommended_focus_tag") or "").strip()
        if profile_tag and profile_tag != WIDE_VARIETY_TAG:
            return profile_tag
        most_problematic = self.fail_vault.analytics.most_problematic_category
        if most_problematic:
            return str(most_problematic)
        return WIDE_VARIETY_TAG

    def learning_pressure(self, save: bool = True) -> dict[str, Any]:
        self._update_vault_analytics()
        profile = self.current_task_shape_profile()
        vault_score, vault_details = self.vault_starvation_score()
        easy_score, easy_details = self.easy_batch_score()
        validation_score, validation_details = self.validation_plateau_score()
        shape_score, shape_details = self.shape_gap_score(profile)

        weighted_score = (
            0.30 * vault_score
            + 0.25 * easy_score
            + 0.25 * validation_score
            + 0.20 * shape_score
        )
        candidate_trigger = (
            self.pipeline.current_round >= HARD_GENERATION_MIN_ROUND
            and weighted_score >= HARD_GENERATION_SCORE_THRESHOLD
        )

        previous_state = self.pipeline.learning_pressure_state or {}
        history = list(previous_state.get("history") or [])
        recent_candidates = [
            bool(event.get("candidate_trigger"))
            for event in history[-(HARD_GENERATION_HISTORY_WINDOW - 1) :]
        ]
        recent_candidates.append(candidate_trigger)
        hit_count = sum(1 for item in recent_candidates if item)
        cooldown_until = int(previous_state.get("cooldown_until_round") or -1)
        hard_generation_active = (
            candidate_trigger
            and hit_count >= HARD_GENERATION_MIN_HITS
            and self.pipeline.current_round > cooldown_until
        )
        if hard_generation_active:
            cooldown_until = (
                self.pipeline.current_round + HARD_GENERATION_COOLDOWN_ROUNDS
            )

        event = {
            "round_id": self.pipeline.current_round,
            "score": weighted_score,
            "threshold": HARD_GENERATION_SCORE_THRESHOLD,
            "candidate_trigger": candidate_trigger,
            "hard_generation_active": hard_generation_active,
            "recent_candidate_window": recent_candidates,
            "recent_candidate_hits": hit_count,
            "cooldown_until_round": cooldown_until,
            "hard_slot_count": (
                min(HARD_GENERATION_SLOTS, self.batch_size)
                if hard_generation_active
                else 0
            ),
            "hard_focus_tag": self.hard_focus_tag(profile),
            "component_scores": {
                "vault_starvation": vault_score,
                "easy_batch": easy_score,
                "validation_plateau": validation_score,
                "shape_gap": shape_score,
            },
            "component_details": {
                "vault_starvation": vault_details,
                "easy_batch": easy_details,
                "validation_plateau": validation_details,
                "shape_gap": shape_details,
            },
            "task_shape_profile": profile,
        }
        difficulty_state = self.difficulty_policy_state(pressure=event)
        event["difficulty_policy"] = difficulty_state

        history.append(event)
        self.pipeline.learning_pressure_state = {
            "current": event,
            "history": history[-50:],
            "cooldown_until_round": cooldown_until,
            "weights": {
                "vault_starvation": 0.30,
                "easy_batch": 0.25,
                "validation_plateau": 0.25,
                "shape_gap": 0.20,
            },
        }
        self.pipeline.difficulty_state = difficulty_state
        if save:
            save_pipeline_state(self.pipeline, self.section_id)
            append_section_debug_log(
                self.section_id,
                "learning_pressure_decision",
                event,
            )
        return event

    def validation_weakness_profile(self) -> dict[str, Any]:
        profile = getattr(self.pipeline, "validation_weakness_profile", {}) or {}
        if isinstance(profile, dict) and profile.get("top_categories"):
            return profile

        pressure_state = getattr(self.pipeline, "learning_pressure_state", {}) or {}
        profile = pressure_state.get("validation_weakness_profile") or {}
        return profile if isinstance(profile, dict) else {}

    def validation_weakness_categories(self) -> list[dict[str, Any]]:
        profile = self.validation_weakness_profile()
        categories = profile.get("top_categories")
        if not isinstance(categories, list):
            return []

        cleaned: list[dict[str, Any]] = []
        for category in categories:
            if not isinstance(category, dict):
                continue
            tag = str(category.get("tag") or "").strip()
            if not tag or is_exploration_tag(tag):
                continue
            cleaned.append(category)
        return cleaned[:3]

    def recent_validation_improvement(self) -> float | None:
        scored_history: list[dict[str, Any]] = []
        for event in self.pipeline.round_health_history:
            if event.get("decision_loss") is None:
                continue
            try:
                float(event["decision_loss"])
            except (TypeError, ValueError):
                continue
            scored_history.append(event)

        if len(scored_history) < 2:
            return None

        previous = float(scored_history[-2]["decision_loss"])
        current = float(scored_history[-1]["decision_loss"])
        if previous <= VALIDATION_IMPROVEMENT_MIN_DENOMINATOR:
            return None
        return (previous - current) / previous

    def validation_weakness_allocation(
        self,
        *,
        generation_capacity: int,
        pressure: dict[str, Any],
    ) -> dict[str, Any]:
        categories = self.validation_weakness_categories()
        if generation_capacity <= 0 or not categories:
            return {
                "enabled": False,
                "reason": "no_capacity_or_profile",
                "generation_capacity": generation_capacity,
                "slot_count": 0,
                "fraction": 0.0,
                "tags": [],
                "top_categories": categories,
            }

        latest_health = (
            self.pipeline.round_health_history[-1]
            if self.pipeline.round_health_history
            else {}
        )
        latest_action = str(latest_health.get("action") or "")
        component_scores = pressure.get("component_scores") or {}
        easy_batch_score = float(component_scores.get("easy_batch", 0.0) or 0.0)
        validation_plateau_score = float(
            component_scores.get("validation_plateau", 0.0) or 0.0
        )
        improvement = self.recent_validation_improvement()

        fraction = VALIDATION_WEAKNESS_BASE_FRACTION
        reason = "base"
        if latest_action in {"degraded", "rollback_next", "rollback_unavailable"}:
            fraction = VALIDATION_WEAKNESS_DEGRADED_FRACTION
            reason = f"health_{latest_action}"
        elif easy_batch_score >= 0.80 and validation_plateau_score >= 0.80:
            fraction = VALIDATION_WEAKNESS_PLATEAU_FRACTION
            reason = "easy_training_validation_plateau"
        elif (
            improvement is not None
            and improvement >= VALIDATION_WEAKNESS_IMPROVEMENT_DELTA
        ) or latest_action == "new_best":
            fraction = VALIDATION_WEAKNESS_IMPROVING_FRACTION
            reason = "validation_improving"
        elif improvement is not None and improvement > 0:
            fraction = 0.35
            reason = "validation_slightly_improving"

        slot_count = int(round(generation_capacity * fraction))
        slot_count = max(VALIDATION_WEAKNESS_MIN_SLOTS, slot_count)
        slot_count = min(
            slot_count,
            generation_capacity,
            VALIDATION_WEAKNESS_MAX_SLOTS,
        )

        category_tags = [
            str(category.get("tag") or "").strip()
            for category in categories
            if str(category.get("tag") or "").strip()
        ]
        tags = [
            category_tags[index % len(category_tags)]
            for index in range(slot_count)
        ] if category_tags else []

        return {
            "enabled": bool(tags),
            "reason": reason,
            "generation_capacity": generation_capacity,
            "slot_count": len(tags),
            "fraction": fraction,
            "tags": tags,
            "top_categories": categories,
            "latest_health_action": latest_action or None,
            "validation_improvement": improvement,
            "easy_batch_score": easy_batch_score,
            "validation_plateau_score": validation_plateau_score,
        }

    def difficulty_policy_state(
        self,
        *,
        pressure: dict[str, Any],
    ) -> dict[str, Any]:
        component_scores = pressure.get("component_scores") or {}
        easy_batch_score = float(component_scores.get("easy_batch", 0.0) or 0.0)
        validation_plateau_score = float(
            component_scores.get("validation_plateau", 0.0) or 0.0
        )
        latest_health = (
            self.pipeline.round_health_history[-1]
            if self.pipeline.round_health_history
            else {}
        )
        latest_action = str(latest_health.get("action") or "")
        improvement = self.recent_validation_improvement()

        regime = "stable"
        difficulty_delta = 0
        force_shape_diversification = False
        increase_distractors_only = False
        reason = "hold_current_difficulty"

        if latest_action in {"degraded", "rollback_next", "rollback_unavailable"}:
            regime = "struggling"
            difficulty_delta = -1
            reason = f"health_{latest_action}"
        elif easy_batch_score >= 0.80 and validation_plateau_score >= 0.80:
            regime = "plateau"
            difficulty_delta = 1
            force_shape_diversification = True
            reason = "easy_training_validation_plateau"
        elif easy_batch_score >= 0.80 and improvement is not None and improvement > 0:
            regime = "coasting"
            difficulty_delta = 1
            reason = "training_easy_validation_improving"
        elif improvement is None:
            regime = "noisy"
            increase_distractors_only = True
            reason = "unclear_validation_signal"
        elif improvement > 0:
            regime = "stable"
            reason = "mild_validation_improvement"

        return {
            "round_id": self.pipeline.current_round,
            "regime": regime,
            "difficulty_delta": difficulty_delta,
            "force_shape_diversification": force_shape_diversification,
            "increase_distractors_only": increase_distractors_only,
            "reason": reason,
            "latest_health_action": latest_action or None,
            "validation_improvement": improvement,
            "easy_batch_score": easy_batch_score,
            "validation_plateau_score": validation_plateau_score,
        }

    def InstructionRecipe(self, save: bool = True) -> Instructor_Recipe:
        """Create and optionally save the next instructor recipe."""
        self.refresh()
        tag_plan = self.WeightTagLoss()
        pressure = self.learning_pressure(save=save)
        hard_slot_count = int(pressure.get("hard_slot_count") or 0)
        hard_focus_tag = str(pressure.get("hard_focus_tag") or WIDE_VARIETY_TAG)
        recycle_slot_limit = (
            min(self.max_recycle_slots, HARD_GENERATION_MAX_RECYCLE_SLOTS)
            if hard_slot_count
            else self.max_recycle_slots
        )
        ledger_ids = {row.ledger_id for row in self.learning_data}
        vault_item_status = [
            {
                "ledger_id": item.ledger_id,
                "tag": item.tag,
                "loss": item.loss,
                "recycle_count": item.recycle_count,
                "last_recycled_round": item.last_recycled_round,
                "active": item.active,
                "retired_reason": item.retired_reason,
                "status": self.vault_item_status(item),
                "can_recycle": (
                    item.ledger_id in ledger_ids
                    and self.can_recycle_item(item)
                ),
            }
            for item in self.fail_vault.top_failures
        ]

        slots: list[Instructor_Slot] = []
        recycle_candidates = sorted(
            [
                item
                for item in self.fail_vault.top_failures
                if item.ledger_id in ledger_ids
                and self.can_recycle_item(item)
            ],
            key=lambda item: (-item.loss, item.recycle_count, item.ledger_id),
        )

        for item in recycle_candidates:
            if len(slots) >= recycle_slot_limit:
                break

            slots.append(
                Instructor_Slot(
                    slot_id=len(slots) + 1,
                    command=CommandQuery.RECYCLE,
                    tag=item.tag,
                    failure_ledger_id=item.ledger_id,
                )
            )

        validation_allocation = self.validation_weakness_allocation(
            generation_capacity=self.batch_size - len(slots),
            pressure=pressure,
        )
        for validation_tag in validation_allocation.get("tags", []):
            if len(slots) >= self.batch_size:
                break
            slots.append(
                Instructor_Slot(
                    slot_id=len(slots) + 1,
                    command=CommandQuery.GENERATE,
                    tag=validation_generation_tag(str(validation_tag)),
                )
            )

        for _index in range(hard_slot_count):
            if len(slots) >= self.batch_size:
                break
            slots.append(
                Instructor_Slot(
                    slot_id=len(slots) + 1,
                    command=CommandQuery.GENERATE,
                    tag=hard_generation_tag(hard_focus_tag),
                )
            )

        generation_index = 0
        while len(slots) < self.batch_size and generation_index < len(tag_plan):
            slots.append(
                Instructor_Slot(
                    slot_id=len(slots) + 1,
                    command=CommandQuery.GENERATE,
                    tag=tag_plan[generation_index],
                )
            )
            generation_index += 1

        fallback_tags: list[str] | None = None
        fallback_index = 0
        while len(slots) < self.batch_size:
            if fallback_tags is None:
                fallback_tags = self.WeightTagLoss()
            slots.append(
                Instructor_Slot(
                    slot_id=len(slots) + 1,
                    command=CommandQuery.GENERATE,
                    tag=fallback_tags[fallback_index % len(fallback_tags)],
                )
            )
            fallback_index += 1

        recipe = Instructor_Recipe(slots=slots[: self.batch_size])

        if save:
            self.pipeline = save_pipeline_recipe(recipe, self.section_id)
            used_recycle_ids = [
                slot.failure_ledger_id
                for slot in recipe.slots
                if slot.command == CommandQuery.RECYCLE
                and slot.failure_ledger_id is not None
            ]
            if any(slot.command == CommandQuery.RECYCLE for slot in recipe.slots):
                self.pipeline.consecutive_instructor_no_recycle = 0
            else:
                self.pipeline.consecutive_instructor_no_recycle += 1
            if used_recycle_ids:
                self._mark_vault_items_recycled(used_recycle_ids)
                self.pipeline.recycled_failure_ledger_ids = list(
                    dict.fromkeys(
                        self.pipeline.recycled_failure_ledger_ids
                        + used_recycle_ids
                    )
                )
            save_pipeline_state(self.pipeline, self.section_id)
            append_section_debug_log(
                self.section_id,
                "policy_recipe_decision",
                {
                    "round_id": self.pipeline.current_round,
                    "tag_plan": tag_plan,
                    "learning_pressure": pressure,
                    "task_shape_profile": self.pipeline.task_shape_profile,
                    "hard_slot_count": hard_slot_count,
                    "hard_focus_tag": hard_focus_tag,
                    "recycle_slot_limit": recycle_slot_limit,
                    "validation_weakness_allocation": validation_allocation,
                    "recipe_slots": [
                        slot.model_dump(mode="json") for slot in recipe.slots
                    ],
                    "selected_recycle_ids": used_recycle_ids,
                    "recycle_candidate_ids": [
                        item.ledger_id for item in recycle_candidates
                    ],
                    "vault_total_items": len(self.fail_vault.top_failures),
                    "vault_usable_items": sum(
                        1 for item in vault_item_status if item["can_recycle"]
                    ),
                    "vault_exhausted_items": sum(
                        1
                        for item in vault_item_status
                        if item["status"] == "exhausted"
                    ),
                    "vault_cooling_down_items": sum(
                        1
                        for item in vault_item_status
                        if item["status"] == "cooling_down"
                    ),
                    "vault_retired_items": sum(
                        1
                        for item in vault_item_status
                        if item["status"] == "retired"
                    ),
                    "vault_item_status": vault_item_status,
                    "consecutive_instructor_no_recycle": (
                        self.pipeline.consecutive_instructor_no_recycle
                    ),
                    "grouping_error": self.last_grouping_error,
                },
            )

        return recipe

    def vault_item_status(self, item) -> str:
        if item.recycle_count >= MAX_RECYCLE_COUNT_PER_ITEM:
            return "exhausted"
        if not getattr(item, "active", True) or getattr(item, "retired_reason", None):
            return "retired"
        if item.last_recycled_round is not None:
            rounds_since_recycle = self.pipeline.current_round - item.last_recycled_round
            if rounds_since_recycle <= RECYCLE_COOLDOWN_ROUNDS:
                return "cooling_down"
        return "usable"

    def can_recycle_item(self, item) -> bool:
        return self.vault_item_status(item) == "usable"

    def _fallback_focus_tags(self, target_size: int) -> list[str]:
        state_tags = [
            tag
            for tag, _count in sorted(
                self.pipeline.learning_tag_counts.items(),
                key=lambda item: item[1],
                reverse=True,
            )
            if tag != "good" and not is_exploration_tag(tag)
        ]
        tag_pool = [
            tag
            for tag in (state_tags or self.default_tags)
            if tag and not is_exploration_tag(tag)
        ]
        if not tag_pool:
            return []

        return [
            tag_pool[index % len(tag_pool)]
            for index in range(target_size)
        ]

    def _expand_allocations(
        self,
        allocations: dict[str, int],
        metrics: list[dict[str, float | int | str | list[str]]],
        *,
        target_size: int,
    ) -> list[str]:
        ordered_tags = [str(row["tag"]) for row in metrics]
        tag_plan: list[str] = []

        for tag in ordered_tags:
            tag_plan.extend([tag] * allocations.get(tag, 0))

        if len(tag_plan) < target_size:
            fallback_tags = self._fallback_focus_tags(target_size) or ordered_tags
            fallback_index = 0
            while fallback_tags and len(tag_plan) < target_size:
                tag_plan.append(fallback_tags[fallback_index % len(fallback_tags)])
                fallback_index += 1

        return tag_plan[:target_size]

    def _parse_grouping_response(
        self,
        response: str,
        exact_metrics: list[dict[str, float | int | str | list[str]]],
    ) -> list[dict[str, float | int | str | list[str]]]:
        raw_response = json.loads(response)
        raw_tags = raw_response.get("top_tags")
        if not isinstance(raw_tags, list):
            raise ValueError("Grouped tag response must include top_tags list.")

        exact_by_tag = {
            str(metric["tag"]): metric
            for metric in exact_metrics
        }

        grouped_metrics: list[dict[str, float | int | str | list[str]]] = []
        for item in raw_tags:
            if not isinstance(item, dict):
                continue

            tag = str(item.get("tag", "")).strip()
            if not tag:
                continue

            source_tags = item.get("source_tags")
            if not isinstance(source_tags, list):
                source_tags = [tag] if tag in exact_by_tag else []
            source_tags = [
                str(source_tag).strip()
                for source_tag in source_tags
                if str(source_tag).strip()
            ]

            oldest_ids = [
                int(exact_by_tag[source_tag].get("oldest_ledger_id", 0))
                for source_tag in source_tags
                if source_tag in exact_by_tag
            ]

            grouped_metrics.append(
                {
                    "tag": tag,
                    "avg_loss": float(item["avg_loss"]),
                    "count": len(source_tags) or 1,
                    "oldest_ledger_id": min(oldest_ids) if oldest_ids else 0,
                    "source_tags": source_tags,
                }
            )

        return self._sort_metrics(grouped_metrics)

    def _sort_metrics(
        self,
        metrics: list[dict[str, float | int | str | list[str]]],
    ) -> list[dict[str, float | int | str | list[str]]]:
        return sorted(
            metrics,
            key=lambda row: (
                -float(row["avg_loss"]),
                int(row.get("oldest_ledger_id", 0)) or 10**12,
                -int(row.get("count", 1)),
            ),
        )

    def _mark_vault_items_recycled(self, ledger_ids: list[int]) -> None:
        used_ids = set(ledger_ids)
        for item in self.fail_vault.top_failures:
            if item.ledger_id in used_ids:
                item.recycle_count += 1
                item.last_recycled_round = self.pipeline.current_round
                if item.recycle_count >= MAX_RECYCLE_COUNT_PER_ITEM:
                    item.active = False
                    item.retired_reason = "max_recycle_count"
        self._update_vault_analytics()
        save_failure_vault(self.fail_vault, self.section_id)

    def _update_vault_analytics(self) -> None:
        for item in self.fail_vault.top_failures:
            if item.recycle_count >= MAX_RECYCLE_COUNT_PER_ITEM:
                item.active = False
                item.retired_reason = item.retired_reason or "max_recycle_count"

        losses = [item.loss for item in self.fail_vault.top_failures]
        self.average = mean(losses) if losses else 0.0
        self.fail_vault.analytics.avg_loss = self.average
        self.fail_vault.analytics.total_failures_tracked = len(
            self.fail_vault.top_failures
        )
        status_counts = Counter(
            self.vault_item_status(item) for item in self.fail_vault.top_failures
        )
        self.fail_vault.analytics.active_failures = (
            status_counts["usable"] + status_counts["cooling_down"]
        )
        self.fail_vault.analytics.usable_failures = status_counts["usable"]
        self.fail_vault.analytics.cooling_down_failures = status_counts["cooling_down"]
        self.fail_vault.analytics.exhausted_failures = status_counts["exhausted"]
        self.fail_vault.analytics.retired_failures = status_counts["retired"]

        if self.fail_vault.top_failures:
            self.fail_vault.analytics.avg_token_length = mean(
                item.token_length for item in self.fail_vault.top_failures
            )
            tag_counts = Counter(item.tag for item in self.fail_vault.top_failures)
            self.fail_vault.analytics.most_problematic_category = (
                tag_counts.most_common(1)[0][0]
            )
        else:
            self.fail_vault.analytics.avg_token_length = 0.0
            self.fail_vault.analytics.most_problematic_category = None
