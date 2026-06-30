from collections import Counter
import math

from .dataschema import (
    Evaluation_Output,
    Failure_VaultDB,
    Vault_Analytics,
    failure_vault_item,
    learning_record,
)
from .helper import (
    append_section_debug_log,
    get_section_ledger_path,
    load_pipeline_state,
    load_section_vault,
    load_section_ledger,
    save_pipeline_state,
    write_model_list_file,
    save_failure_vault,
)
from .student import TrainyModel

VAULT_MAX_ITEMS = 10
VAULT_RETIRE_LOSS_THRESHOLD = 0.35
VAULT_MIN_ENTRY_LOSS_FLOOR = 0.35
VAULT_MIN_ROUND_CANDIDATES = 3
VAULT_ACTIVE_MIN_USABLE = 3
VAULT_RECENT_REFILL_ROUNDS = 3
VAULT_RECENT_REFILL_LOSS_FLOOR = 0.15
VAULT_MAX_RECYCLE_COUNT_PER_ITEM = 2
VAULT_RECYCLE_COOLDOWN_ROUNDS = 1


def finite_loss_value(value: object) -> float | None:
    try:
        loss = float(value)
    except (TypeError, ValueError):
        return None
    return loss if math.isfinite(loss) else None


def is_finite_loss(value: object) -> bool:
    return finite_loss_value(value) is not None


class Evaluator:
    """
    Evaluates the current student batch after training.

    The loss is teacher-forced cross-entropy on the gold summary only:
    prompt tokens are masked out, so high loss means the student model is
    surprised by the correct answer.
    """

    def __init__(
        self,
        model_name: str,
        *,
        student: TrainyModel | None = None,
        generate_answers: bool = False,
    ) -> None:
        self.student = student or TrainyModel(model_name)
        self.generate_answers = generate_answers
        self.section_id = self.student.section_id
        self.pipeline = self.student.pipeline

    @staticmethod
    def preview_text(text: str | None, limit: int = 220) -> str:
        return " ".join((text or "").split())[:limit]

    def evaluation_debug_row(self, evaluation: Evaluation_Output) -> dict:
        loss = finite_loss_value(evaluation.loss)
        return {
            "source": evaluation.source,
            "source_ledger_id": evaluation.source_ledger_id,
            "tag": evaluation.tag,
            "loss": loss if loss is not None else str(evaluation.loss),
            "token_length": evaluation.token_length,
            "instruct_preview": self.preview_text(evaluation.instruct),
            "sample_preview": self.preview_text(evaluation.sample),
            "student_answer_preview": self.preview_text(evaluation.student_ans),
            "gold_preview": self.preview_text(evaluation.gold_summary),
        }

    @staticmethod
    def vault_item_debug_row(item: failure_vault_item) -> dict:
        loss = finite_loss_value(item.loss)
        last_loss = finite_loss_value(item.last_loss)
        best_recycle_loss = finite_loss_value(item.best_recycle_loss)
        return {
            "ledger_id": item.ledger_id,
            "tag": item.tag,
            "loss": loss if loss is not None else str(item.loss),
            "token_length": item.token_length,
            "recycle_count": item.recycle_count,
            "last_recycled_round": item.last_recycled_round,
            "active": item.active,
            "retired_reason": item.retired_reason,
            "first_seen_round": item.first_seen_round,
            "last_seen_round": item.last_seen_round,
            "last_loss": last_loss if last_loss is not None else str(item.last_loss),
            "best_recycle_loss": (
                best_recycle_loss
                if best_recycle_loss is not None
                else str(item.best_recycle_loss)
            ),
        }

    def evaluate_current_batch(self) -> list[Evaluation_Output]:
        if not self.pipeline.current_training_batch:
            raise ValueError("There is no current training batch to evaluate.")
 
        if self.student.model is None:
            self.student.load_trained_model()
        return self.student.evaluate_samples(
            self.pipeline.current_training_batch,
            generate_answers=self.generate_answers,
        )

    def append_evaluations_to_ledger(
        self,
        evaluations: list[Evaluation_Output],
    ) -> list[learning_record]:
        ledger = load_section_ledger(self.section_id)
        next_ledger_id = max((row.ledger_id for row in ledger), default=0) + 1

        new_records: list[learning_record] = []
        for offset, evaluation in enumerate(evaluations):
            new_records.append(
                learning_record(
                    ledger_id=next_ledger_id + offset,
                    sample=evaluation.sample,
                    instruct=evaluation.instruct,
                    gold_summary=evaluation.gold_summary,
                    student_ans=evaluation.student_ans,
                    token_length=evaluation.token_length,
                    loss=evaluation.loss,
                    tag=evaluation.tag,
                    round_id=self.pipeline.current_round,
                    source=evaluation.source,
                    source_ledger_id=evaluation.source_ledger_id,
                )
            )

        updated_ledger = ledger + new_records
        write_model_list_file(get_section_ledger_path(self.section_id), updated_ledger)
        state = load_pipeline_state(self.section_id)
        state.processed_samples = updated_ledger
        self.pipeline = state
        save_pipeline_state(state, self.section_id)
        return updated_ledger

    @staticmethod
    def vault_item_status(item: failure_vault_item, current_round: int) -> str:
        if item.recycle_count >= VAULT_MAX_RECYCLE_COUNT_PER_ITEM:
            return "exhausted"
        if not item.active or item.retired_reason:
            return "retired"
        if item.last_recycled_round is not None:
            rounds_since_recycle = current_round - item.last_recycled_round
            if rounds_since_recycle <= VAULT_RECYCLE_COOLDOWN_ROUNDS:
                return "cooling_down"
        return "usable"

    @classmethod
    def can_recycle_item_for_round(
        cls,
        item: failure_vault_item,
        current_round: int,
    ) -> bool:
        return cls.vault_item_status(item, current_round) == "usable"

    @classmethod
    def active_usable_count(
        cls,
        items: list[failure_vault_item],
        current_round: int,
    ) -> int:
        return sum(
            1
            for item in items
            if cls.can_recycle_item_for_round(item, current_round)
        )

    @staticmethod
    def item_from_record(
        row: learning_record,
        existing_item: failure_vault_item | None = None,
    ) -> failure_vault_item:
        first_seen_round = (
            existing_item.first_seen_round
            if existing_item and existing_item.first_seen_round is not None
            else row.round_id
        )
        return failure_vault_item(
            ledger_id=row.ledger_id,
            tag=row.tag,
            token_length=row.token_length,
            loss=row.loss,
            recycle_count=(
                existing_item.recycle_count if existing_item is not None else 0
            ),
            last_recycled_round=(
                existing_item.last_recycled_round
                if existing_item is not None
                else None
            ),
            active=(
                existing_item.active
                if existing_item is not None
                else True
            ),
            retired_reason=(
                existing_item.retired_reason
                if existing_item is not None
                else None
            ),
            first_seen_round=first_seen_round,
            last_seen_round=row.round_id,
            last_loss=row.loss,
            best_recycle_loss=(
                existing_item.best_recycle_loss
                if existing_item is not None
                else None
            ),
        )

    @classmethod
    def keep_priority(
        cls,
        item: failure_vault_item,
        current_round: int,
    ) -> tuple[int, float, int]:
        status = cls.vault_item_status(item, current_round)
        status_rank = {
            "usable": 4,
            "cooling_down": 3,
            "exhausted": 1,
            "retired": 0,
        }.get(status, 0)
        seen_round = item.last_seen_round if item.last_seen_round is not None else -1
        loss = finite_loss_value(item.loss)
        return (
            status_rank,
            loss if loss is not None else float("-inf"),
            int(seen_round),
        )

    @classmethod
    def add_or_replace_vault_item(
        cls,
        vault_items: dict[int, failure_vault_item],
        candidate: failure_vault_item,
        *,
        current_round: int,
        max_items: int,
        force: bool = False,
    ) -> None:
        if candidate.ledger_id in vault_items:
            vault_items[candidate.ledger_id] = candidate
            return

        if len(vault_items) < max_items:
            vault_items[candidate.ledger_id] = candidate
            return

        lowest_item = min(
            vault_items.values(),
            key=lambda item: cls.keep_priority(item, current_round),
        )
        if force or (
            cls.keep_priority(candidate, current_round)
            > cls.keep_priority(lowest_item, current_round)
        ):
            del vault_items[lowest_item.ledger_id]
            vault_items[candidate.ledger_id] = candidate

    @classmethod
    def mark_vault_statuses(
        cls,
        vault_items: dict[int, failure_vault_item],
    ) -> None:
        for item in vault_items.values():
            loss = finite_loss_value(item.loss)
            if loss is None:
                item.active = False
                item.retired_reason = item.retired_reason or "nonfinite_loss"
            elif loss < VAULT_RECENT_REFILL_LOSS_FLOOR:
                item.active = False
                item.retired_reason = item.retired_reason or "low_loss"
            elif item.recycle_count > 0 and loss <= VAULT_RETIRE_LOSS_THRESHOLD:
                item.active = False
                item.retired_reason = item.retired_reason or "low_loss_after_recycle"
            elif item.recycle_count >= VAULT_MAX_RECYCLE_COUNT_PER_ITEM:
                item.active = False
                item.retired_reason = item.retired_reason or "max_recycle_count"

    @staticmethod
    def recent_generated_records(
        ledger: list[learning_record],
        *,
        current_round: int,
        window: int = VAULT_RECENT_REFILL_ROUNDS,
    ) -> list[learning_record]:
        min_round = max(0, current_round - window + 1)
        return [
            row
            for row in ledger
            if row.source != "recycled"
            and row.round_id >= min_round
            and (finite_loss_value(row.loss) or float("-inf"))
            >= VAULT_RECENT_REFILL_LOSS_FLOOR
        ]

    def build_failure_vault(
        self,
        ledger: list[learning_record],
        evaluations: list[Evaluation_Output] | None = None,
        max_items: int = VAULT_MAX_ITEMS,
    ) -> Failure_VaultDB:
        existing_vault = Failure_VaultDB.model_validate(
            load_section_vault(self.section_id) or {}
        )
        vault_items = {
            item.ledger_id: item
            for item in existing_vault.top_failures
            if is_finite_loss(item.loss)
        }
        ledger = [row for row in ledger if is_finite_loss(row.loss)]
        evaluations = [
            evaluation
            for evaluation in (evaluations or [])
            if is_finite_loss(evaluation.loss)
        ]

        self.update_recycled_vault_scores(vault_items, evaluations)
        self.mark_vault_statuses(vault_items)

        current_round_records = [
            row
            for row in ledger
            if row.round_id == self.pipeline.current_round
            and row.source != "recycled"
        ]
        minimum_entry_loss = self.minimum_vault_entry_loss(
            current_round_records=current_round_records,
            current_vault_items=list(vault_items.values()),
        )

        sorted_round_records = sorted(
            current_round_records,
            key=lambda record: (-(finite_loss_value(record.loss) or 0.0), record.ledger_id),
        )
        priority_round_ids = {
            row.ledger_id
            for row in sorted_round_records[:VAULT_MIN_ROUND_CANDIDATES]
            if (finite_loss_value(row.loss) or float("-inf"))
            >= VAULT_MIN_ENTRY_LOSS_FLOOR
        }

        for row in sorted_round_records:
            row_loss = finite_loss_value(row.loss)
            if row_loss is None:
                continue
            if (
                row_loss < minimum_entry_loss
                and row.ledger_id not in priority_round_ids
            ):
                continue

            existing_item = vault_items.get(row.ledger_id)
            candidate = self.item_from_record(row, existing_item)
            self.add_or_replace_vault_item(
                vault_items,
                candidate,
                current_round=self.pipeline.current_round + 1,
                max_items=max_items,
            )

        next_round = self.pipeline.current_round + 1
        usable_count = self.active_usable_count(
            list(vault_items.values()),
            next_round,
        )
        if usable_count < VAULT_ACTIVE_MIN_USABLE:
            refill_needed = VAULT_ACTIVE_MIN_USABLE - usable_count
            refill_records = sorted(
                self.recent_generated_records(
                    ledger,
                    current_round=self.pipeline.current_round,
                ),
                key=lambda row: (-(finite_loss_value(row.loss) or 0.0), -row.round_id, row.ledger_id),
            )
            added = 0
            for row in refill_records:
                if added >= refill_needed:
                    break
                if row.ledger_id in vault_items:
                    continue

                candidate = self.item_from_record(row)
                self.add_or_replace_vault_item(
                    vault_items,
                    candidate,
                    current_round=next_round,
                    max_items=max_items,
                    force=True,
                )
                added += 1

        self.mark_vault_statuses(vault_items)

        top_failures = sorted(
            vault_items.values(),
            key=lambda item: (
                -self.keep_priority(item, self.pipeline.current_round + 1)[0],
                -(finite_loss_value(item.loss) or 0.0),
                item.ledger_id,
            ),
        )[:max_items]

        return Failure_VaultDB(
            top_failures=top_failures,
            analytics=self.build_vault_analytics(
                top_failures,
                minimum_entry_loss=minimum_entry_loss,
                current_round=self.pipeline.current_round + 1,
            ),
        )

    def update_recycled_vault_scores(
        self,
        vault_items: dict[int, failure_vault_item],
        evaluations: list[Evaluation_Output],
    ) -> None:
        for evaluation in evaluations:
            loss = finite_loss_value(evaluation.loss)
            if (
                evaluation.source != "recycled"
                or evaluation.source_ledger_id is None
                or loss is None
            ):
                continue

            item = vault_items.get(evaluation.source_ledger_id)
            if item is None:
                continue

            item.loss = loss
            item.last_loss = loss
            item.token_length = evaluation.token_length
            item.tag = evaluation.tag
            item.last_seen_round = self.pipeline.current_round
            if item.first_seen_round is None:
                item.first_seen_round = self.pipeline.current_round
            if (
                item.best_recycle_loss is None
                or loss < item.best_recycle_loss
            ):
                item.best_recycle_loss = loss

            if loss <= VAULT_RETIRE_LOSS_THRESHOLD:
                item.active = False
                item.retired_reason = "low_loss_after_recycle"
            elif item.recycle_count >= VAULT_MAX_RECYCLE_COUNT_PER_ITEM:
                item.active = False
                item.retired_reason = "max_recycle_count"

    @staticmethod
    def prune_low_loss_vault_items(
        vault_items: dict[int, failure_vault_item],
        minimum_loss: float = VAULT_MIN_ENTRY_LOSS_FLOOR,
    ) -> None:
        for ledger_id, item in list(vault_items.items()):
            loss = finite_loss_value(item.loss)
            if loss is None or loss < minimum_loss:
                del vault_items[ledger_id]

    @staticmethod
    def average_loss_values(losses: list[float]) -> float | None:
        finite_losses = [
            loss
            for loss in (finite_loss_value(value) for value in losses)
            if loss is not None
        ]
        if not finite_losses:
            return None
        return sum(finite_losses) / len(finite_losses)

    @classmethod
    def average_lowest_half_loss(
        cls,
        vault_items: list[failure_vault_item],
    ) -> float | None:
        if not vault_items:
            return None

        sorted_losses = sorted(
            loss
            for loss in (finite_loss_value(item.loss) for item in vault_items)
            if loss is not None
        )
        if not sorted_losses:
            return None
        half_count = max(1, len(sorted_losses) // 2)
        return cls.average_loss_values(sorted_losses[:half_count])

    @classmethod
    def minimum_vault_entry_loss(
        cls,
        *,
        current_round_records: list[learning_record],
        current_vault_items: list[failure_vault_item],
    ) -> float:
        thresholds = [VAULT_MIN_ENTRY_LOSS_FLOOR]
        round_average = cls.average_loss_values(
            [row.loss for row in current_round_records]
        )
        lowest_half_average = cls.average_lowest_half_loss(current_vault_items)

        dynamic_thresholds = [
            threshold
            for threshold in (round_average, lowest_half_average)
            if threshold is not None
        ]
        if dynamic_thresholds:
            thresholds.append(min(dynamic_thresholds))

        return max(thresholds)

    @classmethod
    def build_vault_analytics(
        cls,
        top_failures: list[failure_vault_item],
        minimum_entry_loss: float = VAULT_MIN_ENTRY_LOSS_FLOOR,
        current_round: int = 0,
    ) -> Vault_Analytics:
        tag_counts = Counter(item.tag for item in top_failures)
        status_counts = Counter(
            cls.vault_item_status(item, current_round)
            for item in top_failures
        )
        avg_token_length = (
            sum(item.token_length for item in top_failures) / len(top_failures)
            if top_failures
            else 0.0
        )
        finite_losses = [
            loss
            for loss in (finite_loss_value(item.loss) for item in top_failures)
            if loss is not None
        ]
        avg_loss = (
            sum(finite_losses) / len(finite_losses)
            if finite_losses
            else 0.0
        )

        return Vault_Analytics(
            avg_token_length=avg_token_length,
            most_problematic_category=(
                tag_counts.most_common(1)[0][0] if tag_counts else None
            ),
            total_failures_tracked=len(top_failures),
            active_failures=(
                status_counts["usable"] + status_counts["cooling_down"]
            ),
            usable_failures=status_counts["usable"],
            cooling_down_failures=status_counts["cooling_down"],
            exhausted_failures=status_counts["exhausted"],
            retired_failures=status_counts["retired"],
            avg_loss=avg_loss,
            minimum_entry_loss=minimum_entry_loss,
        )

    def advance_round(self) -> None:
        state = load_pipeline_state(self.section_id)
        state.current_round = max(
            int(state.current_round),
            int(self.pipeline.current_round),
        ) + 1
        state.active_recipe = []
        state.current_training_batch = []
        self.pipeline = state
        save_pipeline_state(state, self.section_id)

    def run(self) -> tuple[list[Evaluation_Output], list[learning_record], Failure_VaultDB]:
        evaluations = self.evaluate_current_batch()
        finite_evaluations = [
            evaluation for evaluation in evaluations if is_finite_loss(evaluation.loss)
        ]
        dropped_evaluations = [
            evaluation for evaluation in evaluations if not is_finite_loss(evaluation.loss)
        ]
        if dropped_evaluations:
            append_section_debug_log(
                self.section_id,
                "nonfinite_evaluations_dropped",
                {
                    "round_id": self.pipeline.current_round,
                    "dropped_count": len(dropped_evaluations),
                    "evaluations": [
                        self.evaluation_debug_row(evaluation)
                        for evaluation in dropped_evaluations
                    ],
                },
            )
        ledger = self.append_evaluations_to_ledger(finite_evaluations)
        vault = save_failure_vault(
            self.build_failure_vault(ledger, evaluations=finite_evaluations),
            self.section_id,
        )
        append_section_debug_log(
            self.section_id,
            "vault_update",
            {
                "round_id": self.pipeline.current_round,
                "evaluated_count": len(finite_evaluations),
                "dropped_nonfinite_count": len(dropped_evaluations),
                "evaluations": [
                    self.evaluation_debug_row(evaluation)
                    for evaluation in finite_evaluations
                ],
                "vault_analytics": vault.analytics.model_dump(mode="json"),
                "vault_items": [
                    {
                        **self.vault_item_debug_row(item),
                        "status_for_next_round": self.vault_item_status(
                            item,
                            self.pipeline.current_round + 1,
                        ),
                    }
                    for item in vault.top_failures
                ],
            },
        )
        self.advance_round()
        return finite_evaluations, ledger, vault
