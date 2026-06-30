from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any, Literal

from .dataschema import (
    CommandQuery,
    Evaluation_Output,
    Failure_VaultDB,
    Instructor_Recipe,
    Instructor_Slot,
    Pipeline_State,
    User_Example,
    validate_instructor_recipe_size,
)
from .evaluator import Evaluator
from .helper import (
    append_section_debug_log,
    choose_section,
    create_new_section,
    get_section_data_prompts_dir,
    get_section_data_responses_dir,
    get_section_dir,
    get_section_file_path,
    get_section_ledger_path,
    get_section_prompt_dir,
    get_section_state_path,
    get_section_vault_path,
    load_sections_index,
    load_pipeline_state,
    load_section_ledger,
    read_json_file,
    save_pipeline_recipe,
    save_pipeline_state,
    write_json_file,
    write_model_file,
)
from .helper import ensure_current_section
from .instructor import Instructor
from .policy import (
    DEFAULT_POLICY_TAGS,
    PolicyMaker,
    compose_generation_tag_plan,
    focused_generation_slot_count,
)
from .student import ADAPTER_COMPLETE_FILENAME, TrainyModel


EXAMPLE_FILENAME = "example.txt"
ExampleFormat = Literal["auto", "plain", "contextual"]
EXAMPLE_FORMATS = {"auto", "plain", "contextual"}
PLAIN_EXAMPLE_TEMPLATE = """You can add more examples by repeating the Sample/Output block.

Sample:
Write the source text here.

Output:
Write the expected output here.
"""
CONTEXTUAL_EXAMPLE_TEMPLATE = """You can add more examples by repeating the Instruction/Context/Output block.

Instruction:
Write the question or task here.

Context:
Write the source text or input context here.

Output:
Write the expected output here.
"""
EXAMPLE_HEADER_PATTERN = re.compile(
    r"^\s*(sample|instruction|instruct|question|context|input|output)\s*(?::\s*(.*))?\s*$",
    re.IGNORECASE,
)
ROUND_HEALTH_MIN_DELTA = 0.03
ROUND_HEALTH_PATIENCE = 2
ROUND_HEALTH_HISTORY_LIMIT = 50
ROLLBACK_LOOKBACK_ROUNDS = 3


@dataclass
class OrchestratorResult:
    section_id: str
    status: str
    message: str
    example_path: Path
    current_round: int
    batch_size: int
    adapter_path: Path | None = None


class Orchestrator:
    """
    Resumable end-to-end active-learning runner.

    It infers progress from saved files:
    - no usable example and no batch: create example.txt and stop
    - current_training_batch exists: train/evaluate that batch
    - no batch: create or resume a recipe, then build the batch
    """

    def __init__(
        self,
        *,
        section_id: str | None = None,
        student_model_id: str = "gemma-4-E2B-it",
        instructor_model_id: str = "gpt-5.4-mini",
        batch_size: int = 10,
        use_api_grouping: bool = True,
        interactive_section_selection: bool = False,
        example_format: ExampleFormat = "auto",
    ) -> None:
        batch_size = validate_instructor_recipe_size(batch_size)
        self.example_format = self.validate_example_format(example_format)

        if section_id is not None:
            self.section_id = choose_section(section_id)
        elif interactive_section_selection:
            self.section_id = self.select_section_interactively()
        else:
            self.section_id = ensure_current_section()

        self.student_model_id = student_model_id
        self.instructor_model_id = instructor_model_id
        self.batch_size = batch_size
        self.use_api_grouping = use_api_grouping
        self.last_round_health: dict[str, Any] | None = None

    @staticmethod
    def validate_example_format(example_format: str) -> ExampleFormat:
        normalized = example_format.strip().lower()
        if normalized not in EXAMPLE_FORMATS:
            allowed = ", ".join(sorted(EXAMPLE_FORMATS))
            raise ValueError(
                f"example_format must be one of {allowed}, got {example_format!r}."
            )
        return normalized  # type: ignore[return-value]

    @staticmethod
    def select_section_interactively() -> str:
        index = load_sections_index()
        existing_sections = [
            section_id
            for section_id in index.sections
            if get_section_dir(section_id).exists()
        ]

        if not existing_sections:
            section_id = create_new_section()
            print(f"Created first section: {section_id}")
            return section_id

        current_section_id = (
            index.current_section_id
            if index.current_section_id in existing_sections
            else existing_sections[-1]
        )

        print("\nAvailable sections:")
        for item_index, section_id in enumerate(existing_sections, start=1):
            active_marker = " (current)" if section_id == current_section_id else ""
            try:
                state = load_pipeline_state(section_id)
                summary = (
                    f"round={state.current_round}, "
                    f"batch={len(state.current_training_batch)}"
                )
            except Exception:
                summary = "state=unreadable"

            print(f"  {item_index}. {section_id}{active_marker} [{summary}]")

        prompt = (
            "\nPress Enter to continue current section, "
            "'n' for a new section, or enter a section number/id: "
        )

        while True:
            try:
                choice = input(prompt).strip()
            except EOFError:
                return choose_section(current_section_id)

            if not choice:
                return choose_section(current_section_id)

            if choice.lower() in {"n", "new"}:
                return create_new_section()

            if choice.isdigit():
                selected_index = int(choice)
                if 1 <= selected_index <= len(existing_sections):
                    return choose_section(existing_sections[selected_index - 1])

            if choice in existing_sections:
                return choose_section(choice)

            print("Unknown section choice. Try again.")

    def run(
        self,
        *,
        max_rounds: int = 1,
        build_only: bool = False,
        train: bool = True,
        evaluate: bool = True,
    ) -> OrchestratorResult:
        if max_rounds < 1:
            raise ValueError("max_rounds must be at least 1.")

        self.bootstrap_components()
        setup_result = self.prepare_example_state()
        if setup_result is not None:
            return setup_result

        completed_rounds = 0
        last_adapter_path: Path | None = None
        last_status = "ready"
        last_message = "Orchestrator is ready."

        while completed_rounds < max_rounds:
            state = load_pipeline_state(self.section_id)

            if not state.current_training_batch:
                recipe = self.next_or_active_recipe(state)
                self.build_training_batch(recipe)
                state = load_pipeline_state(self.section_id)
                last_status = "batch_ready"
                last_message = "Training batch is ready."

                if build_only:
                    break

            if not train and not evaluate:
                break

            trained_student: TrainyModel | None = None
            try:
                if train:
                    (
                        last_adapter_path,
                        trained_student,
                    ) = self.train_current_batch_with_model(keep_model=evaluate)
                    last_status = "trained"
                    last_message = "Training finished."

                if evaluate:
                    self.evaluate_current_batch(
                        adapter_path=last_adapter_path,
                        student=trained_student,
                        generate_answers=False,
                    )
                    completed_rounds += 1
                    last_status = "evaluated"
                    last_message = "Evaluation finished and the round advanced."
                else:
                    break
            finally:
                if trained_student is not None:
                    trained_student.release_runtime()

        final_state = load_pipeline_state(self.section_id)
        return OrchestratorResult(
            section_id=self.section_id,
            status=last_status,
            message=last_message,
            example_path=self.example_path(),
            current_round=final_state.current_round,
            batch_size=len(final_state.current_training_batch),
            adapter_path=last_adapter_path,
        )

    def bootstrap_components(self) -> None:
        section_dir = get_section_dir(self.section_id)
        section_dir.mkdir(parents=True, exist_ok=True)
        get_section_prompt_dir(self.section_id).mkdir(parents=True, exist_ok=True)
        get_section_data_prompts_dir(self.section_id).mkdir(
            parents=True,
            exist_ok=True,
        )
        get_section_data_responses_dir(self.section_id).mkdir(
            parents=True,
            exist_ok=True,
        )

        if not get_section_ledger_path(self.section_id).exists():
            write_json_file(get_section_ledger_path(self.section_id), [])

        if not get_section_vault_path(self.section_id).exists():
            write_model_file(
                get_section_vault_path(self.section_id),
                Failure_VaultDB(),
            )

        if not get_section_state_path(self.section_id).exists():
            write_model_file(
                get_section_state_path(self.section_id),
                Pipeline_State(
                    section_id=self.section_id,
                    example_format=self.example_format,
                ),
            )

    def prepare_example_state(self) -> OrchestratorResult | None:
        state = load_pipeline_state(self.section_id)
        if self.example_format != "auto" and state.example_format != self.example_format:
            state.example_format = self.example_format
            save_pipeline_state(state, self.section_id)

        active_example_format = self.active_example_format(state)
        example_path = self.example_path()

        if not example_path.exists():
            if state.user_examples:
                self.write_example_file(state.user_examples)
                return None

            example_path.write_text(
                self.example_template(active_example_format),
                encoding="utf-8",
            )
            if state.current_training_batch:
                return None
            return self.result(
                status="needs_example",
                message=f"Created {example_path}. Fill it, then run again.",
            )

        parsed_examples = self.load_example_file()
        if not parsed_examples:
            if state.current_training_batch:
                return None
            return self.result(
                status="needs_example",
                message=(
                    f"Fill {example_path} with examples matching "
                    f"example_format={active_example_format}."
                ),
            )

        new_examples = [
            example
            for example in parsed_examples
            if not self.state_has_example(state, example)
        ]
        if new_examples:
            state.user_examples.extend(new_examples)
            save_pipeline_state(state, self.section_id)

        return None

    def active_example_format(
        self,
        state: Pipeline_State | None = None,
    ) -> ExampleFormat:
        if self.example_format != "auto":
            return self.example_format

        active_state = state or load_pipeline_state(self.section_id)
        return self.validate_example_format(active_state.example_format)

    def example_template(self, example_format: ExampleFormat | None = None) -> str:
        active_format = example_format or self.active_example_format()
        if active_format == "contextual":
            return CONTEXTUAL_EXAMPLE_TEMPLATE
        return PLAIN_EXAMPLE_TEMPLATE

    def next_or_active_recipe(self, state: Pipeline_State) -> Instructor_Recipe:
        if state.active_recipe:
            return Instructor_Recipe(slots=state.active_recipe)

        if state.current_round == 0 and not load_section_ledger(self.section_id):
            recipe = self.full_generation_recipe()
            save_pipeline_recipe(recipe, self.section_id)
            append_section_debug_log(
                self.section_id,
                "policy_recipe_decision",
                {
                    "round_id": state.current_round,
                    "reason": "first_round_full_generation",
                    "recipe_slots": [
                        slot.model_dump(mode="json") for slot in recipe.slots
                    ],
                    "selected_recycle_ids": [],
                    "vault_total_items": 0,
                    "vault_usable_items": 0,
                    "vault_exhausted_items": 0,
                    "vault_cooling_down_items": 0,
                    "vault_retired_items": 0,
                },
            )
            return recipe

        return PolicyMaker(
            self.section_id,
            batch_size=self.batch_size,
            model_id=self.instructor_model_id,
            use_api_grouping=self.use_api_grouping,
        ).InstructionRecipe(save=True)

    def full_generation_recipe(self) -> Instructor_Recipe:
        tags = self.first_round_tags()
        slots = [
            Instructor_Slot(
                slot_id=index + 1,
                command=CommandQuery.GENERATE,
                tag=tags[index],
            )
            for index in range(self.batch_size)
        ]
        return Instructor_Recipe(slots=slots)

    def first_round_tags(self) -> list[str]:
        state = load_pipeline_state(self.section_id)
        state_tags = [
            tag
            for tag, _count in sorted(
                state.learning_tag_counts.items(),
                key=lambda item: item[1],
                reverse=True,
            )
            if tag != "good"
        ]
        tag_pool = state_tags or list(DEFAULT_POLICY_TAGS)
        focus_slots = focused_generation_slot_count(self.batch_size)
        focused_tags = [
            tag_pool[index % len(tag_pool)]
            for index in range(focus_slots)
        ] if tag_pool else []
        return compose_generation_tag_plan(focused_tags, self.batch_size)

    def build_training_batch(self, recipe: Instructor_Recipe) -> None:
        runner = Instructor(model_id=self.instructor_model_id)
        runner.receive_data(recipe)

    def train_current_batch(self) -> Path:
        adapter_path, student = self.train_current_batch_with_model(keep_model=False)
        student.release_runtime()
        return adapter_path

    def train_current_batch_with_model(
        self,
        *,
        keep_model: bool = True,
    ) -> tuple[Path, TrainyModel]:
        student = TrainyModel(self.student_model_id)
        if student.adapter_is_complete(student.output_dir):
            append_section_debug_log(
                self.section_id,
                "training_skipped_existing_adapter",
                {
                    "round_id": student.current_round,
                    "output_adapter_path": str(student.output_dir),
                    "reason": "adapter_already_complete",
                },
            )
            if keep_model:
                student.load_trained_model(student.output_dir)
            return student.output_dir, student

        if student.adapter_has_partial_artifacts(student.output_dir):
            print(
                "Existing adapter artifacts are incomplete; "
                "training this round again."
            )

        adapter_path = student.train(keep_model=keep_model)
        return adapter_path, student

    def evaluate_current_batch(
        self,
        adapter_path: Path | None = None,
        validation_loss: float | None = None,
        validation_source: str | None = None,
        student: TrainyModel | None = None,
        generate_answers: bool = False,
    ) -> tuple[list[Evaluation_Output], list[Any], Failure_VaultDB]:
        state_before_eval = load_pipeline_state(self.section_id)
        round_id = state_before_eval.current_round
        evaluations, ledger, vault = Evaluator(
            self.student_model_id,
            student=student,
            generate_answers=generate_answers,
        ).run()
        self.last_round_health = self.record_round_health(
            round_id=round_id,
            evaluations=evaluations,
            adapter_path=adapter_path,
            validation_loss=validation_loss,
            validation_source=validation_source,
        )
        return evaluations, ledger, vault

    @staticmethod
    def adapter_path_for_round(
        round_id: int,
        *,
        reference_adapter_path: Path | str | None,
    ) -> Path | None:
        if reference_adapter_path is None:
            return None
        return Path(reference_adapter_path).parent / f"round_{round_id:03d}"

    @staticmethod
    def adapter_path_is_complete(adapter_path: Path | None) -> bool:
        if adapter_path is None:
            return False
        return (
            (adapter_path / "adapter_config.json").exists()
            and (adapter_path / ADAPTER_COMPLETE_FILENAME).exists()
        )

    def recent_rollback_target(
        self,
        *,
        state: Pipeline_State,
        current_round_id: int,
        current_adapter_path: Path | None,
    ) -> dict[str, Any]:
        min_round_id = max(0, current_round_id - ROLLBACK_LOOKBACK_ROUNDS)
        reference_adapter_path = current_adapter_path or state.best_adapter_path
        candidates: list[dict[str, Any]] = []

        for event in state.round_health_history:
            try:
                candidate_round_id = int(event.get("round_id"))
            except (TypeError, ValueError):
                continue

            if not (min_round_id <= candidate_round_id < current_round_id):
                continue

            candidate_adapter_path = event.get("adapter_path") or event.get(
                "produced_adapter_path"
            )
            if candidate_adapter_path is None:
                candidate_adapter = self.adapter_path_for_round(
                    candidate_round_id,
                    reference_adapter_path=reference_adapter_path,
                )
            else:
                candidate_adapter = Path(candidate_adapter_path)

            candidate_loss = event.get("decision_loss", event.get("avg_loss"))
            try:
                normalized_loss = (
                    float(candidate_loss) if candidate_loss is not None else None
                )
            except (TypeError, ValueError):
                normalized_loss = None

            candidates.append(
                {
                    "round_id": candidate_round_id,
                    "adapter_path": str(candidate_adapter) if candidate_adapter else None,
                    "decision_loss": normalized_loss,
                    "action": event.get("action"),
                    "complete": self.adapter_path_is_complete(candidate_adapter),
                }
            )

        scored_candidates = [
            candidate
            for candidate in candidates
            if candidate["complete"] and candidate["decision_loss"] is not None
        ]
        fallback_candidates = [
            candidate
            for candidate in candidates
            if candidate["complete"]
        ]

        target = None
        if scored_candidates:
            target = sorted(
                scored_candidates,
                key=lambda candidate: (
                    float(candidate["decision_loss"]),
                    -int(candidate["round_id"]),
                ),
            )[0]
        elif fallback_candidates:
            target = sorted(
                fallback_candidates,
                key=lambda candidate: int(candidate["round_id"]),
                reverse=True,
            )[0]

        return {
            "lookback_rounds": ROLLBACK_LOOKBACK_ROUNDS,
            "eligible_round_range": [min_round_id, max(min_round_id, current_round_id - 1)],
            "candidates": candidates,
            "target": target,
        }

    def record_round_health(
        self,
        *,
        round_id: int,
        evaluations: list[Evaluation_Output],
        adapter_path: Path | None,
        validation_loss: float | None = None,
        validation_source: str | None = None,
    ) -> dict[str, Any]:
        losses = [
            float(evaluation.loss)
            for evaluation in evaluations
            if evaluation.loss is not None
        ]
        if not losses:
            return {}

        training_avg_loss = sum(losses) / len(losses)
        decision_loss = (
            float(validation_loss)
            if validation_loss is not None
            else training_avg_loss
        )
        decision_source = validation_source or "training_batch"
        state = load_pipeline_state(self.section_id)
        previous_best_loss = state.best_eval_loss
        previous_raw_best_loss = state.raw_best_eval_loss
        rollback_decision: dict[str, Any] | None = None
        action = "watch"

        if previous_best_loss is None:
            materially_better = True
            materially_worse = False
        else:
            materially_better = decision_loss < (
                previous_best_loss * (1.0 - ROUND_HEALTH_MIN_DELTA)
            )
            materially_worse = decision_loss > (
                previous_best_loss * (1.0 + ROUND_HEALTH_MIN_DELTA)
            )

        if previous_raw_best_loss is None or decision_loss < previous_raw_best_loss:
            state.raw_best_eval_loss = decision_loss
            state.raw_best_round_id = round_id
            state.raw_best_adapter_path = str(adapter_path) if adapter_path else None

        if materially_better:
            state.best_eval_loss = decision_loss
            state.best_round_id = round_id
            state.best_adapter_path = str(adapter_path) if adapter_path else None
            state.bad_round_streak = 0
            state.base_adapter_override_path = None
            action = "new_best"
        elif materially_worse:
            state.bad_round_streak += 1
            action = "degraded"
            if (
                state.bad_round_streak >= ROUND_HEALTH_PATIENCE
            ):
                rollback_decision = self.recent_rollback_target(
                    state=state,
                    current_round_id=round_id,
                    current_adapter_path=adapter_path,
                )
                rollback_target = rollback_decision.get("target")
                if rollback_target and rollback_target.get("adapter_path"):
                    state.base_adapter_override_path = str(
                        rollback_target["adapter_path"]
                    )
                    state.last_rollback_decision = rollback_decision
                    state.bad_round_streak = 0
                    action = "rollback_next"
                else:
                    state.last_rollback_decision = rollback_decision
                    action = "rollback_unavailable"
        else:
            state.bad_round_streak = 0
            action = "stable"

        health_event = {
            "round_id": round_id,
            "adapter_path": str(adapter_path) if adapter_path else None,
            "produced_adapter_path": str(adapter_path) if adapter_path else None,
            "avg_loss": decision_loss,
            "training_avg_loss": training_avg_loss,
            "validation_loss": validation_loss,
            "decision_loss": decision_loss,
            "decision_source": decision_source,
            "previous_best_loss": previous_best_loss,
            "previous_raw_best_loss": previous_raw_best_loss,
            "best_round_id": state.best_round_id,
            "best_eval_loss": state.best_eval_loss,
            "best_adapter_path": state.best_adapter_path,
            "raw_best_round_id": state.raw_best_round_id,
            "raw_best_eval_loss": state.raw_best_eval_loss,
            "raw_best_adapter_path": state.raw_best_adapter_path,
            "bad_round_streak": state.bad_round_streak,
            "base_adapter_override_path": state.base_adapter_override_path,
            "rollback_decision": rollback_decision,
            "action": action,
            "min_delta": ROUND_HEALTH_MIN_DELTA,
            "patience": ROUND_HEALTH_PATIENCE,
            "rollback_lookback_rounds": ROLLBACK_LOOKBACK_ROUNDS,
        }
        state.round_health_history.append(health_event)
        state.round_health_history = state.round_health_history[
            -ROUND_HEALTH_HISTORY_LIMIT:
        ]
        save_pipeline_state(state, self.section_id)
        append_section_debug_log(
            self.section_id,
            "round_health_decision",
            health_event,
        )
        return health_event

    def example_path(self) -> Path:
        return get_section_file_path(self.section_id, EXAMPLE_FILENAME)

    def load_example_file(self) -> list[User_Example]:
        text = self.example_path().read_text(encoding="utf-8").strip()
        if not text or self.example_file_has_template_text(text):
            return []

        example_format = self.active_example_format()
        parsed_json = self.try_parse_json_examples(text, example_format)
        if parsed_json:
            return parsed_json

        return self.parse_text_examples(text, example_format)

    @staticmethod
    def example_file_has_template_text(text: str) -> bool:
        return any(
            marker in text
            for marker in (
                "Write the source text here.",
                "Write the question or task here.",
                "Write the source text or input context here.",
                "Write the expected output here.",
            )
        )

    @staticmethod
    def normalize_example_header(header: str) -> str:
        normalized = header.strip().lower()
        if normalized in {"instruction", "instruct", "question"}:
            return "instruction"
        if normalized in {"context", "input"}:
            return "context"
        return normalized

    @staticmethod
    def parse_text_examples(
        text: str,
        example_format: ExampleFormat = "auto",
    ) -> list[User_Example]:
        example_format = Orchestrator.validate_example_format(example_format)
        examples: list[User_Example] = []
        current_section: str | None = None
        section_lines: dict[str, list[str]] = {
            "sample": [],
            "instruction": [],
            "context": [],
            "output": [],
        }

        def flush_current() -> bool:
            example = Orchestrator.example_from_sections(
                section_lines,
                example_format,
            )
            if example is None and not Orchestrator.sections_have_text(section_lines):
                return True
            if example is None:
                return False
            examples.append(example)
            return True

        for line in text.splitlines():
            marker_match = EXAMPLE_HEADER_PATTERN.match(line)
            if marker_match:
                marker = Orchestrator.normalize_example_header(marker_match.group(1))
                if example_format == "plain" and marker not in {"sample", "output"}:
                    marker_match = None

            if marker_match:
                inline_text = (marker_match.group(2) or "").rstrip()

                if marker in {"sample", "instruction"} and Orchestrator.sections_have_text(section_lines):
                    if section_lines["output"]:
                        if not flush_current():
                            return []
                        section_lines = {
                            "sample": [],
                            "instruction": [],
                            "context": [],
                            "output": [],
                        }

                current_section = marker
                if inline_text:
                    section_lines[current_section].append(inline_text)
                continue

            if current_section is not None:
                section_lines[current_section].append(line)

        if not flush_current():
            return []

        return examples

    @staticmethod
    def sections_have_text(section_lines: dict[str, list[str]]) -> bool:
        return any(
            "\n".join(lines).strip()
            for lines in section_lines.values()
        )

    @staticmethod
    def example_from_sections(
        section_lines: dict[str, list[str]],
        example_format: ExampleFormat,
    ) -> User_Example | None:
        sample = "\n".join(section_lines["sample"]).strip()
        instruction = "\n".join(section_lines["instruction"]).strip()
        context = "\n".join(section_lines["context"]).strip()
        output = "\n".join(section_lines["output"]).strip()

        if not output:
            return None

        if example_format == "plain":
            if not sample:
                return None
            return User_Example(sample=sample, output=output, mode="plain")

        if instruction and context:
            return User_Example(
                sample=context,
                instruct=instruction,
                context=context,
                output=output,
                mode="contextual",
            )

        if sample and context:
            return User_Example(
                sample=context,
                instruct=sample,
                context=context,
                output=output,
                mode="contextual",
            )

        if example_format == "contextual":
            return None

        if sample:
            return User_Example(sample=sample, output=output, mode="plain")

        return None

    def try_parse_json_examples(
        self,
        text: str,
        example_format: ExampleFormat = "auto",
    ) -> list[User_Example]:
        example_format = self.validate_example_format(example_format)
        try:
            raw_examples = read_json_file(self.example_path(), {})
        except Exception:
            return []

        if isinstance(raw_examples, dict) and isinstance(raw_examples.get("examples"), list):
            candidate_examples = raw_examples["examples"]
        elif isinstance(raw_examples, dict):
            candidate_examples = [raw_examples]
        elif isinstance(raw_examples, list):
            candidate_examples = raw_examples
        else:
            return []

        examples: list[User_Example] = []
        for raw_example in candidate_examples:
            if not isinstance(raw_example, dict):
                return []

            example = self.example_from_json(raw_example, example_format)
            if example is None:
                return []

            examples.append(example)

        return examples

    @staticmethod
    def example_from_json(
        raw_example: dict,
        example_format: ExampleFormat,
    ) -> User_Example | None:
        output = str(
            raw_example.get("output")
            or raw_example.get("gold_summary")
            or raw_example.get("summary")
            or ""
        ).strip()
        if not output:
            return None

        sample = str(raw_example.get("sample") or "").strip()
        instruction = str(
            raw_example.get("instruction")
            or raw_example.get("instruct")
            or raw_example.get("question")
            or ""
        ).strip()
        context = str(
            raw_example.get("context")
            or raw_example.get("input")
            or ""
        ).strip()

        if example_format == "plain":
            if not sample:
                return None
            return User_Example(sample=sample, output=output, mode="plain")

        if instruction and not context and sample:
            context = sample
        if context and not instruction and sample:
            instruction = sample

        if instruction and context:
            return User_Example(
                sample=context,
                instruct=instruction,
                context=context,
                output=output,
                mode="contextual",
            )

        if example_format == "contextual":
            return None

        if sample:
            return User_Example(sample=sample, output=output, mode="plain")

        return None

    def write_example_file(self, examples: User_Example | list[User_Example]) -> None:
        if isinstance(examples, User_Example):
            examples = [examples]
        rendered_examples = "\n\n".join(
            self.render_example_file_block(example)
            for example in examples
        )
        self.example_path().write_text(
            f"{rendered_examples}\n",
            encoding="utf-8",
        )

    @staticmethod
    def render_example_file_block(example: User_Example) -> str:
        if example.mode == "contextual" or example.instruct or example.context:
            instruction = example.instruct or ""
            context = example.context or example.sample
            return (
                f"Instruction:\n{instruction}\n\n"
                f"Context:\n{context}\n\n"
                f"Output:\n{example.output}"
            )

        return f"Sample:\n{example.sample}\n\nOutput:\n{example.output}"

    def state_has_example(
        self,
        state: Pipeline_State,
        example: User_Example,
    ) -> bool:
        return any(
            existing.sample == example.sample
            and existing.output == example.output
            and existing.instruct == example.instruct
            and existing.context == example.context
            and existing.mode == example.mode
            for existing in state.user_examples
        )

    def result(self, status: str, message: str) -> OrchestratorResult:
        state = load_pipeline_state(self.section_id)
        return OrchestratorResult(
            section_id=self.section_id,
            status=status,
            message=message,
            example_path=self.example_path(),
            current_round=state.current_round,
            batch_size=len(state.current_training_batch),
        )


def run_orchestrator(**kwargs) -> OrchestratorResult:
    return Orchestrator().run(**kwargs)
