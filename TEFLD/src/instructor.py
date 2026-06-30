from __future__ import annotations

import json
import os
import random
import re
from collections import Counter
from difflib import get_close_matches
from typing import Any

from .dataschema import (
    Orchestrator_Output,
    Instructor_Slot,
    Instructor_Recipe,
    Training_Sample,
    learning_record,
    Pipeline_State,
    User_Example,
    CommandQuery
)
from .diversity_planner import (
    DiversityPlanner,
    GenerationBlueprint,
    base_generation_tag,
    is_hard_generation_tag,
    split_generation_intent,
)

from .helper import (
    DATA_DIR,
    append_section_debug_log,
    ensure_current_section,
    ensure_user_example,
    format_learning_tag_counts,
    get_cached_instruction,
    get_cached_instruction_response,
    get_learning_tag_counts,
    get_section_file_path,
    get_user_examples,
    load_section_ledger,
    load_section_vault,
    load_prompt_template,
    render_prompt_template,
    save_data_prompt,
    save_data_response,
    save_cached_instruction_response,
    save_generated_instruction,
    save_instruction_prompt,
    save_pipeline_recipe,
    save_pipeline_state,
    save_pipeline_training_batch,
    sync_learning_tag_counts_from_ledger,
    update_learning_tag_count,
    write_json_file,
    call_client,
    create_client
)

try:
    from dotenv import load_dotenv
except ModuleNotFoundError:

    def load_dotenv(*_args: Any, **_kwargs: Any) -> bool:
        return False

load_dotenv(DATA_DIR / ".env")
load_dotenv()

JSON_FENCE_PATTERN = re.compile(r"^```(?:json)?\s*(.*?)\s*```$", re.IGNORECASE | re.DOTALL)
INSTRUCTION_CACHE_VERSION = "dynamic_tags_v5"
WIDE_VARIETY_TAG = "wide_variety"
META_INSTRUCTION_MARKERS = (
    "generate one synthetic training sample",
    "you generate",
    "the downstream data generator",
    "use the user's example as a signal",
    "the sample should be broad",
)
NON_CATEGORY_TAGS = {"good", "wide_variety"}
TAG_SIMILARITY_CUTOFF = 0.92
WIDE_VARIETY_MAX_ATTEMPTS = 2
MAX_WIDE_VARIETY_TAGS_PER_ROUND = 1
WIDE_VARIETY_SECTION_SHARE_LIMIT = 0.35
WIDE_VARIETY_SECTION_COUNT_LIMIT = 3
CONTEXTUAL_QA_TAGS = {
    "contextual_qa",
    "contextual_factual_qa",
    "factual_qa",
    "contextual_question_answering",
    "open_book_qa",
    "question_answering",
    "reading_comprehension",
    "information_extraction",
}
CONTEXTUAL_QA_BANNED_TAGS = {
    "recommendation",
    "email_rewriting",
    "text_rewriting",
    "meeting_summarization",
    "customer_support_summarization",
    "recipe_rewriting",
    "faq_generation",
}


def normalize_learning_tag(tag: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "_", tag.strip().lower())
    normalized = re.sub(r"_+", "_", normalized).strip("_")
    return normalized


def known_category_counts(tag_counts: dict[str, int]) -> dict[str, int]:
    categories: dict[str, int] = {}
    for tag, count in tag_counts.items():
        normalized_tag = normalize_learning_tag(tag)
        if not normalized_tag or normalized_tag in NON_CATEGORY_TAGS:
            continue
        categories[normalized_tag] = categories.get(normalized_tag, 0) + count
    return categories


def format_known_categories(tag_counts: dict[str, int]) -> str:
    categories = known_category_counts(tag_counts)
    if not categories:
        return (
            "No known categories yet. Create a clear new snake_case tag for "
            "the generated sample."
        )

    return "\n".join(
        f"- {tag} ({count})"
        for tag, count in sorted(categories.items(), key=lambda item: (-item[1], item[0]))
    )


def canonicalize_learning_tag(raw_tag: str, tag_counts: dict[str, int]) -> str:
    candidate = normalize_learning_tag(base_generation_tag(raw_tag))
    if candidate.startswith("hard_"):
        candidate = candidate[len("hard_") :]
    if not candidate:
        return "uncategorized"
    if candidate in NON_CATEGORY_TAGS or candidate.startswith(f"{WIDE_VARIETY_TAG}_"):
        return "uncategorized"

    categories = known_category_counts(tag_counts)
    if candidate in categories:
        return candidate

    close_matches = get_close_matches(
        candidate,
        categories.keys(),
        n=1,
        cutoff=TAG_SIMILARITY_CUTOFF,
    )
    if close_matches:
        return close_matches[0]

    return candidate


def is_wide_variety_tag(tag: str) -> bool:
    normalized = normalize_learning_tag(base_generation_tag(tag))
    return normalized == WIDE_VARIETY_TAG or normalized.startswith(f"{WIDE_VARIETY_TAG}_")


def looks_like_meta_instruction(text: str) -> bool:
    normalized = re.sub(r"\s+", " ", text or "").strip().lower()
    return any(marker in normalized for marker in META_INSTRUCTION_MARKERS)


def format_user_example_for_prompt(example: User_Example) -> str:
    if example.mode == "contextual" or example.instruct or example.context:
        instruction = example.instruct or "Use the context to produce the output."
        context = example.context or example.sample
        return (
            f"Instruction:\n{instruction}\n\n"
            f"Context:\n{context}\n\n"
            f"Output:\n{example.output}"
        )

    return f"Input:\n{example.sample}\n\nOutput:\n{example.output}"


class Instructor:
    def __init__(
        self,
        model_id: str = "gpt-5.4-mini",
        orchestrator_orders: Orchestrator_Output | None = None,
        **legacy_kwargs: Any,
    ) -> None:
        if "modelid" in legacy_kwargs:
            model_id = legacy_kwargs.pop("modelid")

        if "orchester_oders" in legacy_kwargs:
            if orchestrator_orders is not None:
                raise TypeError(
                    "Pass only one of orchestrator_orders or orchester_oders."
                )
            orchestrator_orders = legacy_kwargs.pop("orchester_oders")

        if legacy_kwargs:
            unexpected = ", ".join(sorted(legacy_kwargs))
            raise TypeError(f"Unexpected Instructor argument(s): {unexpected}")

        self.Oders: Orchestrator_Output | None = orchestrator_orders
        self.Recipe_list:Instructor_Recipe | None = (
            self.Oders.recipe if self.Oders is not None else None
        )

        self.list_of_recycle: list[Instructor_Slot] = []
        self.list_of_generating: list[Instructor_Slot] = []
        self.SampleList: list[Training_Sample] = []

        self.fail_vault: dict = {}
        self.learning_data: list[learning_record] = []

        self.pipeline: Pipeline_State = Pipeline_State()
        self.examples: list[User_Example] = []
        self.section_id: str = ""

        self.section_id = ensure_current_section()
        self.fail_vault = load_section_vault(self.section_id)
        self.pipeline = sync_learning_tag_counts_from_ledger(self.section_id)
        
        self.examples = get_user_examples(self.section_id)
        self.learning_data = load_section_ledger(self.section_id)
        self.model_id = model_id

    @staticmethod
    def preview_text(text: str | None, limit: int = 240) -> str:
        return re.sub(r"\s+", " ", text or "").strip()[:limit]

    def sample_debug_row(self, sample: Training_Sample) -> dict[str, Any]:
        return {
            "source": sample.source,
            "source_ledger_id": sample.source_ledger_id,
            "tag": sample.tag,
            "instruct_preview": self.preview_text(sample.instruct),
            "sample_preview": self.preview_text(sample.sample),
            "gold_preview": self.preview_text(sample.gold_summary),
        }

    def receive_data(self, recipe_received: Instructor_Recipe) -> None:
        
        if not recipe_received.slots:
            print("there is no data!")
            return

        self.Recipe_list = recipe_received
        self.pipeline = save_pipeline_recipe(recipe_received, self.section_id)

        self.list_of_recycle = [slot for slot in recipe_received.slots if slot.command == CommandQuery.RECYCLE]

        self.list_of_generating = [slot for slot in recipe_received.slots if slot.command == CommandQuery.GENERATE]

        print(
            "[instructor] recipe split "
            f"recycle={len(self.list_of_recycle)} "
            f"generate={len(self.list_of_generating)}"
        )
        self.SampleList = self.extracting_recycles(self.list_of_recycle)
        generated_list = self.generation(recipe_received)

        if generated_list:
            self.SampleList.extend(generated_list)

        random.shuffle(self.SampleList)
        self.validate_batch_for_student(self.SampleList)
        self.pipeline = save_pipeline_training_batch(
            training_batch=self.SampleList,
            recipe=recipe_received,
            section_id=self.section_id,
        )
        append_section_debug_log(
            self.section_id,
            "training_batch_ready",
            {
                "round_id": self.pipeline.current_round,
                "recipe_slots": [
                    slot.model_dump(mode="json") for slot in recipe_received.slots
                ],
                "sample_count": len(self.SampleList),
                "source_counts": dict(
                    Counter(sample.source for sample in self.SampleList)
                ),
                "tag_counts": dict(Counter(sample.tag for sample in self.SampleList)),
                "samples": [
                    self.sample_debug_row(sample) for sample in self.SampleList
                ],
            },
        )

    def extracting_recycles(self, list_recycles: list[Instructor_Slot],) -> list[Training_Sample]:
        
        ledger_by_id = {row.ledger_id: row for row in self.learning_data}

        recycled_samples: list[Training_Sample] = []

        for slot in list_recycles:
            if slot.failure_ledger_id is None:
                raise ValueError("Recycle slot is missing failure_ledger_id.")

            extracted_row = ledger_by_id.get(slot.failure_ledger_id)

            if extracted_row is None:
                raise ValueError(
                    f"No ledger record found for ledger_id={slot.failure_ledger_id}"
                )

            sample = Training_Sample(
                sample=extracted_row.sample,
                instruct=extracted_row.instruct,
                gold_summary=extracted_row.gold_summary,
                source="recycled",
                source_ledger_id=extracted_row.ledger_id,
                tag=extracted_row.tag,
            )

            recycled_samples.append(sample)

        recycled_ids = [
            sample.source_ledger_id
            for sample in recycled_samples
            if sample.source_ledger_id is not None
        ]
        if recycled_ids:
            print(f"[instructor] recycled ledger_ids={recycled_ids}")
        return recycled_samples

    def validate_batch_for_student(self, batch: list[Training_Sample]) -> None:
        if not batch:
            raise ValueError("Pipeline training batch is empty.")

        for index, sample in enumerate(batch, start=1):
            if not sample.sample.strip():
                raise ValueError(f"Training sample {index} is missing sample text.")

            if not sample.instruct.strip():
                raise ValueError(f"Training sample {index} is missing instruction.")

            if not sample.gold_summary or not sample.gold_summary.strip():
                raise ValueError(f"Training sample {index} is missing gold_summary.")

            if not sample.tag.strip():
                raise ValueError(f"Training sample {index} is missing tag.")


    def build_prompt(
        self,
        prompt_type: str,
        example: User_Example | str,
        slot: Instructor_Slot,
        generation_constraints: str | None = None,
    ) -> str:
        raw_tag_counts = get_learning_tag_counts(self.section_id)
        tag_counts = format_learning_tag_counts(raw_tag_counts)
        known_categories = format_known_categories(raw_tag_counts)
        constraints = generation_constraints or "No extra constraints."
        prompt_tag = base_generation_tag(slot.tag)

        if prompt_type.lower() == "instruct":
            if not isinstance(example, User_Example):
                raise TypeError("Instruction prompt requires a User_Example.")

            template = load_prompt_template("instruction_prompt.txt")
            placeholders_items = {
                "example_block": format_user_example_for_prompt(example),
                "tag": prompt_tag,
                "tag_counts": tag_counts,
                "known_categories": known_categories,
                "generation_constraints": constraints,
            }
        elif prompt_type.lower() == "data":
            template = load_prompt_template("data_prompt.txt")
            placeholders_items = {
                "instruct": example,
                "tag": prompt_tag,
                "tag_counts": tag_counts,
                "known_categories": known_categories,
                "generation_constraints": constraints,
            }
        else:
            raise ValueError(f"Unknown prompt type: {prompt_type}")

        print(f"[prompt] rendered type={prompt_type.lower()} tag={slot.tag}")
        return render_prompt_template(
            template,
            placeholders_items
        )

    def parse_generated_sample(
        self,
        response: str,
        instruction: str,
        tag: str,
        tag_counts: dict[str, int] | None = None,
    ) -> Training_Sample:
        raw_sample = self.parse_generated_json(response)
        return self.training_sample_from_generated_object(
            raw_sample=raw_sample,
            instruction=instruction,
            tag=tag,
            tag_counts=tag_counts,
        )

    def training_sample_from_generated_object(
        self,
        raw_sample: Any,
        instruction: str,
        tag: str,
        tag_counts: dict[str, int] | None = None,
    ) -> Training_Sample:
        if not isinstance(raw_sample, dict):
            raise ValueError("Instructor response JSON must be an object.")

        if raw_sample.get("sample") is None:
            raise ValueError("Instructor response JSON is missing 'sample'.")

        raw_sample_text = str(raw_sample["sample"]).strip()
        raw_instruct = str(raw_sample.get("instruct") or "").strip()
        if not raw_instruct:
            raise ValueError("Instructor response JSON is missing non-empty 'instruct'.")
        if looks_like_meta_instruction(raw_instruct):
            raise ValueError(
                "Instructor response used the generation meta-instruction as "
                "the student-facing instruct."
            )
        if looks_like_meta_instruction(raw_sample_text):
            raise ValueError(
                "Instructor response used the generation meta-instruction as "
                "the student-facing sample."
            )

        gold_summary = (
            raw_sample.get("gold_summary")
            or raw_sample.get("summary")
            or raw_sample.get("output")
        )

        raw_tag = str(raw_sample.get("tag") or tag)
        canonical_tag = canonicalize_learning_tag(raw_tag, tag_counts or {})

        sample_preview = str(raw_sample["sample"]).replace("\n", " ")[:90]
        print(
            "[instructor] parsed generated sample "
            f"tag={canonical_tag} sample='{sample_preview}'"
        )
        return Training_Sample(
            sample=raw_sample_text,
            instruct=raw_instruct,
            gold_summary=str(gold_summary) if gold_summary is not None else None,
            source="generated",
            tag=canonical_tag,
        )

    def parse_generated_json(self, response: str) -> Any:
        payload = self.strip_json_fences(response)

        try:
            return json.loads(payload)
        except json.JSONDecodeError as exc:
            extracted_payload = self.extract_json_payload(payload)
            if extracted_payload is not None:
                try:
                    return json.loads(extracted_payload)
                except json.JSONDecodeError:
                    pass

            preview = payload[:500].replace("\n", "\\n")
            raise ValueError(
                "Instructor response was not valid JSON. "
                f"Response preview: {preview}"
            ) from exc

    def strip_json_fences(self, response: str) -> str:
        payload = response.strip()
        fenced_match = JSON_FENCE_PATTERN.match(payload)
        if fenced_match:
            return fenced_match.group(1).strip()

        lines = payload.splitlines()
        if lines and lines[0].strip().startswith("```"):
            lines = lines[1:]
            if lines and lines[-1].strip().startswith("```"):
                lines = lines[:-1]
            return "\n".join(lines).strip()

        return payload

    def extract_json_payload(self, payload: str) -> str | None:
        object_start = payload.find("{")
        array_start = payload.find("[")
        starts = [
            index
            for index in (object_start, array_start)
            if index != -1
        ]
        if not starts:
            return None

        start = min(starts)
        closing = "}" if payload[start] == "{" else "]"
        end = payload.rfind(closing)
        if end == -1 or end <= start:
            return None
        return payload[start : end + 1]

    def extract_json_object(self, payload: str) -> str | None:
        start = payload.find("{")
        end = payload.rfind("}")
        if start == -1 or end == -1 or end <= start:
            return None
        return payload[start : end + 1]

    def instruction_cache_tag(self, slot: Instructor_Slot, attempt: int = 1) -> str:
        generation_intent, _base_tag = split_generation_intent(slot.tag)
        tag = normalize_learning_tag(base_generation_tag(slot.tag)) or "untagged"
        if generation_intent == "hard":
            tag = f"hard_{tag}"
        elif generation_intent == "validation":
            tag = f"validation_{tag}"
        if is_wide_variety_tag(slot.tag):
            return (
                f"{tag}_round_{self.pipeline.current_round:03d}"
                f"_slot_{slot.slot_id:02d}_attempt_{attempt:02d}"
                f"__{INSTRUCTION_CACHE_VERSION}"
            )
        return f"{tag}__{INSTRUCTION_CACHE_VERSION}"

    def contextual_examples(self) -> list[User_Example]:
        return [
            example
            for example in self.pipeline.user_examples
            if example.mode == "contextual"
            and (example.context or example.sample)
            and example.instruct
        ]

    def contextual_qa_slot_id(
        self,
        slots: list[Instructor_Slot],
    ) -> int | None:
        if not self.contextual_examples():
            return None

        for slot in slots:
            if is_wide_variety_tag(slot.tag):
                return slot.slot_id

        return None

    @staticmethod
    def combine_generation_constraints(*parts: str | None) -> str | None:
        cleaned_parts = [
            part.strip()
            for part in parts
            if part and part.strip() and part.strip() != "No extra constraints."
        ]
        if not cleaned_parts:
            return None
        return "\n\n".join(cleaned_parts)

    def planned_generation_constraints(
        self,
        slot: Instructor_Slot,
        *,
        generation_index: int,
        round_tag_counts: Counter[str],
        rejected_attempts: list[dict[str, str]] | None = None,
        force_contextual_qa: bool = False,
        attempt: int = 1,
    ) -> tuple[GenerationBlueprint, str]:
        blueprint = DiversityPlanner(
            pipeline=self.pipeline,
            ledger=self.learning_data,
        ).plan(
            slot=slot,
            generation_index=generation_index,
            round_tag_counts=round_tag_counts,
            force_contextual_qa=force_contextual_qa,
            attempt=attempt,
        )
        exploration_constraints = (
            self.diversity_constraints(
                slot,
                round_tag_counts,
                rejected_attempts,
                force_contextual_qa=force_contextual_qa,
            )
            if is_wide_variety_tag(slot.tag)
            else None
        )
        constraints = self.combine_generation_constraints(
            blueprint.as_constraints(),
            exploration_constraints,
        )
        return blueprint, constraints or "No extra constraints."

    def save_planner_snapshot(self, generation_items: list[dict[str, Any]]) -> None:
        planner = DiversityPlanner(
            pipeline=self.pipeline,
            ledger=self.learning_data,
        )
        mode_payload = planner.planner_state_payload()
        payload = {
            "section_id": self.section_id,
            "round_id": self.pipeline.current_round,
            **mode_payload,
            "learning_pressure": self.pipeline.learning_pressure_state.get("current", {}),
            "validation_weakness_profile": (
                self.pipeline.validation_weakness_profile or {}
            ),
            "generation_slots": [
                {
                    "slot_id": item["slot_id"],
                    "curriculum_focus": item["tag"],
                    "generation_intent": item["blueprint"].generation_intent,
                    "task_shape_profile": item["blueprint"].task_shape_profile,
                    "force_contextual_qa": item["force_contextual_qa"],
                    "blueprint": item["blueprint"].as_payload(),
                }
                for item in generation_items
            ],
        }

        self.pipeline.resolved_task_mode = payload["resolved_task_mode"]
        self.pipeline.task_shape_profile = payload["task_shape_profile"]
        self.pipeline.planner_state = payload
        self.pipeline.planner_history.append(payload)
        self.pipeline.planner_history = self.pipeline.planner_history[-50:]
        save_pipeline_state(self.pipeline, self.section_id)
        write_json_file(
            get_section_file_path(self.section_id, "planner_state.json"),
            payload,
        )
        append_section_debug_log(
            self.section_id,
            "planner_snapshot",
            {
                "round_id": self.pipeline.current_round,
                "task_mode": payload["task_mode"],
                "resolved_task_mode": payload["resolved_task_mode"],
                "mode_confidence": payload["mode_confidence"],
                "mode_evidence": payload["mode_evidence"],
                "visual_score": payload["visual_score"],
                "text_score": payload["text_score"],
                "blocked_tags": payload["blocked_tags"],
                "task_shape_profile": payload["task_shape_profile"],
                "learning_pressure": payload["learning_pressure"],
                "validation_weakness_profile": payload["validation_weakness_profile"],
                "generation_slots": payload["generation_slots"],
            },
        )

    def diversity_constraints(
        self,
        slot: Instructor_Slot,
        round_tag_counts: Counter[str] | None = None,
        rejected_attempts: list[dict[str, str]] | None = None,
        force_contextual_qa: bool = False,
    ) -> str:
        known_counts = known_category_counts(get_learning_tag_counts(self.section_id))
        repeated_round_tags = sorted(
            tag
            for tag, count in (round_tag_counts or Counter()).items()
            if count >= MAX_WIDE_VARIETY_TAGS_PER_ROUND
        )
        constraints = [
            "This is a wide_variety exploration slot.",
            "The API should decide the actual category based on the generated data.",
            "Do not copy or narrowly imitate the user's example unless no other useful task is possible.",
            "Make the task materially different from categories already repeated in the current round.",
            "Change the task operation, domain, input shape, output format, required reasoning, or modality when possible; do not only reword the same extraction/summarization task.",
            "Prefer a rare known category or a genuinely new category.",
        ]
        if force_contextual_qa:
            constraints.extend(
                [
                    "This slot is required to explore context-grounded factual QA because the user provided contextual QA examples.",
                    "Generate a short factual context passage inside the sample field.",
                    "The instruct field must ask a factual question answerable from only that context passage.",
                    "The gold_summary field must answer directly and concisely using facts from the context.",
                    "Prefer tag contextual_qa or factual_qa; information_extraction is acceptable if it best fits.",
                    "Do not make this a recommendation, rewriting, generic summarization, recipe, FAQ, or customer-support task.",
                ]
            )
        if repeated_round_tags:
            constraints.append(
                "Avoid tags already used in this round: "
                + ", ".join(repeated_round_tags)
                + "."
            )
        if known_counts:
            total = sum(known_counts.values())
            dominant_tags = [
                tag
                for tag, count in known_counts.items()
                if count >= WIDE_VARIETY_SECTION_COUNT_LIMIT
                and total > 0
                and (count / total) >= WIDE_VARIETY_SECTION_SHARE_LIMIT
            ]
            if dominant_tags:
                constraints.append(
                    "Avoid these already-dominant section categories unless no "
                    "other category fits: "
                    + ", ".join(sorted(dominant_tags))
                    + "."
                )
        if rejected_attempts:
            constraints.append("Previous rejected attempts in this slot:")
            for index, attempt in enumerate(rejected_attempts, start=1):
                constraints.append(
                    f"{index}. tag={attempt['tag']}; reason={attempt['reason']}; "
                    f"instruction={attempt['instruct']}; sample_preview={attempt['sample']}"
                )
            constraints.append(
                "For the next attempt, do not return the same tag and do not "
                "make a near-paraphrase of the rejected instruction or sample."
            )
        return "\n".join(f"- {line}" for line in constraints)

    def wide_variety_instruction(
        self,
        slot: Instructor_Slot,
        generation_constraints: str | None = None,
    ) -> str:
        constraints = generation_constraints or "No extra constraints."
        return (
            "Generate one synthetic training sample aligned with the user's "
            "examples and the current curriculum. The sample should be broad, "
            "not tied to any fixed local category list or single dataset style. "
            "The API should choose the actual task type and return a canonical "
            "tag that best describes the generated sample. Future datasets may "
            "include different subjects or modalities, so choose from the "
            "available context instead of from a hard-coded taxonomy. Produce "
            "a realistic input, a clear instruction, an expected output, and "
            "an API-assigned canonical tag. "
            f"Additional constraints:\n{constraints}"
        )

    def get_or_create_section_instruction(
        self,
        client: Any,
        example: User_Example,
        slot: Instructor_Slot,
        *,
        attempt: int = 1,
        generation_constraints: str | None = None,
    ) -> str:
        if is_wide_variety_tag(slot.tag):
            return self.wide_variety_instruction(slot, generation_constraints)

        cache_tag = self.instruction_cache_tag(slot, attempt)
        cached_instruction = get_cached_instruction(self.section_id, cache_tag)
        if cached_instruction:
            return cached_instruction

        instruction_prompt = self.build_prompt(
            "instruct",
            example=example,
            slot=slot,
            generation_constraints=generation_constraints,
        )
        save_instruction_prompt(instruction_prompt, self.section_id, cache_tag)

        cached_global_instruction = get_cached_instruction_response(
            prompt=instruction_prompt,
            model_id=self.model_id,
            cache_version=INSTRUCTION_CACHE_VERSION,
        )
        if cached_global_instruction:
            print("Reused global cached instruction response.")
            save_generated_instruction(
                cached_global_instruction,
                self.section_id,
                cache_tag,
            )
            return cached_global_instruction

        instruction = call_client(client=client, prompt=instruction_prompt, model_id=self.model_id)
        save_cached_instruction_response(
            prompt=instruction_prompt,
            response=instruction,
            model_id=self.model_id,
            cache_version=INSTRUCTION_CACHE_VERSION,
            metadata={
                "section_id": self.section_id,
                "cache_tag": cache_tag,
                "slot_id": slot.slot_id,
                "tag": slot.tag,
            },
        )
        save_generated_instruction(instruction, self.section_id, cache_tag)
        return instruction

    def should_accept_wide_variety_sample(
        self,
        sample: Training_Sample,
        round_tag_counts: Counter[str],
        force_contextual_qa: bool = False,
    ) -> tuple[bool, str | None]:
        if force_contextual_qa:
            normalized_tag = normalize_learning_tag(sample.tag)
            if (
                normalized_tag in CONTEXTUAL_QA_BANNED_TAGS
                or normalized_tag not in CONTEXTUAL_QA_TAGS
            ):
                return (
                    False,
                    "contextual QA slot returned non-contextual tag "
                    f"'{sample.tag}'",
                )
            return True, None

        if round_tag_counts[sample.tag] >= MAX_WIDE_VARIETY_TAGS_PER_ROUND:
            return (
                False,
                f"tag already appears {round_tag_counts[sample.tag]} time(s) in this round",
            )

        known_counts = known_category_counts(get_learning_tag_counts(self.section_id))
        section_total = sum(known_counts.values())
        section_count = known_counts.get(sample.tag, 0)
        if (
            section_total > 0
            and section_count >= WIDE_VARIETY_SECTION_COUNT_LIMIT
            and (section_count / section_total) >= WIDE_VARIETY_SECTION_SHARE_LIMIT
        ):
            return (
                False,
                f"tag already dominates the section ({section_count}/{section_total})",
            )

        return True, None

    def build_generation_items(
        self,
        client: Any,
        round_tag_counts: Counter[str],
    ) -> list[dict[str, Any]]:
        generation_items: list[dict[str, Any]] = []
        contextual_examples = self.contextual_examples()
        contextual_qa_slot_id = self.contextual_qa_slot_id(self.list_of_generating)
        for index, slot in enumerate(self.list_of_generating):
            is_exploration = is_wide_variety_tag(slot.tag)
            force_contextual_qa = (
                is_exploration
                and contextual_qa_slot_id is not None
                and slot.slot_id == contextual_qa_slot_id
            )
            example_pool = (
                contextual_examples
                if force_contextual_qa and contextual_examples
                else self.pipeline.user_examples
            )
            example = example_pool[index % len(example_pool)]
            blueprint, constraints = self.planned_generation_constraints(
                slot,
                generation_index=index,
                round_tag_counts=round_tag_counts,
                force_contextual_qa=force_contextual_qa,
            )
            instruction = self.get_or_create_section_instruction(
                client=client,
                example=example,
                slot=slot,
                generation_constraints=constraints,
            )
            generation_items.append(
                {
                    "slot": slot,
                    "slot_id": slot.slot_id,
                    "tag": slot.tag,
                    "example": example,
                    "blueprint": blueprint,
                    "instruction": instruction,
                    "generation_constraints": constraints,
                    "is_exploration": is_exploration,
                    "force_contextual_qa": force_contextual_qa,
                }
            )

        self.save_planner_snapshot(generation_items)
        return generation_items

    def build_batch_data_prompt(
        self,
        generation_items: list[dict[str, Any]],
    ) -> str:
        raw_tag_counts = get_learning_tag_counts(self.section_id)
        tag_counts = format_learning_tag_counts(raw_tag_counts)
        known_categories = format_known_categories(raw_tag_counts)
        slot_payload = [
            {
                "slot_id": item["slot_id"],
                "generation_instruction": item["instruction"],
                "curriculum_focus": item["tag"],
                "generation_intent": item["blueprint"].generation_intent,
                "task_shape_profile": item["blueprint"].task_shape_profile,
                "generation_constraints": (
                    item["generation_constraints"] or "No extra constraints."
                ),
                "user_example_signal": format_user_example_for_prompt(item["example"]),
                "force_contextual_qa": item["force_contextual_qa"],
            }
            for item in generation_items
        ]

        return (
            f"You generate {len(generation_items)} training samples for a "
            "student model.\n\n"
            "Input slots JSON:\n"
            f"{json.dumps(slot_payload, ensure_ascii=False, indent=2)}\n\n"
            "Previously generated tag counts:\n"
            f"{tag_counts}\n\n"
            "Known categories from prior API outputs:\n"
            f"{known_categories}\n\n"
            "Return only one valid JSON array. The array must contain exactly "
            "one object for each input slot, and each object must include these "
            "keys:\n"
            "slot_id\n"
            "sample\n"
            "instruct\n"
            "gold_summary\n"
            "tag\n\n"
            "Rules:\n"
            "- Use each slot's generation_instruction only as private guidance for creating the item; do not copy it into sample, instruct, or gold_summary.\n"
            "- instruct must be the exact task/question shown to the student, not the generation_instruction.\n"
            "- Treat the Diversity planner blueprint inside generation_constraints as shape guidance, not sample content; invent fresh details appropriate to the user's examples and modality.\n"
            "- Use user_example_signal only as a style and task-shape signal; do not copy its exact facts unless the constraints explicitly require it.\n"
            "- If force_contextual_qa is true, the item must be context-grounded factual QA with a context passage in sample and a question in instruct.\n"
            "- If generation_intent is hard, increase task difficulty using the abstract constraints, but keep the sample solvable and the expected output unambiguous.\n"
            "- If generation_intent is validation, create a fresh item that matches the abstract validation-weakness shape while using new facts, wording, and domain.\n"
            "- If task_shape_profile.shape_id is assistant_qa, use a standalone user question without a separate Context block unless another explicit slot constraint says otherwise.\n"
            "- Treat curriculum_focus as guidance, not as the final tag.\n"
            "- The Known categories list is canonical. If the generated sample fits any known category, return that exact category string as tag.\n"
            "- Do not create synonym tags or small wording variations of known categories.\n"
            "- Create a new snake_case tag only when no known category fits the generated sample.\n"
            "- If the focus is wide_variety, create a broad example-aligned task and choose the actual tag yourself.\n"
            "- If the focus names a learned loss group, create a sample that directly trains that weakness while still varying wording, domain, and structure.\n"
            "- Follow each slot's generation_constraints exactly.\n"
            "- Do not return wide_variety as the tag.\n"
            "- sample is the input/context shown to the student.\n"
            "- instruct is the exact instruction the student should follow.\n"
            "- gold_summary is the expected answer/output.\n"
            "- Do not include markdown fences, explanations, apologies, or extra text outside the JSON array."
        )

    def parse_generated_sample_batch(
        self,
        response: str,
        generation_items: list[dict[str, Any]],
        tag_counts: dict[str, int] | None = None,
    ) -> dict[int, Training_Sample]:
        raw_batch = self.parse_generated_json(response)
        if isinstance(raw_batch, dict):
            raw_samples = raw_batch.get("samples") or raw_batch.get("items")
        else:
            raw_samples = raw_batch

        if not isinstance(raw_samples, list):
            raise ValueError("Batch generation response must be a JSON array.")

        items_by_slot = {
            int(item["slot_id"]): item
            for item in generation_items
        }
        parsed_samples: dict[int, Training_Sample] = {}

        for index, raw_sample in enumerate(raw_samples):
            if not isinstance(raw_sample, dict):
                raise ValueError("Every batch item must be a JSON object.")

            raw_slot_id = raw_sample.get("slot_id")
            if raw_slot_id is None and index < len(generation_items):
                raw_slot_id = generation_items[index]["slot_id"]

            try:
                slot_id = int(raw_slot_id)
            except (TypeError, ValueError) as exc:
                raise ValueError("Every batch item must include a numeric slot_id.") from exc

            item = items_by_slot.get(slot_id)
            if item is None:
                raise ValueError(f"Batch response included unknown slot_id={slot_id}.")

            parsed_samples[slot_id] = self.training_sample_from_generated_object(
                raw_sample=raw_sample,
                instruction=str(item["instruction"]),
                tag=str(item["tag"]),
                tag_counts=tag_counts,
            )

        return parsed_samples

    def generate_samples_in_batch(
        self,
        client: Any,
        round_tag_counts: Counter[str],
    ) -> list[Training_Sample]:
        generation_items = self.build_generation_items(client, round_tag_counts)
        data_prompt = self.build_batch_data_prompt(generation_items)
        data_prompt_path = save_data_prompt(data_prompt, self.section_id)
        data_response = call_client(
            client=client,
            prompt=data_prompt,
            model_id=self.model_id,
        )
        data_response_path = save_data_response(
            data_response,
            self.section_id,
            data_prompt_path,
        )
        samples_by_slot = self.parse_generated_sample_batch(
            response=data_response,
            generation_items=generation_items,
            tag_counts=get_learning_tag_counts(self.section_id),
        )

        generated_samples: list[Training_Sample] = []
        for item in generation_items:
            slot = item["slot"]
            sample = samples_by_slot.get(item["slot_id"])
            if sample is None or not (sample.gold_summary or "").strip():
                print(
                    "Batch generation missed a valid sample for "
                    f"slot {item['slot_id']}; falling back to single generation."
                )
                sample = self.generate_sample_for_slot(
                    client=client,
                    example=item["example"],
                    slot=slot,
                    round_tag_counts=round_tag_counts,
                    force_contextual_qa=item["force_contextual_qa"],
                    generation_index=int(item["slot_id"]),
                )
            elif item["is_exploration"]:
                accepted, reason = self.should_accept_wide_variety_sample(
                    sample,
                    round_tag_counts,
                    force_contextual_qa=item["force_contextual_qa"],
                )
                if not accepted:
                    print(
                        "Batch wide_variety sample rejected for "
                        f"slot {item['slot_id']}: {reason}. "
                        "Falling back to single generation."
                    )
                    sample = self.generate_sample_for_slot(
                        client=client,
                        example=item["example"],
                        slot=slot,
                        round_tag_counts=round_tag_counts,
                        force_contextual_qa=item["force_contextual_qa"],
                        generation_index=int(item["slot_id"]),
                    )
                else:
                    update_learning_tag_count(sample.tag, self.section_id)
            else:
                update_learning_tag_count(sample.tag, self.section_id)

            generated_samples.append(sample)
            round_tag_counts[sample.tag] += 1

        append_section_debug_log(
            self.section_id,
            "generation_samples",
            {
                "round_id": self.pipeline.current_round,
                "mode": "batch",
                "data_prompt_path": str(data_prompt_path),
                "data_response_path": str(data_response_path),
                "generated_count": len(generated_samples),
                "tag_counts": dict(Counter(sample.tag for sample in generated_samples)),
                "generation_slots": [
                    {
                        "slot_id": item["slot_id"],
                        "tag": item["tag"],
                        "generation_intent": item["blueprint"].generation_intent,
                        "task_shape": item["blueprint"].task_shape_profile.get("shape_id"),
                        "blueprint": item["blueprint"].as_payload(),
                    }
                    for item in generation_items
                ],
                "samples": [
                    self.sample_debug_row(sample) for sample in generated_samples
                ],
            },
        )
        return generated_samples

    def generate_sample_for_slot(
        self,
        client: Any,
        example: User_Example,
        slot: Instructor_Slot,
        round_tag_counts: Counter[str] | None = None,
        force_contextual_qa: bool = False,
        generation_index: int = 0,
    ) -> Training_Sample:
        active_round_counts = round_tag_counts if round_tag_counts is not None else Counter()
        is_exploration = is_wide_variety_tag(slot.tag)
        max_attempts = WIDE_VARIETY_MAX_ATTEMPTS if is_exploration else 1
        rejected_attempts: list[dict[str, str]] = []
        last_sample: Training_Sample | None = None

        for attempt in range(1, max_attempts + 1):
            blueprint, constraints = self.planned_generation_constraints(
                slot,
                generation_index=generation_index,
                round_tag_counts=active_round_counts,
                rejected_attempts=rejected_attempts,
                force_contextual_qa=force_contextual_qa,
                attempt=attempt,
            )
            instruction = self.get_or_create_section_instruction(
                client=client,
                example=example,
                slot=slot,
                attempt=attempt,
                generation_constraints=constraints,
            )

            data_prompt = self.build_prompt(
                "data",
                example=instruction,
                slot=slot,
                generation_constraints=constraints,
            )
            data_prompt_path = save_data_prompt(data_prompt, self.section_id)
            data_response = call_client(
                client=client,
                prompt=data_prompt,
                model_id=self.model_id,
            )
            data_response_path = save_data_response(
                data_response,
                self.section_id,
                data_prompt_path,
            )

            sample = self.parse_generated_sample(
                response=data_response,
                instruction=instruction,
                tag=slot.tag,
                tag_counts=get_learning_tag_counts(self.section_id),
            )
            last_sample = sample

            if not is_exploration:
                update_learning_tag_count(sample.tag, self.section_id)
                append_section_debug_log(
                    self.section_id,
                    "generation_single_sample",
                    {
                        "round_id": self.pipeline.current_round,
                        "slot": slot.model_dump(mode="json"),
                        "attempt": attempt,
                        "accepted": True,
                        "force_contextual_qa": force_contextual_qa,
                        "data_prompt_path": str(data_prompt_path),
                        "data_response_path": str(data_response_path),
                        "generation_intent": blueprint.generation_intent,
                        "task_shape": blueprint.task_shape_profile.get("shape_id"),
                        "blueprint": blueprint.as_payload(),
                        "sample": self.sample_debug_row(sample),
                    },
                )
                return sample

            accepted, reason = self.should_accept_wide_variety_sample(
                sample,
                active_round_counts,
                force_contextual_qa=force_contextual_qa,
            )
            if accepted:
                update_learning_tag_count(sample.tag, self.section_id)
                append_section_debug_log(
                    self.section_id,
                    "generation_single_sample",
                    {
                        "round_id": self.pipeline.current_round,
                        "slot": slot.model_dump(mode="json"),
                        "attempt": attempt,
                        "accepted": True,
                        "force_contextual_qa": force_contextual_qa,
                        "data_prompt_path": str(data_prompt_path),
                        "data_response_path": str(data_response_path),
                        "generation_intent": blueprint.generation_intent,
                        "task_shape": blueprint.task_shape_profile.get("shape_id"),
                        "blueprint": blueprint.as_payload(),
                        "sample": self.sample_debug_row(sample),
                    },
                )
                return sample

            print(
                "Rejected wide_variety sample "
                f"with tag '{sample.tag}' on attempt {attempt}/{max_attempts}: "
                f"{reason}"
            )
            append_section_debug_log(
                self.section_id,
                "generation_rejected",
                {
                    "round_id": self.pipeline.current_round,
                    "slot": slot.model_dump(mode="json"),
                    "attempt": attempt,
                    "reason": str(reason or "diversity guard"),
                    "force_contextual_qa": force_contextual_qa,
                    "data_prompt_path": str(data_prompt_path),
                    "data_response_path": str(data_response_path),
                    "generation_intent": blueprint.generation_intent,
                    "task_shape": blueprint.task_shape_profile.get("shape_id"),
                    "blueprint": blueprint.as_payload(),
                    "sample": self.sample_debug_row(sample),
                },
            )
            rejected_attempts.append(
                {
                    "tag": sample.tag[:80],
                    "reason": str(reason or "diversity guard")[:160],
                    "instruct": sample.instruct.replace("\n", " ")[:220],
                    "sample": sample.sample.replace("\n", " ")[:220],
                }
            )

        if last_sample is None:
            raise ValueError("Wide variety generation failed before producing a sample.")

        print(
            "Accepting wide_variety sample after short retry limit "
            f"with tag '{last_sample.tag}'."
        )
        update_learning_tag_count(last_sample.tag, self.section_id)
        append_section_debug_log(
            self.section_id,
            "generation_single_sample",
            {
                "round_id": self.pipeline.current_round,
                "slot": slot.model_dump(mode="json"),
                "accepted": True,
                "accepted_after_retry_limit": True,
                "force_contextual_qa": force_contextual_qa,
                "generation_intent": (
                    "hard" if is_hard_generation_tag(slot.tag) else "normal"
                ),
                "sample": self.sample_debug_row(last_sample),
            },
        )
        return last_sample

    def generation(self, Recipe_recieved: Instructor_Recipe | None = None) -> list[Training_Sample] | None:
        if Recipe_recieved is not None:
            self.list_of_generating = [
                slot
                for slot in Recipe_recieved.slots
                if slot.command == CommandQuery.GENERATE
            ]

        if not self.list_of_generating:
            print("there is no data to generate!")
            return None

        state = ensure_user_example(self.section_id)
        self.pipeline = state
        self.examples = state.user_examples

        if not self.pipeline.user_examples:
            raise ValueError("Pipeline state does not have a user example.")

        client = create_client()
        round_tag_counts = Counter(sample.tag for sample in self.SampleList)
        try:
            return self.generate_samples_in_batch(client, round_tag_counts)
        except Exception as exc:
            print(
                "Batch generation failed; falling back to per-slot generation. "
                f"Reason: {exc}"
            )

        generated_samples: list[Training_Sample] = []
        contextual_examples = self.contextual_examples()
        contextual_qa_slot_id = self.contextual_qa_slot_id(self.list_of_generating)
        for index, slot in enumerate(self.list_of_generating):
            force_contextual_qa = (
                is_wide_variety_tag(slot.tag)
                and contextual_qa_slot_id is not None
                and slot.slot_id == contextual_qa_slot_id
            )
            example_pool = (
                contextual_examples
                if force_contextual_qa and contextual_examples
                else self.pipeline.user_examples
            )
            example = example_pool[index % len(example_pool)]
            sample = self.generate_sample_for_slot(
                client=client,
                example=example,
                slot=slot,
                round_tag_counts=round_tag_counts,
                force_contextual_qa=force_contextual_qa,
                generation_index=index,
            )
            generated_samples.append(sample)
            round_tag_counts[sample.tag] += 1

        return generated_samples


instructor = Instructor
