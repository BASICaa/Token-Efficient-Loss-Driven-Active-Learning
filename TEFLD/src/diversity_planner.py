from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
import hashlib
import re
from typing import Any

from .dataschema import Instructor_Slot, Pipeline_State, User_Example, learning_record
from .difficulty import (
    difficulty_budget_for,
    format_difficulty_budget,
    load_difficulty_config,
    shift_difficulty,
)


WIDE_VARIETY_TAG = "wide_variety"
HARD_GENERATION_PREFIX = "hard:"
VALIDATION_GENERATION_PREFIX = "validation:"
ASSISTANT_QA_SHAPE_ID = "assistant_qa"
CONTEXTUAL_QA_SHAPE_ID = "contextual_qa"
GENERIC_TEXT_SHAPE_ID = "generic_text_task"
STRUCTURED_TEXT_SHAPE_ID = "structured_text_task"
VISUAL_GENERATION_SHAPE_ID = "visual_generation"

DOMAIN_HINTS = [
    "education or local public services",
    "workplace operations",
    "travel, transport, or scheduling",
    "retail, product, or customer support",
    "science, museum, library, or community program",
    "home planning or everyday logistics",
    "environment, parks, or civic records",
    "technology use in a small organization",
    "creative visual production or media design",
]

TEXT_DOMAIN_HINTS = [
    hint
    for hint in DOMAIN_HINTS
    if hint != "creative visual production or media design"
]

VISUAL_DOMAIN_HINTS = [
    "creative visual production or media design",
    "product photography or visual merchandising",
    "editorial illustration or campaign assets",
    "interface, icon, or visual asset design",
]

INPUT_SHAPES = [
    "short direct task with no separate context",
    "standalone user question with no separate context",
    "context passage plus one focused question",
    "list of items that must be classified, compared, or transformed",
    "notes, schedule, or constraints that require a decision",
    "short paragraph that must be rewritten into a specific style",
    "small table-like facts described in text",
    "dialogue, email, or log excerpt requiring extraction",
    "simple numeric, date, or time facts requiring calculation",
]

VISUAL_INPUT_SHAPES = [
    "brief creative request with subject, setting, and constraints",
    "visual prompt that specifies composition, style, lighting, and mood",
    "image-edit request with source description and desired transformation",
    "asset-generation request with format, background, and usage constraints",
    "scene description with objects, relationships, and visual priorities",
    "style-transfer request with reference style described in words",
]

OUTPUT_SHAPES = [
    "natural assistant answer",
    "one concise sentence",
    "two short sentences",
    "bullet list",
    "numbered steps",
    "compact table in markdown",
    "JSON array or JSON object",
    "rewritten text only",
    "label for each input item",
]

VISUAL_OUTPUT_SHAPES = [
    "final image prompt only",
    "positive prompt plus negative prompt",
    "structured JSON with subject, style, composition, lighting, and exclusions",
    "short production brief",
    "caption plus generation prompt",
    "edit instruction sequence",
]

REASONING_REQUIREMENTS = [
    "answer a standalone question using stable general knowledge",
    "single-hop factual lookup",
    "choose the relevant fact while ignoring distractors",
    "compare multiple items before answering",
    "apply all stated constraints",
    "perform simple arithmetic or time reasoning",
    "transform wording while preserving meaning",
    "extract only requested fields",
    "summarize briefly before giving the final answer",
]

ASSISTANT_QA_INPUT_SHAPES = [
    "standalone user question with no separate context",
    "short open-ended question phrased like a normal user request",
    "question-only prompt asking for an explanation, definition, list, or how-to answer",
    "direct user question that does not provide a source passage",
]

ASSISTANT_QA_OUTPUT_SHAPES = [
    "natural assistant answer",
    "concise explanatory paragraph",
    "short helpful answer with enough detail to be complete",
    "brief bullet list only when the question asks for a list",
]

ASSISTANT_QA_REASONING_REQUIREMENTS = [
    "answer a standalone question using stable general knowledge",
    "explain the core idea without relying on a provided context passage",
    "include the most relevant facts while staying concise",
    "handle a broad user question without turning it into extraction or rewriting",
]

ASSISTANT_QA_HARD_LEVERS = [
    "ask for a definition plus one important distinction",
    "require a concise explanation with two or three key facts",
    "ask for a short list where each item needs a brief qualifier",
    "include a common misconception that the answer should avoid",
    "ask a multi-part question that still has one natural assistant-style answer",
]

CONTEXTUAL_QA_HARD_LEVERS = [
    "include one answer fact and two plausible distractor facts",
    "use two similar entities that require careful disambiguation",
    "ask about a date, number, or named entity embedded in a short passage",
    "require the answer to be grounded only in the provided context",
]

VISUAL_GENERATION_HARD_LEVERS = [
    "combine subject, composition, lighting, and exclusion constraints",
    "require separation of content details from style details",
    "include format or usage constraints such as aspect ratio or asset purpose",
    "ask for a production-ready prompt instead of a vague visual idea",
]

STRUCTURED_TEXT_HARD_LEVERS = [
    "combine two constraints that both affect the final answer",
    "require comparison before producing the final output",
    "include a compact output format requirement",
    "add one distractor detail that should not be used",
]

VISUAL_REASONING_REQUIREMENTS = [
    "preserve the requested subject while varying style and composition",
    "satisfy visual constraints without adding excluded elements",
    "translate abstract mood into concrete visual details",
    "separate required image content from optional style details",
    "produce a usable prompt with clear composition and quality constraints",
]

DIFFICULTY_LEVELS = ["easy", "medium", "medium", "medium_hard"]

TEXT_TASK_MARKERS = {
    "answer": 1,
    "question": 1,
    "context": 1,
    "classify": 1,
    "identify": 1,
    "rewrite": 1,
    "summarize": 1,
    "extract": 1,
    "compare": 1,
    "calculate": 1,
    "translate": 1,
    "explain": 1,
    "list": 1,
    "output": 1,
}

VISUAL_TASK_MARKERS = {
    "generate an image": 3,
    "create an image": 3,
    "image prompt": 3,
    "prompt for an image": 3,
    "negative prompt": 3,
    "photo": 2,
    "photograph": 2,
    "illustration": 2,
    "render": 2,
    "camera angle": 2,
    "lighting": 2,
    "composition": 2,
    "aspect ratio": 2,
    "logo": 2,
    "poster": 2,
    "sprite": 2,
    "visual asset": 2,
    "image-edit": 2,
    "image edit": 2,
    "background removal": 2,
}

VISUAL_TAGS = {
    "visual_generation",
    "image_generation",
    "image_prompting",
    "image_editing",
    "poster_design",
    "logo_design",
    "visual_asset_generation",
}

TEXT_MODE_BLOCKED_TAGS = sorted(VISUAL_TAGS)

QUESTION_START_PATTERN = re.compile(
    r"^\s*(what|when|where|who|whom|whose|why|how|which|can|could|would|should|"
    r"is|are|do|does|did|name|list|give|tell|explain|describe)\b",
    re.IGNORECASE,
)

CONTEXT_MARKERS = {
    "context:",
    "passage:",
    "article:",
    "source text:",
    "given the following",
    "based on the following",
}

STRUCTURED_OPERATION_MARKERS = {
    "classify": 2,
    "categorize": 2,
    "rewrite": 2,
    "extract": 2,
    "label": 2,
    "translate": 2,
    "convert": 2,
    "return json": 2,
    "json object": 2,
    "json array": 2,
    "table": 1,
    "compare the": 1,
    "calculate": 1,
    "sort": 1,
    "rank": 1,
}


def normalize_tag(tag: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "_", tag.strip().lower())
    return re.sub(r"_+", "_", normalized).strip("_")


def split_generation_intent(tag: str) -> tuple[str, str]:
    raw_tag = (tag or "").strip()
    if raw_tag.lower().startswith(HARD_GENERATION_PREFIX):
        return "hard", raw_tag[len(HARD_GENERATION_PREFIX) :].strip() or WIDE_VARIETY_TAG
    if raw_tag.lower().startswith(VALIDATION_GENERATION_PREFIX):
        return (
            "validation",
            raw_tag[len(VALIDATION_GENERATION_PREFIX) :].strip() or WIDE_VARIETY_TAG,
        )
    return "normal", raw_tag


def base_generation_tag(tag: str) -> str:
    _intent, base_tag = split_generation_intent(tag)
    return base_tag


def is_hard_generation_tag(tag: str) -> bool:
    intent, _base_tag = split_generation_intent(tag)
    return intent == "hard"


def hard_generation_tag(base_tag: str) -> str:
    cleaned = base_tag.strip() or WIDE_VARIETY_TAG
    return f"{HARD_GENERATION_PREFIX}{cleaned}"


def is_validation_generation_tag(tag: str) -> bool:
    intent, _base_tag = split_generation_intent(tag)
    return intent == "validation"


def validation_generation_tag(base_tag: str) -> str:
    cleaned = base_tag.strip() or WIDE_VARIETY_TAG
    return f"{VALIDATION_GENERATION_PREFIX}{cleaned}"


def stable_index(seed_parts: list[Any], size: int) -> int:
    if size <= 0:
        raise ValueError("Cannot choose from an empty planner list.")

    seed = "|".join(str(part) for part in seed_parts)
    digest = hashlib.sha256(seed.encode("utf-8")).hexdigest()
    return int(digest[:12], 16) % size


def choose_rotated(options: list[str], *seed_parts: Any) -> str:
    return options[stable_index(list(seed_parts), len(options))]


def compact_text(text: str, limit: int = 140) -> str:
    cleaned = re.sub(r"\s+", " ", text).strip()
    return cleaned[:limit]


def marker_hits(text: str, markers: dict[str, int]) -> tuple[list[str], int]:
    hits: list[str] = []
    score = 0
    for marker, weight in markers.items():
        pattern = r"(?<![a-z0-9])" + re.escape(marker) + r"(?![a-z0-9])"
        if re.search(pattern, text):
            hits.append(marker)
            score += weight
    return hits, score


def is_question_like(text: str) -> bool:
    cleaned = re.sub(r"\s+", " ", text or "").strip()
    if not cleaned:
        return False
    return "?" in cleaned[:220] or bool(QUESTION_START_PATTERN.search(cleaned))


def has_context_signal(*parts: str) -> bool:
    text = "\n".join(part for part in parts if part).lower()
    if any(marker in text for marker in CONTEXT_MARKERS):
        return True
    # A long sample paired with a short question usually behaves like context QA.
    sample = parts[0] if parts else ""
    instruction = parts[1] if len(parts) > 1 else ""
    return len(sample.split()) >= 45 and is_question_like(instruction)


def structured_operation_score(text: str) -> int:
    _hits, score = marker_hits((text or "").lower(), STRUCTURED_OPERATION_MARKERS)
    return score


def record_text_fields(record: Any) -> tuple[str, str, str, str | None, str | None]:
    if isinstance(record, dict):
        sample = str(record.get("sample") or "")
        instruct = str(record.get("instruct") or record.get("instruction") or "")
        output = str(
            record.get("output")
            or record.get("gold_summary")
            or record.get("summary")
            or ""
        )
        context = record.get("context")
        mode = record.get("mode")
        return (
            sample,
            instruct,
            output,
            str(context) if context is not None else None,
            str(mode) if mode is not None else None,
        )

    sample = str(getattr(record, "sample", "") or "")
    instruct = str(getattr(record, "instruct", "") or "")
    output = str(
        getattr(record, "output", None)
        or getattr(record, "gold_summary", None)
        or ""
    )
    context = getattr(record, "context", None)
    mode = getattr(record, "mode", None)
    return (
        sample,
        instruct,
        output,
        str(context) if context is not None else None,
        str(mode) if mode is not None else None,
    )


def classify_task_shape_record(record: Any) -> tuple[str, list[str]]:
    sample, instruct, output, context, mode = record_text_fields(record)
    combined_text = " ".join(part for part in (sample, instruct, output, context or "") if part)
    combined_lower = combined_text.lower()
    visual_hits, visual_score = marker_hits(combined_lower, VISUAL_TASK_MARKERS)
    _text_hits, structured_score = marker_hits(combined_lower, STRUCTURED_OPERATION_MARKERS)
    context_like = (
        bool(context)
        or mode == "contextual"
        or has_context_signal(context or sample, instruct)
    )
    question_text = instruct or sample
    question_like = is_question_like(question_text)

    evidence: list[str] = []
    if visual_hits:
        evidence.extend(f"visual:{hit}" for hit in visual_hits[:3])
    if context_like:
        evidence.append("context_signal")
    if question_like:
        evidence.append("question_like")
    if structured_score:
        evidence.append(f"structured_score:{structured_score}")

    if visual_score >= 3:
        return VISUAL_GENERATION_SHAPE_ID, evidence
    if context_like and question_like:
        return CONTEXTUAL_QA_SHAPE_ID, evidence
    if question_like and structured_score < 2:
        return ASSISTANT_QA_SHAPE_ID, evidence
    if structured_score > 0:
        return STRUCTURED_TEXT_SHAPE_ID, evidence
    return GENERIC_TEXT_SHAPE_ID, evidence


def task_shape_levers(shape_id: str) -> list[str]:
    if shape_id == ASSISTANT_QA_SHAPE_ID:
        return ASSISTANT_QA_HARD_LEVERS
    if shape_id == CONTEXTUAL_QA_SHAPE_ID:
        return CONTEXTUAL_QA_HARD_LEVERS
    if shape_id == VISUAL_GENERATION_SHAPE_ID:
        return VISUAL_GENERATION_HARD_LEVERS
    return STRUCTURED_TEXT_HARD_LEVERS


def task_shape_focus_tag(shape_id: str) -> str:
    if shape_id == ASSISTANT_QA_SHAPE_ID:
        return "assistant_qa"
    if shape_id == CONTEXTUAL_QA_SHAPE_ID:
        return "contextual_qa"
    if shape_id == VISUAL_GENERATION_SHAPE_ID:
        return "visual_generation"
    if shape_id == STRUCTURED_TEXT_SHAPE_ID:
        return "structured_text_task"
    return WIDE_VARIETY_TAG


def infer_task_shape_profile(
    records: list[Any] | dict[str, Any],
    *,
    source: str,
    fallback_modality: str = "text",
) -> dict[str, Any]:
    if isinstance(records, dict):
        raw_records = (
            records.get("samples")
            or records.get("items")
            or records.get("rows")
            or []
        )
        records = raw_records if isinstance(raw_records, list) else []

    if not records:
        return {
            "shape_id": GENERIC_TEXT_SHAPE_ID,
            "modality": fallback_modality,
            "input_shape": "unknown",
            "context_mode": "unknown",
            "output_shape": "unknown",
            "answer_style": "unknown",
            "reasoning_pattern": "unknown",
            "confidence": 0.0,
            "evidence": ["no_records"],
            "source": [source],
            "recommended_focus_tag": WIDE_VARIETY_TAG,
            "hard_levers": task_shape_levers(GENERIC_TEXT_SHAPE_ID),
            "shape_counts": {},
            "record_count": 0,
        }

    counts: Counter[str] = Counter()
    evidence_counts: Counter[str] = Counter()
    for record in records:
        shape_id, evidence = classify_task_shape_record(record)
        counts[shape_id] += 1
        evidence_counts.update(evidence)

    shape_id, top_count = counts.most_common(1)[0]
    record_count = len(records)
    share = top_count / record_count
    confidence = min(0.95, 0.35 + share * 0.60)
    modality = "visual" if shape_id == VISUAL_GENERATION_SHAPE_ID else "text"
    if fallback_modality == "visual" and shape_id != VISUAL_GENERATION_SHAPE_ID:
        modality = "mixed"

    if shape_id == ASSISTANT_QA_SHAPE_ID:
        input_shape = "standalone_question"
        context_mode = "none"
        output_shape = "natural_assistant_answer"
        answer_style = "helpful_concise"
        reasoning_pattern = "open_knowledge_or_explanation"
    elif shape_id == CONTEXTUAL_QA_SHAPE_ID:
        input_shape = "context_plus_question"
        context_mode = "required"
        output_shape = "grounded_direct_answer"
        answer_style = "concise_context_grounded"
        reasoning_pattern = "select_relevant_fact_from_context"
    elif shape_id == VISUAL_GENERATION_SHAPE_ID:
        input_shape = "creative_visual_request"
        context_mode = "visual_constraints"
        output_shape = "image_prompt_or_visual_brief"
        answer_style = "production_ready_visual_spec"
        reasoning_pattern = "compose_visual_constraints"
    elif shape_id == STRUCTURED_TEXT_SHAPE_ID:
        input_shape = "structured_text_task"
        context_mode = "optional"
        output_shape = "format_constrained_answer"
        answer_style = "task_specific"
        reasoning_pattern = "apply_explicit_operation"
    else:
        input_shape = "example_aligned_text"
        context_mode = "optional"
        output_shape = "task_specific_answer"
        answer_style = "example_aligned"
        reasoning_pattern = "follow_user_task"

    return {
        "shape_id": shape_id,
        "modality": modality,
        "input_shape": input_shape,
        "context_mode": context_mode,
        "output_shape": output_shape,
        "answer_style": answer_style,
        "reasoning_pattern": reasoning_pattern,
        "confidence": round(confidence, 4),
        "evidence": [
            f"{shape}:{count}" for shape, count in counts.most_common()
        ][:6]
        + [
            f"signal:{signal}:{count}"
            for signal, count in evidence_counts.most_common(6)
        ],
        "source": [source],
        "recommended_focus_tag": task_shape_focus_tag(shape_id),
        "hard_levers": task_shape_levers(shape_id),
        "shape_counts": dict(counts),
        "record_count": record_count,
    }


@dataclass(frozen=True)
class PlannerModeInfo:
    task_mode: str
    resolved_task_mode: str
    mode_confidence: float
    mode_evidence: list[str]
    visual_score: int
    text_score: int
    example_hash: str

    def as_payload(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class GenerationBlueprint:
    slot_id: int
    curriculum_focus: str
    generation_intent: str
    task_shape_profile: dict[str, Any]
    target_tag_hint: str
    novelty_goal: str
    difficulty: str
    difficulty_budget: dict[str, Any]
    domain_hint: str
    input_shape: str
    output_shape: str
    reasoning_requirement: str
    context_policy: str
    modality_hint: str
    task_mode: str
    resolved_task_mode: str
    mode_confidence: float
    avoid_tags: list[str]
    avoid_recent_instructions: list[str]
    avoid_recent_sample_shapes: list[str]

    def as_payload(self) -> dict[str, Any]:
        return asdict(self)

    def as_constraints(self) -> str:
        hard_levers = self.task_shape_profile.get("hard_levers") or []
        lines = [
            "Diversity planner blueprint. Treat this as shape guidance, not as sample content.",
            f"Generation intent: {self.generation_intent}.",
            (
                "Discovered task shape: "
                f"{self.task_shape_profile.get('shape_id', 'unknown')} "
                f"(source={', '.join(self.task_shape_profile.get('source', []))}, "
                f"confidence={float(self.task_shape_profile.get('confidence', 0.0)):.2f})."
            ),
            f"Target tag hint: {self.target_tag_hint}.",
            f"Novelty goal: {self.novelty_goal}.",
            f"Difficulty: {self.difficulty}.",
            format_difficulty_budget(self.difficulty_budget),
            f"Domain hint: {self.domain_hint}.",
            f"Input shape: {self.input_shape}.",
            f"Output shape: {self.output_shape}.",
            f"Reasoning requirement: {self.reasoning_requirement}.",
            f"Context policy: {self.context_policy}.",
            f"Modality hint: {self.modality_hint}.",
            (
                "Resolved task mode: "
                f"{self.resolved_task_mode} (configured task_mode={self.task_mode}, "
                f"confidence={self.mode_confidence:.2f})."
            ),
        ]

        if self.resolved_task_mode == "text":
            lines.append(
                "Stay in text-task mode; do not create image, poster, logo, visual asset, or image-prompt tasks."
            )
        elif self.resolved_task_mode == "visual":
            lines.append(
                "Stay in visual-task mode; create image, visual prompt, asset, or edit-instruction tasks."
            )
        else:
            lines.append(
                "Mixed mode is allowed, but keep the generated item consistent with the selected modality hint."
            )

        if self.generation_intent == "hard":
            lines.append(
                "This is a hard-generation slot: make the task harder than recent generated samples while keeping it valid, teachable, and aligned with the discovered task shape."
            )
            if hard_levers:
                lines.append("Use one or two of these abstract difficulty levers:")
                lines.extend(f"  - {lever}" for lever in hard_levers[:5])

        if self.generation_intent == "validation":
            lines.append(
                "This is a validation-weakness slot: create a fresh training item that matches the abstract task shape the model recently struggled with on shared validation, without copying validation content or facts."
            )
            lines.append(
                "Favor the indicated input shape, output shape, and reasoning pattern over broad exploration."
            )

        if self.task_shape_profile.get("shape_id") == ASSISTANT_QA_SHAPE_ID:
            lines.append(
                "For assistant_qa, use a standalone user question and do not add a separate Context block unless another explicit constraint overrides this."
            )

        output_format = self.difficulty_budget.get("output_format")
        if output_format == "strict":
            lines.append(
                "For strict output format, require JSON, a markdown table, or exact labels in the student-facing instruction and expected output."
            )
        elif output_format == "structured":
            lines.append(
                "For structured output format, prefer bullets, numbered steps, compact key-value lines, or labeled fields."
            )

        if self.avoid_tags:
            lines.append("Avoid overused tags when possible: " + ", ".join(self.avoid_tags) + ".")
        if self.avoid_recent_instructions:
            lines.append("Avoid near-duplicates of these recent instruction shapes:")
            lines.extend(f"  - {item}" for item in self.avoid_recent_instructions)
        if self.avoid_recent_sample_shapes:
            lines.append("Avoid near-duplicates of these recent sample shapes:")
            lines.extend(f"  - {item}" for item in self.avoid_recent_sample_shapes)

        lines.append(
            "Invent fresh details appropriate to the user's examples; do not copy wording, entities, visual concepts, numbers, or labels from this blueprint."
        )
        return "\n".join(f"- {line}" for line in lines)


class DiversityPlanner:
    """
    Builds lightweight generation blueprints from the current section state.

    The planner does not contain task examples. It rotates generic dimensions
    such as input shape, output shape, domain hint, and reasoning requirement
    so the API creates varied data without a hard-coded sample bank.
    """

    def __init__(
        self,
        *,
        pipeline: Pipeline_State,
        ledger: list[learning_record],
    ) -> None:
        self.pipeline = pipeline
        self.ledger = ledger

    def plan(
        self,
        *,
        slot: Instructor_Slot,
        generation_index: int,
        round_tag_counts: Counter[str],
        force_contextual_qa: bool = False,
        attempt: int = 1,
    ) -> GenerationBlueprint:
        generation_intent, base_tag = split_generation_intent(slot.tag)
        focus = normalize_tag(base_tag) or base_tag
        is_exploration = focus == WIDE_VARIETY_TAG
        seed_parts = [
            self.pipeline.section_id,
            self.pipeline.current_round,
            slot.slot_id,
            generation_index,
            base_tag,
            generation_intent,
            attempt,
        ]

        avoid_tags = self.overused_tags(round_tag_counts)
        mode_info = self.resolve_task_mode()
        task_shape_profile = self.task_shape_profile(mode_info)
        if generation_intent == "validation":
            task_shape_profile = self.validation_generation_profile(
                focus=focus,
                mode_info=mode_info,
                fallback_profile=task_shape_profile,
            )
        blocked_tags = self.blocked_tags_for_mode(mode_info)
        avoid_tags = list(dict.fromkeys([*avoid_tags, *blocked_tags]))
        if not is_exploration:
            avoid_tags = [tag for tag in avoid_tags if tag != focus]
        if force_contextual_qa:
            avoid_tags = [
                tag
                for tag in avoid_tags
                if tag not in {"contextual_qa", "factual_qa", "information_extraction"}
            ]
        rare_tags = self.rare_known_tags(avoid_tags)
        target_tag_hint = self.target_tag_hint(
            focus=focus,
            is_exploration=is_exploration,
            force_contextual_qa=force_contextual_qa,
            rare_tags=rare_tags,
            generation_intent=generation_intent,
            task_shape_profile=task_shape_profile,
        )

        modality_hint = self.modality_hint(
            focus=focus,
            is_exploration=is_exploration,
            seed_parts=seed_parts,
            mode_info=mode_info,
            generation_intent=generation_intent,
            task_shape_profile=task_shape_profile,
        )
        input_shapes = (
            VISUAL_INPUT_SHAPES if modality_hint == "visual_generation" else INPUT_SHAPES
        )
        output_shapes = (
            VISUAL_OUTPUT_SHAPES if modality_hint == "visual_generation" else OUTPUT_SHAPES
        )
        reasoning_requirements = (
            VISUAL_REASONING_REQUIREMENTS
            if modality_hint == "visual_generation"
            else REASONING_REQUIREMENTS
        )

        input_shape = choose_rotated(input_shapes, *seed_parts, "input")
        output_shape = choose_rotated(output_shapes, *seed_parts, "output")
        reasoning_requirement = choose_rotated(reasoning_requirements, *seed_parts, "reasoning")

        if generation_intent in {"hard", "validation"}:
            (
                modality_hint,
                input_shape,
                output_shape,
                reasoning_requirement,
            ) = self.hard_generation_shape(
                task_shape_profile=task_shape_profile,
                focus=focus,
                seed_parts=seed_parts,
                current_modality_hint=modality_hint,
            )

        if force_contextual_qa:
            modality_hint = "text_contextual_qa"
            input_shape = "context passage plus one focused question"
            output_shape = "one concise sentence"
            reasoning_requirement = "choose the relevant fact while ignoring distractors"

        difficulty = self.difficulty(*seed_parts, generation_intent=generation_intent)
        shape_id = str(task_shape_profile.get("shape_id") or "")
        difficulty_budget = difficulty_budget_for(
            difficulty=difficulty,
            tag=target_tag_hint if isinstance(target_tag_hint, str) else focus,
            shape_id=shape_id,
            config=load_difficulty_config(self.pipeline.section_id),
        )

        return GenerationBlueprint(
            slot_id=slot.slot_id,
            curriculum_focus=slot.tag,
            generation_intent=generation_intent,
            task_shape_profile=task_shape_profile,
            target_tag_hint=target_tag_hint,
            novelty_goal=self.novelty_goal(is_exploration, rare_tags, generation_intent),
            difficulty=difficulty,
            difficulty_budget=difficulty_budget,
            domain_hint=choose_rotated(
                VISUAL_DOMAIN_HINTS
                if modality_hint == "visual_generation"
                else TEXT_DOMAIN_HINTS,
                *seed_parts,
                "domain",
            ),
            input_shape=input_shape,
            output_shape=output_shape,
            reasoning_requirement=reasoning_requirement,
            context_policy=self.context_policy(
                force_contextual_qa,
                input_shape,
                task_shape_profile=task_shape_profile,
            ),
            modality_hint=modality_hint,
            task_mode=mode_info.task_mode,
            resolved_task_mode=mode_info.resolved_task_mode,
            mode_confidence=mode_info.mode_confidence,
            avoid_tags=avoid_tags[:5],
            avoid_recent_instructions=self.recent_instructions(),
            avoid_recent_sample_shapes=self.recent_sample_shapes(),
        )

    def tag_counts(self) -> Counter[str]:
        counts: Counter[str] = Counter()
        for tag, count in self.pipeline.learning_tag_counts.items():
            normalized = normalize_tag(tag)
            if normalized and normalized != WIDE_VARIETY_TAG:
                counts[normalized] += int(count)
        return counts

    def overused_tags(self, round_tag_counts: Counter[str]) -> list[str]:
        counts = self.tag_counts()
        total = sum(counts.values())
        overused = [
            tag
            for tag, count in counts.items()
            if total > 0 and count >= 3 and (count / total) >= 0.30
        ]
        overused.extend(
            normalize_tag(tag)
            for tag, count in round_tag_counts.items()
            if count >= 1 and normalize_tag(tag)
        )
        return list(dict.fromkeys(tag for tag in overused if tag))

    def rare_known_tags(self, avoid_tags: list[str]) -> list[str]:
        counts = self.tag_counts()
        if not counts:
            return []

        avoid = set(avoid_tags)
        min_count = min(counts.values())
        return [
            tag
            for tag, count in sorted(counts.items(), key=lambda item: (item[1], item[0]))
            if count <= min_count + 1 and tag not in avoid
        ][:4]

    @staticmethod
    def target_tag_hint(
        *,
        focus: str,
        is_exploration: bool,
        force_contextual_qa: bool,
        rare_tags: list[str],
        generation_intent: str = "normal",
        task_shape_profile: dict[str, Any] | None = None,
    ) -> str:
        if force_contextual_qa:
            return "contextual_qa or factual_qa"
        if generation_intent == "hard":
            profile_tag = (task_shape_profile or {}).get("recommended_focus_tag")
            if focus and focus != WIDE_VARIETY_TAG:
                return focus
            if profile_tag:
                return str(profile_tag)
        if generation_intent == "validation":
            profile_tag = (task_shape_profile or {}).get("recommended_focus_tag")
            if focus and focus != WIDE_VARIETY_TAG:
                return focus
            if profile_tag:
                return str(profile_tag)
        if is_exploration:
            if rare_tags:
                return "prefer one of these rare known tags if it fits: " + ", ".join(rare_tags)
            return "API should choose a useful non-wide_variety tag"
        return focus

    @staticmethod
    def novelty_goal(
        is_exploration: bool,
        rare_tags: list[str],
        generation_intent: str = "normal",
    ) -> str:
        if generation_intent == "hard":
            return (
                "create a harder-but-valid sample in the discovered task shape, "
                "using fresh content and avoiding noisy or impossible questions"
            )
        if generation_intent == "validation":
            return (
                "train the current shared-validation weakness with fresh content, "
                "same abstract task shape, and different facts, wording, and domain"
            )
        if is_exploration and rare_tags:
            return "explore an underrepresented known category or adjacent new category"
        if is_exploration:
            return "create a fresh example-aligned training task with a distinct shape"
        return "train the focused weakness while changing domain, wording, and structure"

    def difficulty(self, *seed_parts: Any, generation_intent: str = "normal") -> str:
        if generation_intent == "hard":
            options = ["medium_hard", "hard", "hard"]
            base_difficulty = choose_rotated(options, *seed_parts, "difficulty")
        elif generation_intent == "validation":
            options = ["medium", "medium_hard", "medium_hard"]
            base_difficulty = choose_rotated(options, *seed_parts, "difficulty")
        elif self.pipeline.current_round >= 8:
            options = ["medium", "medium_hard", "medium_hard"]
            base_difficulty = choose_rotated(options, *seed_parts, "difficulty")
        elif self.pipeline.current_round >= 3:
            options = ["easy", "medium", "medium_hard"]
            base_difficulty = choose_rotated(options, *seed_parts, "difficulty")
        else:
            options = DIFFICULTY_LEVELS
            base_difficulty = choose_rotated(options, *seed_parts, "difficulty")

        difficulty_state = getattr(self.pipeline, "difficulty_state", {}) or {}
        delta = int(difficulty_state.get("difficulty_delta") or 0)
        return shift_difficulty(base_difficulty, delta)

    @staticmethod
    def context_policy(
        force_contextual_qa: bool,
        input_shape: str,
        *,
        task_shape_profile: dict[str, Any] | None = None,
    ) -> str:
        if force_contextual_qa:
            return (
                "must include a self-contained factual context with one answer fact "
                "and at least one distractor fact"
            )
        if (task_shape_profile or {}).get("shape_id") == ASSISTANT_QA_SHAPE_ID:
            return (
                "do not include a separate Context block; make the sample a "
                "standalone user question and answer from stable general knowledge"
            )
        if "context" in input_shape or "facts" in input_shape or "log" in input_shape:
            return "include enough input details so the answer is grounded in the sample"
        return "context optional; keep the task self-contained"

    def original_example_text(self) -> str:
        return " ".join(
            " ".join(
                part
                for part in (
                    example.sample,
                    example.instruct or "",
                    example.context or "",
                    example.output,
                )
                if part
            )
            for example in self.pipeline.user_examples
        ).lower()

    def example_hash(self) -> str:
        text = self.original_example_text()
        return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]

    def resolve_task_mode(self) -> PlannerModeInfo:
        configured_mode = getattr(self.pipeline, "task_mode", "auto") or "auto"
        text = self.original_example_text()
        visual_hits, visual_score = marker_hits(text, VISUAL_TASK_MARKERS)
        text_hits, text_score = marker_hits(text, TEXT_TASK_MARKERS)
        evidence = [
            *(f"visual:{hit}" for hit in visual_hits),
            *(f"text:{hit}" for hit in text_hits),
        ]

        if configured_mode in {"text", "visual", "mixed"}:
            return PlannerModeInfo(
                task_mode=configured_mode,
                resolved_task_mode=configured_mode,
                mode_confidence=1.0,
                mode_evidence=evidence or [f"explicit:{configured_mode}"],
                visual_score=visual_score,
                text_score=text_score,
                example_hash=self.example_hash(),
            )

        if visual_score >= 3 and text_score >= 2:
            resolved_mode = "mixed"
            confidence = 0.75
        elif visual_score >= 3:
            resolved_mode = "visual"
            confidence = 0.85
        else:
            resolved_mode = "text"
            confidence = 0.90 if text_score else 0.70

        return PlannerModeInfo(
            task_mode="auto",
            resolved_task_mode=resolved_mode,
            mode_confidence=confidence,
            mode_evidence=evidence or ["default:text_no_strong_visual_evidence"],
            visual_score=visual_score,
            text_score=text_score,
            example_hash=self.example_hash(),
        )

    def user_example_task_shape_profile(self, mode_info: PlannerModeInfo) -> dict[str, Any]:
        fallback_modality = (
            "visual" if mode_info.resolved_task_mode == "visual" else "text"
        )
        return infer_task_shape_profile(
            list(self.pipeline.user_examples),
            source="original_user_examples",
            fallback_modality=fallback_modality,
        )

    def task_shape_profile(self, mode_info: PlannerModeInfo | None = None) -> dict[str, Any]:
        mode_info = mode_info or self.resolve_task_mode()
        user_profile = self.user_example_task_shape_profile(mode_info)
        user_example_shape_id = user_profile.get("shape_id")
        user_example_confidence = user_profile.get("confidence")
        pressure_state = getattr(self.pipeline, "learning_pressure_state", {}) or {}
        validation_profile = pressure_state.get("validation_task_shape_profile") or {}

        recent_records = self.ledger[-60:] if self.ledger else []
        if len(recent_records) >= 10:
            fallback_modality = (
                "visual" if mode_info.resolved_task_mode == "visual" else "text"
            )
            ledger_profile = infer_task_shape_profile(
                recent_records,
                source="recent_learning_ledger",
                fallback_modality=fallback_modality,
            )
            merged_counts: Counter[str] = Counter(
                user_profile.get("shape_counts") or {}
            )
            for shape, count in (ledger_profile.get("shape_counts") or {}).items():
                merged_counts[str(shape)] += int(count) * 2

            if merged_counts:
                top_shape, top_count = merged_counts.most_common(1)[0]
                total = sum(merged_counts.values())
                share = top_count / total
                user_profile = dict(ledger_profile)
                user_profile["shape_id"] = top_shape
                user_profile["shape_counts"] = dict(merged_counts)
                user_profile["confidence"] = round(
                    min(0.95, 0.35 + share * 0.60),
                    4,
                )
                user_profile["recommended_focus_tag"] = task_shape_focus_tag(top_shape)
                user_profile["hard_levers"] = task_shape_levers(top_shape)
                user_profile["source"] = [
                    "recent_learning_ledger",
                    "original_user_examples",
                ]
                user_profile["ledger_shape_id"] = ledger_profile.get("shape_id")
                user_profile["ledger_confidence"] = ledger_profile.get("confidence")
                user_profile["user_example_shape_id"] = user_example_shape_id
                user_profile["user_example_confidence"] = user_example_confidence

        if (
            isinstance(validation_profile, dict)
            and mode_info.resolved_task_mode == "text"
            and validation_profile.get("shape_id") == ASSISTANT_QA_SHAPE_ID
            and float(validation_profile.get("confidence", 0.0)) >= 0.55
        ):
            combined = dict(validation_profile)
            combined["source"] = list(
                dict.fromkeys(
                    [
                        *(validation_profile.get("source") or ["validation_eval_profile"]),
                        "original_user_examples",
                    ]
                )
            )
            combined["user_example_shape_id"] = user_profile.get("shape_id")
            combined["user_example_confidence"] = user_profile.get("confidence")
            return combined

        return user_profile

    def active_validation_weakness_profile(self) -> dict[str, Any]:
        profile = getattr(self.pipeline, "validation_weakness_profile", {}) or {}
        if isinstance(profile, dict) and profile.get("top_categories"):
            return profile

        pressure_state = getattr(self.pipeline, "learning_pressure_state", {}) or {}
        profile = pressure_state.get("validation_weakness_profile") or {}
        return profile if isinstance(profile, dict) else {}

    def validation_generation_profile(
        self,
        *,
        focus: str,
        mode_info: PlannerModeInfo,
        fallback_profile: dict[str, Any],
    ) -> dict[str, Any]:
        weakness_profile = self.active_validation_weakness_profile()
        categories = weakness_profile.get("top_categories")
        if not isinstance(categories, list) or not categories:
            return fallback_profile

        normalized_focus = normalize_tag(focus)
        selected_category: dict[str, Any] | None = None
        for category in categories:
            if not isinstance(category, dict):
                continue
            category_tag = normalize_tag(str(category.get("tag") or ""))
            if category_tag and category_tag == normalized_focus:
                selected_category = category
                break

        if selected_category is None:
            selected_category = next(
                (category for category in categories if isinstance(category, dict)),
                None,
            )
        if selected_category is None:
            return fallback_profile

        shape_id = str(
            selected_category.get("shape_id")
            or fallback_profile.get("shape_id")
            or GENERIC_TEXT_SHAPE_ID
        )
        if mode_info.resolved_task_mode == "text" and shape_id == VISUAL_GENERATION_SHAPE_ID:
            return fallback_profile

        profile = dict(fallback_profile)
        profile.update(
            {
                "shape_id": shape_id,
                "modality": (
                    "visual"
                    if shape_id == VISUAL_GENERATION_SHAPE_ID
                    else "text"
                ),
                "input_shape": selected_category.get(
                    "input_shape",
                    fallback_profile.get("input_shape", "example_aligned_text"),
                ),
                "context_mode": selected_category.get(
                    "context_mode",
                    fallback_profile.get("context_mode", "optional"),
                ),
                "output_shape": selected_category.get(
                    "output_shape",
                    fallback_profile.get("output_shape", "task_specific_answer"),
                ),
                "answer_style": selected_category.get(
                    "answer_style",
                    fallback_profile.get("answer_style", "example_aligned"),
                ),
                "reasoning_pattern": selected_category.get(
                    "reasoning_pattern",
                    fallback_profile.get("reasoning_pattern", "follow_user_task"),
                ),
                "confidence": max(
                    float(fallback_profile.get("confidence", 0.0) or 0.0),
                    float(selected_category.get("confidence", 0.0) or 0.0),
                ),
                "source": list(
                    dict.fromkeys(
                        [
                            "shared_validation_loss",
                            *(fallback_profile.get("source") or []),
                        ]
                    )
                ),
                "recommended_focus_tag": str(
                    selected_category.get("tag")
                    or task_shape_focus_tag(shape_id)
                ),
                "hard_levers": task_shape_levers(shape_id),
                "validation_weakness_rank": selected_category.get("rank"),
                "validation_category_avg_loss": selected_category.get("avg_loss"),
                "validation_category_max_loss": selected_category.get("max_loss"),
                "validation_round_id": weakness_profile.get("round_id"),
                "validation_profile_avg_loss": weakness_profile.get("avg_loss"),
            }
        )
        return profile

    def hard_generation_shape(
        self,
        *,
        task_shape_profile: dict[str, Any],
        focus: str,
        seed_parts: list[Any],
        current_modality_hint: str,
    ) -> tuple[str, str, str, str]:
        shape_id = task_shape_profile.get("shape_id")
        if focus == ASSISTANT_QA_SHAPE_ID or shape_id == ASSISTANT_QA_SHAPE_ID:
            return (
                "text_assistant_qa",
                choose_rotated(ASSISTANT_QA_INPUT_SHAPES, *seed_parts, "hard_input"),
                choose_rotated(ASSISTANT_QA_OUTPUT_SHAPES, *seed_parts, "hard_output"),
                choose_rotated(
                    ASSISTANT_QA_REASONING_REQUIREMENTS,
                    *seed_parts,
                    "hard_reasoning",
                ),
            )
        if focus == CONTEXTUAL_QA_SHAPE_ID or shape_id == CONTEXTUAL_QA_SHAPE_ID:
            return (
                "text_contextual_qa",
                "context passage plus one focused question",
                "grounded direct answer",
                "choose the relevant fact while ignoring distractors",
            )
        if focus in VISUAL_TAGS or shape_id == VISUAL_GENERATION_SHAPE_ID:
            return (
                "visual_generation",
                choose_rotated(VISUAL_INPUT_SHAPES, *seed_parts, "hard_input"),
                choose_rotated(VISUAL_OUTPUT_SHAPES, *seed_parts, "hard_output"),
                choose_rotated(
                    VISUAL_REASONING_REQUIREMENTS,
                    *seed_parts,
                    "hard_reasoning",
                ),
            )
        return (
            current_modality_hint,
            choose_rotated(INPUT_SHAPES, *seed_parts, "hard_input"),
            choose_rotated(OUTPUT_SHAPES, *seed_parts, "hard_output"),
            choose_rotated(REASONING_REQUIREMENTS, *seed_parts, "hard_reasoning"),
        )

    @staticmethod
    def blocked_tags_for_mode(mode_info: PlannerModeInfo) -> list[str]:
        if mode_info.resolved_task_mode == "text":
            return TEXT_MODE_BLOCKED_TAGS
        return []

    def modality_hint(
        self,
        *,
        focus: str,
        is_exploration: bool,
        seed_parts: list[Any],
        mode_info: PlannerModeInfo,
        generation_intent: str = "normal",
        task_shape_profile: dict[str, Any] | None = None,
    ) -> str:
        if generation_intent in {"hard", "validation"}:
            shape_id = (task_shape_profile or {}).get("shape_id")
            if focus == ASSISTANT_QA_SHAPE_ID or shape_id == ASSISTANT_QA_SHAPE_ID:
                return "text_assistant_qa"
            if focus == CONTEXTUAL_QA_SHAPE_ID or shape_id == CONTEXTUAL_QA_SHAPE_ID:
                return "text_contextual_qa"

        if mode_info.resolved_task_mode == "visual":
            return "visual_generation"
        if mode_info.resolved_task_mode == "text":
            return "example_aligned_text"

        if focus in VISUAL_TAGS:
            return "visual_generation"
        if is_exploration and stable_index([*seed_parts, "mixed_modality"], 3) == 0:
            return "visual_generation"
        return "example_aligned_text"

    def planner_state_payload(self) -> dict[str, Any]:
        mode_info = self.resolve_task_mode()
        task_shape_profile = self.task_shape_profile(mode_info)
        return {
            **mode_info.as_payload(),
            "blocked_tags": self.blocked_tags_for_mode(mode_info),
            "mode_source": (
                "explicit_task_mode"
                if mode_info.task_mode != "auto"
                else "original_user_examples"
            ),
            "task_shape_profile": task_shape_profile,
        }

    def recent_instructions(self, limit: int = 3) -> list[str]:
        seen: list[str] = []
        for row in reversed(self.ledger[-25:]):
            text = compact_text(row.instruct)
            if text and text not in seen:
                seen.append(text)
            if len(seen) >= limit:
                break
        return seen

    def recent_sample_shapes(self, limit: int = 3) -> list[str]:
        seen: list[str] = []
        for row in reversed(self.ledger[-25:]):
            text = compact_text(row.sample)
            if text and text not in seen:
                seen.append(text)
            if len(seen) >= limit:
                break
        return seen

    def example_shape_signal(self, example: User_Example | None) -> str:
        if example is None:
            return "No user example available."
        parts = []
        if example.mode:
            parts.append(f"mode={example.mode}")
        if example.instruct:
            parts.append("has_instruction=true")
        if example.context:
            parts.append("has_context=true")
        sample_text = example.context or example.sample
        if sample_text:
            parts.append(f"sample_words={len(sample_text.split())}")
        return "; ".join(parts) or "plain input-output example"
