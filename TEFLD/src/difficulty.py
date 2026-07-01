from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass
import re
from typing import Any


DIFFICULTY_LEVELS = ["easy", "medium", "medium_hard", "hard"]
DIFFICULTY_CONFIG_FILENAME = "difficulty_config.json"
DIFFICULTY_MAX_ATTEMPTS = 2

DEFAULT_DIFFICULTY_CONFIG: dict[str, Any] = {
    "global": {
        "easy": {
            "distractors": 0,
            "constraints": 0,
            "reasoning": "direct",
            "output_format": "free",
            "target_output_tokens": 80,
        },
        "medium": {
            "distractors": 1,
            "constraints": 1,
            "reasoning": "single_hop",
            "output_format": "free",
            "target_output_tokens": 120,
        },
        "medium_hard": {
            "distractors": 1,
            "constraints": 2,
            "reasoning": "comparison",
            "output_format": "structured",
            "target_output_tokens": 150,
        },
        "hard": {
            "distractors": 2,
            "constraints": 2,
            "reasoning": "multi_hop",
            "output_format": "strict",
            "target_output_tokens": 180,
        },
    },
    "shapes": {
        "assistant_qa": {},
        "contextual_qa": {},
        "structured_text_task": {},
        "visual_generation": {},
    },
    "tags": {},
}

DISTRACTOR_MARKERS = {
    "however",
    "but",
    "although",
    "except",
    "instead",
    "irrelevant",
    "plausible",
    "distractor",
    "ignore",
    "not needed",
}

CONSTRAINT_MARKERS = {
    "must",
    "only",
    "exactly",
    "include",
    "exclude",
    "without",
    "at least",
    "at most",
    "both",
    "compare",
    "return",
    "json",
    "table",
    "label",
}

ARITHMETIC_MARKERS = {
    "calculate",
    "sum",
    "total",
    "difference",
    "average",
    "cheaper",
    "cost",
    "minutes",
    "hours",
    "percent",
    "%",
    "+",
    "-",
}

COMPARISON_MARKERS = {
    "compare",
    "which",
    "cheaper",
    "earlier",
    "later",
    "first",
    "best",
    "more",
    "less",
}

MULTI_HOP_MARKERS = {
    "after",
    "before",
    "then",
    "because",
    "given both",
    "using the",
    "based on",
    "infer",
    "deduce",
}


@dataclass(frozen=True)
class DifficultyScore:
    context_length: int
    distractor_count: int
    output_strictness: str
    reasoning_type: str
    constraint_count: int

    def as_payload(self) -> dict[str, Any]:
        return asdict(self)


def default_difficulty_config() -> dict[str, Any]:
    return deepcopy(DEFAULT_DIFFICULTY_CONFIG)


def load_difficulty_config(section_id: str | None = None) -> dict[str, Any]:
    if not section_id:
        return default_difficulty_config()

    from .helper import get_section_file_path, read_json_file, write_json_file

    config_path = get_section_file_path(section_id, DIFFICULTY_CONFIG_FILENAME)
    if not config_path.exists() or config_path.stat().st_size == 0:
        config = default_difficulty_config()
        write_json_file(config_path, config)
        return config

    raw_config = read_json_file(config_path, {})
    if not isinstance(raw_config, dict):
        return default_difficulty_config()

    config = default_difficulty_config()
    for key in ("global", "shapes", "tags"):
        if isinstance(raw_config.get(key), dict):
            config[key].update(raw_config[key])
    return config


def difficulty_index(level: str | None) -> int:
    if level not in DIFFICULTY_LEVELS:
        return 0
    return DIFFICULTY_LEVELS.index(level)


def shift_difficulty(level: str, delta: int) -> str:
    index = max(0, min(len(DIFFICULTY_LEVELS) - 1, difficulty_index(level) + delta))
    return DIFFICULTY_LEVELS[index]


def difficulty_budget_for(
    *,
    difficulty: str,
    tag: str | None = None,
    shape_id: str | None = None,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    active_config = config or DEFAULT_DIFFICULTY_CONFIG
    level = difficulty if difficulty in DIFFICULTY_LEVELS else "medium"
    budget = dict(active_config.get("global", {}).get(level, {}))

    for section_name, key in (("shapes", shape_id), ("tags", tag)):
        if not key:
            continue
        section = active_config.get(section_name, {})
        overrides = section.get(key, {}) if isinstance(section, dict) else {}
        level_overrides = overrides.get(level, {}) if isinstance(overrides, dict) else {}
        budget.update(level_overrides)

    return budget


def format_difficulty_budget(budget: dict[str, Any]) -> str:
    if not budget:
        return "No explicit difficulty budget."
    return (
        "Difficulty budget: "
        f"distractors={budget.get('distractors', 0)}, "
        f"constraints={budget.get('constraints', 0)}, "
        f"reasoning={budget.get('reasoning', 'direct')}, "
        f"output_format={budget.get('output_format', 'free')}, "
        f"target_output_tokens={budget.get('target_output_tokens', 'unspecified')}."
    )


def compact_lower(text: str | None) -> str:
    return re.sub(r"\s+", " ", text or "").strip().lower()


def count_marker_hits(text: str, markers: set[str]) -> int:
    lowered = compact_lower(text)
    return sum(1 for marker in markers if marker in lowered)


def count_distractor_markers(text: str | None) -> int:
    return count_marker_hits(text or "", DISTRACTOR_MARKERS)


def count_constraints(instruct: str | None) -> int:
    text = compact_lower(instruct)
    count = count_marker_hits(text, CONSTRAINT_MARKERS)
    count += max(0, text.count(" and ") - 1)
    return count


def classify_output_format(output: str | None) -> str:
    text = (output or "").strip()
    lowered = text.lower()
    if not text:
        return "free"
    if text.startswith(("{", "[")) or "```json" in lowered:
        return "strict"
    if "|" in text and "\n" in text:
        return "strict"
    if "\n-" in text or "\n1." in text or ":" in text:
        return "structured"
    return "free"


def classify_reasoning(instruct: str | None, output: str | None = None) -> str:
    text = compact_lower(" ".join(part for part in (instruct, output) if part))
    if count_marker_hits(text, ARITHMETIC_MARKERS):
        return "arithmetic"
    if count_marker_hits(text, MULTI_HOP_MARKERS) >= 2:
        return "multi_hop"
    if count_marker_hits(text, COMPARISON_MARKERS):
        return "comparison"
    if count_marker_hits(text, MULTI_HOP_MARKERS):
        return "single_hop"
    return "direct"


def score_generated_difficulty(
    *,
    sample: str,
    instruct: str,
    output: str | None,
    context: str | None = None,
) -> DifficultyScore:
    context_text = context or sample
    return DifficultyScore(
        context_length=len((context_text or "").split()),
        distractor_count=count_distractor_markers(context_text),
        output_strictness=classify_output_format(output),
        reasoning_type=classify_reasoning(instruct, output),
        constraint_count=count_constraints(instruct),
    )


def observed_difficulty(score: DifficultyScore) -> str:
    if (
        score.output_strictness == "strict"
        or score.reasoning_type in {"multi_hop", "arithmetic"}
    ) and (score.constraint_count >= 2 or score.distractor_count >= 2):
        return "hard"
    if (
        score.reasoning_type in {"comparison", "arithmetic", "multi_hop"}
        or score.constraint_count >= 2
        or score.context_length >= 80
    ):
        return "medium_hard"
    if (
        score.constraint_count >= 1
        or score.distractor_count >= 1
        or score.output_strictness == "structured"
        or score.context_length >= 40
    ):
        return "medium"
    return "easy"


def difficulty_miss_reason(
    requested: str | None,
    observed: str | None,
    score: DifficultyScore,
) -> str | None:
    if not requested or not observed:
        return None

    requested_index = difficulty_index(requested)
    observed_index = difficulty_index(observed)
    if requested_index - observed_index >= 2:
        return (
            f"requested {requested} but observed {observed}: "
            f"reasoning={score.reasoning_type}, "
            f"distractors={score.distractor_count}, "
            f"constraints={score.constraint_count}, "
            f"output_format={score.output_strictness}"
        )

    if (
        requested == "hard"
        and score.reasoning_type == "direct"
        and score.distractor_count == 0
        and score.constraint_count < 2
    ):
        return (
            "requested hard but sample is direct with no distractors "
            "and fewer than two constraints"
        )

    return None
