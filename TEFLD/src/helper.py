"""
Need to create functions for updating states and not creating whole file again
"""

from collections.abc import Iterable
from datetime import datetime
from enum import Enum
import hashlib
import json
from pathlib import Path
from typing import TypeVar
import os
import sys
from typing import Any

from pydantic import BaseModel

from .dataschema import (
    Instructor_Recipe,
    Pipeline_State,
    Section_Index,
    Training_Sample,
    User_Example,
    learning_record,
    Failure_VaultDB,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
PROMPT_DIR = PROJECT_ROOT / "src" / "prompts"
SECTIONS_DIR = DATA_DIR / "sections"
SECTIONS_INDEX_PATH = DATA_DIR / "sections_index.json"
PROMPT_CACHE_DIR = DATA_DIR / "prompt_cache"
INSTRUCTION_RESPONSE_CACHE_DIR = PROMPT_CACHE_DIR / "instruction_responses"

LEDGER_FILENAME = "OverAllData.json"
VAULT_FILENAME = "failure_vault.json"
STATE_FILENAME = "pipeline_state.json"
DEBUG_LOG_FILENAME = "debug_log.jsonl"
LEARNING_FILENAME = "LearningData.json"

ModelT = TypeVar("ModelT", bound=BaseModel)


def configure_utf8_stdio() -> None:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8", errors="replace")
            except Exception:
                pass


configure_utf8_stdio()

def _import_openai_client():
    try:
        from openai import OpenAI
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Instructor generation requires 'openai'. "
            "Install the API dependency before calling generation methods."
        ) from exc

    return OpenAI

def read_json_file(path: Path, default):
    if not path.exists() or path.stat().st_size == 0:
        return default

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json_file(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def json_safe(value):
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_safe(item) for item in value]
    return value


def load_model_file(path: Path, model_type: type[ModelT], default) -> ModelT:
    raw_data = read_json_file(path, default)
    return model_type.model_validate(raw_data)


def write_model_file(path: Path, model: BaseModel) -> None:
    write_json_file(path, model.model_dump(mode="json"))


def load_model_list_file( path: Path, model_type: type[ModelT], default, filename: str) -> list[ModelT]:
    raw_rows = read_json_file(path, default)

    if not isinstance(raw_rows, list):
        raise ValueError(f"{filename} must contain a JSON array.")

    return [model_type.model_validate(row) for row in raw_rows]


def write_model_list_file(path: Path, records: Iterable[BaseModel]) -> None:
    write_json_file(
        path,
        [record.model_dump(mode="json") for record in records],
    )


def write_text_file(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def stable_json_hash(payload: dict[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def instruction_response_cache_key(
    *,
    prompt: str,
    model_id: str,
    cache_version: str,
    temperature: float = 0.7,
) -> str:
    normalized_prompt = prompt.replace("\r\n", "\n").strip()
    return stable_json_hash(
        {
            "kind": "instruction_response",
            "cache_version": cache_version,
            "model_id": model_id,
            "temperature": temperature,
            "prompt": normalized_prompt,
        }
    )


def get_cached_instruction_response(
    *,
    prompt: str,
    model_id: str,
    cache_version: str,
    temperature: float = 0.7,
) -> str | None:
    cache_key = instruction_response_cache_key(
        prompt=prompt,
        model_id=model_id,
        cache_version=cache_version,
        temperature=temperature,
    )
    cache_path = INSTRUCTION_RESPONSE_CACHE_DIR / f"{cache_key}.json"
    raw_cache = read_json_file(cache_path, {})
    response = raw_cache.get("response") if isinstance(raw_cache, dict) else None
    return response.strip() if isinstance(response, str) and response.strip() else None


def save_cached_instruction_response(
    *,
    prompt: str,
    response: str,
    model_id: str,
    cache_version: str,
    temperature: float = 0.7,
    metadata: dict[str, Any] | None = None,
) -> str:
    cache_key = instruction_response_cache_key(
        prompt=prompt,
        model_id=model_id,
        cache_version=cache_version,
        temperature=temperature,
    )
    write_json_file(
        INSTRUCTION_RESPONSE_CACHE_DIR / f"{cache_key}.json",
        {
            "cache_key": cache_key,
            "cache_version": cache_version,
            "model_id": model_id,
            "temperature": temperature,
            "prompt": prompt.replace("\r\n", "\n").strip(),
            "response": response.strip(),
            "metadata": metadata or {},
        },
    )
    return cache_key


def load_sections_index() -> Section_Index:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    return load_model_file(SECTIONS_INDEX_PATH, Section_Index, {})


def save_sections_index(index: Section_Index) -> None:
    write_model_file(SECTIONS_INDEX_PATH, index)


def get_learningdata_path() -> Path:
    return DATA_DIR / LEARNING_FILENAME


def save_learningdata(data: list[learning_record]) -> None:
    write_model_list_file(get_learningdata_path(), data)


def load_learningdata() -> list[learning_record]:
    return load_model_list_file(
        get_learningdata_path(),
        learning_record,
        [],
        LEARNING_FILENAME,
    )

def create_client() -> Any:
        api_key = (
            # os.environ.get("DEEPSEEK_API_KEY")
            # or os.environ.get("DEEPSEEK_API")
            os.environ.get("OPENAI_API_KEY")
        )
        if not api_key:
            raise ValueError("DEEPSEEK_API_KEY environment variable is missing.")

        OpenAI = _import_openai_client()
        client = OpenAI(api_key=api_key)
        print("[api] client ready")
        return client

def call_client( client: Any, prompt: str, model_id:Any) -> str:
    response = client.chat.completions.create(
        model=model_id,
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
        temperature=0.7,
    )

    content = response.choices[0].message.content
    if content is None:
        raise ValueError("DeepSeek returned an empty response.")
    
    striped = content.strip()

    print(f"[api] received response model={model_id} chars={len(striped)}")

    return striped

def get_section_dir(section_id: str) -> Path:
    return SECTIONS_DIR / section_id


def get_section_file_path(section_id: str, filename: str) -> Path:
    return get_section_dir(section_id) / filename


def get_section_debug_log_path(section_id: str) -> Path:
    return get_section_file_path(section_id, DEBUG_LOG_FILENAME)


def get_section_ledger_path(section_id: str) -> Path:
    return get_section_file_path(section_id, LEDGER_FILENAME)


def get_section_vault_path(section_id: str) -> Path:
    return get_section_file_path(section_id, VAULT_FILENAME)


def get_section_state_path(section_id: str) -> Path:
    return get_section_file_path(section_id, STATE_FILENAME)


def load_section_ledger(section_id: str) -> list[learning_record]:
    return load_model_list_file(
        get_section_ledger_path(section_id),
        learning_record,
        [],
        LEDGER_FILENAME,
    )


def load_section_vault(section_id: str) -> dict:
    raw_vault = read_json_file(get_section_vault_path(section_id), {})

    if not isinstance(raw_vault, dict):
        raise ValueError(f"{VAULT_FILENAME} must contain a JSON object.")

    return raw_vault


def get_section_prompt_dir(section_id: str) -> Path:
    return get_section_dir(section_id) / "prompts"


def get_section_data_prompts_dir(section_id: str) -> Path:
    return get_section_prompt_dir(section_id) / "data_prompts"


def get_section_data_responses_dir(section_id: str) -> Path:
    return get_section_prompt_dir(section_id) / "data_responses"


def normalize_tag_for_path(tag: str) -> str:
    normalized = "".join(
        char.lower() if char.isalnum() else "_"
        for char in tag.strip()
    ).strip("_")
    return normalized or "untagged"


def get_section_tagged_prompt_path( section_id: str, default_filename: str, tagged_dirname: str, tagged_filename_prefix: str,
        tag: str | None = None) -> Path:
    prompt_dir = get_section_prompt_dir(section_id)

    if tag is None:
        return prompt_dir / default_filename

    return (
        prompt_dir
        / tagged_dirname
        / f"{tagged_filename_prefix}_{normalize_tag_for_path(tag)}.txt"
    )


def get_section_instruction_prompt_path(section_id: str, tag: str | None = None) -> Path:
    return get_section_tagged_prompt_path(
        section_id,
        "instruction_prompt.txt",
        "instruction_prompts",
        "instruction_prompt",
        tag,
    )


def get_section_generated_instruction_path(section_id: str, tag: str | None = None) -> Path:
    return get_section_tagged_prompt_path(
        section_id,
        "generated_instruction.txt",
        "generated_instructions",
        "generated_instruction",
        tag,
    )


def _normalize_pipeline_state(
    raw_state: dict,
    section_id: str | None = None,
) -> Pipeline_State:
    raw_state = dict(raw_state)

    if isinstance(raw_state.get("user_examples"), dict):
        raw_state["user_examples"] = [raw_state["user_examples"]]

    raw_state.setdefault("learning_tag_counts", {})
    if section_id is not None:
        raw_state["section_id"] = raw_state.get("section_id") or section_id

    return Pipeline_State.model_validate(raw_state)


def _has_legacy_data() -> bool:
    legacy_paths = [
        DATA_DIR / LEDGER_FILENAME,
        DATA_DIR / VAULT_FILENAME,
        DATA_DIR / STATE_FILENAME,
    ]
    return any(path.exists() and path.stat().st_size > 0 for path in legacy_paths)


def _initialize_section_files(section_id: str, migrate_legacy: bool = False) -> None:
    section_dir = get_section_dir(section_id)
    prompt_dir = get_section_prompt_dir(section_id)
    data_prompt_dir = get_section_data_prompts_dir(section_id)
    data_response_dir = get_section_data_responses_dir(section_id)

    section_dir.mkdir(parents=True, exist_ok=True)
    prompt_dir.mkdir(parents=True, exist_ok=True)
    data_prompt_dir.mkdir(parents=True, exist_ok=True)
    data_response_dir.mkdir(parents=True, exist_ok=True)

    if migrate_legacy:
        legacy_ledger = read_json_file(DATA_DIR / LEDGER_FILENAME, [])
        legacy_vault = read_json_file(DATA_DIR / VAULT_FILENAME, {})
        legacy_state = read_json_file(DATA_DIR / STATE_FILENAME, {})

        write_json_file(get_section_ledger_path(section_id), legacy_ledger)
        write_json_file(get_section_vault_path(section_id), legacy_vault)
        write_model_file(
            get_section_state_path(section_id),
            _normalize_pipeline_state(legacy_state, section_id),
        )
        return

    write_json_file(get_section_ledger_path(section_id), [])
    write_json_file(get_section_vault_path(section_id), {})
    write_model_file(
        get_section_state_path(section_id),
        Pipeline_State(section_id=section_id),
    )


def _next_section_id(index: Section_Index) -> str:
    existing_ids = set(index.sections)

    if SECTIONS_DIR.exists():
        existing_ids.update(
            path.name
            for path in SECTIONS_DIR.iterdir()
            if path.is_dir() and path.name.startswith("section_")
        )

    next_number = 1
    while f"section_{next_number:03d}" in existing_ids:
        next_number += 1

    return f"section_{next_number:03d}"


def create_new_section() -> str:
    SECTIONS_DIR.mkdir(parents=True, exist_ok=True)

    index = load_sections_index()
    section_id = _next_section_id(index)

    migrate_legacy = not index.sections and _has_legacy_data()
    _initialize_section_files(section_id, migrate_legacy=migrate_legacy)

    index.sections.append(section_id)
    index.current_section_id = section_id
    save_sections_index(index)

    return section_id


def choose_section(section_id: str) -> str:
    section_dir = get_section_dir(section_id)
    if not section_dir.exists():
        raise ValueError(f"Section does not exist: {section_id}")

    index = load_sections_index()
    if section_id not in index.sections:
        index.sections.append(section_id)

    index.current_section_id = section_id
    save_sections_index(index)
    return section_id


def get_current_section_id() -> str | None:
    return load_sections_index().current_section_id


def ensure_current_section() -> str:
    current_section_id = get_current_section_id()

    if current_section_id and get_section_dir(current_section_id).exists():
        return current_section_id

    return create_new_section()


def resolve_section_id(section_id: str | None = None) -> str:
    return section_id or ensure_current_section()


def append_section_debug_log(
    section_id: str | None,
    event_type: str,
    payload: dict[str, Any] | None = None,
) -> None:
    active_section_id = resolve_section_id(section_id)
    log_path = get_section_debug_log_path(active_section_id)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    event = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "section_id": active_section_id,
        "event_type": event_type,
        **json_safe(payload or {}),
    }
    with open(log_path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(event, ensure_ascii=False) + "\n")


def load_pipeline_state(section_id: str | None = None) -> Pipeline_State:
    active_section_id = resolve_section_id(section_id)
    state_path = get_section_state_path(active_section_id)

    raw_state = read_json_file(state_path, {})
    return _normalize_pipeline_state(raw_state, active_section_id)


def save_pipeline_state(state: Pipeline_State, section_id: str | None = None) -> None:
    active_section_id = resolve_section_id(section_id)
    state_path = get_section_state_path(active_section_id)
    write_model_file(state_path, state)

def save_failure_vault(vault: Failure_VaultDB, section_id: str | None = None) -> Failure_VaultDB:
    active_section_id = resolve_section_id(section_id)
    write_json_file(
        get_section_vault_path(active_section_id),
        vault.model_dump(mode="json"),
    )
    return vault

def save_pipeline_training_batch(training_batch: list[Training_Sample], recipe: Instructor_Recipe, section_id: str | None = None) -> Pipeline_State:
    active_section_id = resolve_section_id(section_id)
    state = load_pipeline_state(active_section_id)
    state.active_recipe = recipe.slots
    state.current_training_batch = training_batch
    save_pipeline_state(state, active_section_id)
    return state


def save_pipeline_recipe(recipe: Instructor_Recipe, section_id: str | None = None) -> Pipeline_State:
    active_section_id = resolve_section_id(section_id)
    state = load_pipeline_state(active_section_id)
    state.active_recipe = recipe.slots
    save_pipeline_state(state, active_section_id)
    return state


def get_user_examples(section_id: str | None = None) -> list[User_Example]:
    state = load_pipeline_state(section_id)
    return state.user_examples


def ensure_user_example(section_id: str | None = None) -> Pipeline_State:
    active_section_id = resolve_section_id(section_id)
    state = load_pipeline_state(active_section_id)

    if state.user_examples:
        print("Pipeline already has user example.")
        return state

    example_input = input("Enter example input: ")
    example_output = input("Enter example output: ")

    state.user_examples.append(
        User_Example(
            sample=example_input,
            output=example_output,
        )
    )
    save_pipeline_state(state, active_section_id)
    print("User example saved in pipeline state.")
    return state


def get_learning_tag_counts(section_id: str | None = None) -> dict[str, int]:
    state = load_pipeline_state(section_id)
    return state.learning_tag_counts


def sync_learning_tag_counts_from_ledger(section_id: str | None = None) -> Pipeline_State:
    active_section_id = resolve_section_id(section_id)
    state = load_pipeline_state(active_section_id)
    ledger_path = get_section_ledger_path(active_section_id)

    ledger_rows = read_json_file(ledger_path, [])

    ledger_counts: dict[str, int] = {}
    for row in ledger_rows:
        tag = row.get("tag")
        if not tag:
            continue
        ledger_counts[tag] = ledger_counts.get(tag, 0) + 1

    for tag, count in ledger_counts.items():
        state.learning_tag_counts[tag] = max(
            state.learning_tag_counts.get(tag, 0),
            count,
        )

    save_pipeline_state(state, active_section_id)
    return state


def update_learning_tag_count(tag: str, section_id: str | None = None) -> Pipeline_State:
    active_section_id = resolve_section_id(section_id)
    state = load_pipeline_state(active_section_id)
    state.learning_tag_counts[tag] = state.learning_tag_counts.get(tag, 0) + 1
    save_pipeline_state(state, active_section_id)
    return state


def format_learning_tag_counts(tag_counts: dict[str, int]) -> str:
    if not tag_counts:
        return "No generated tags have been recorded yet."

    return "\n".join(
        f"- {tag}: {count}"
        for tag, count in sorted(tag_counts.items())
    )


def load_prompt_template(template_name: str) -> str:
    prompt_path = PROMPT_DIR / template_name

    with open(prompt_path, "r", encoding="utf-8") as f:
        return f.read()


def render_prompt_template(template: str, replacements: dict[str, object]) -> str:
    rendered = template

    for placeholder, value in replacements.items():
        rendered = rendered.replace(f"***{placeholder}***", str(value))

    if "***" in rendered:
        raise ValueError("Prompt template still has unreplaced placeholders.")

    return rendered


def save_instruction_prompt(prompt: str, section_id: str | None = None, tag: str | None = None) -> Path:
    active_section_id = resolve_section_id(section_id)
    prompt_path = get_section_instruction_prompt_path(active_section_id, tag)
    return write_text_file(prompt_path, prompt)


def get_cached_instruction(section_id: str | None = None, tag: str | None = None) -> str | None:
    active_section_id = resolve_section_id(section_id)
    instruction_path = get_section_generated_instruction_path(active_section_id, tag)

    if not instruction_path.exists() or instruction_path.stat().st_size == 0:
        return None

    return instruction_path.read_text(encoding="utf-8").strip()


def save_generated_instruction(instruction: str, section_id: str | None = None, tag: str | None = None) -> Path:
    active_section_id = resolve_section_id(section_id)
    instruction_path = get_section_generated_instruction_path(active_section_id, tag)
    return write_text_file(instruction_path, instruction)


def save_data_prompt(prompt: str, section_id: str | None = None) -> Path:
    active_section_id = resolve_section_id(section_id)
    data_prompt_dir = get_section_data_prompts_dir(active_section_id)
    data_prompt_dir.mkdir(parents=True, exist_ok=True)

    next_number = len(list(data_prompt_dir.glob("data_prompt_*.txt"))) + 1
    prompt_path = data_prompt_dir / f"data_prompt_{next_number:03d}.txt"
    return write_text_file(prompt_path, prompt)


def save_data_response(
    response: str,
    section_id: str | None = None,
    prompt_path: Path | None = None,
) -> Path:
    active_section_id = resolve_section_id(section_id)
    response_dir = get_section_data_responses_dir(active_section_id)
    response_dir.mkdir(parents=True, exist_ok=True)

    if prompt_path is not None:
        suffix = prompt_path.stem.replace("data_prompt_", "")
        response_path = response_dir / f"data_response_{suffix}.txt"
    else:
        next_number = len(list(response_dir.glob("data_response_*.txt"))) + 1
        response_path = response_dir / f"data_response_{next_number:03d}.txt"

    return write_text_file(response_path, response)
