from enum import Enum
from typing import Any, Literal
from pydantic import BaseModel, Field, model_validator

INSTRUCTOR_RECIPE_SIZE = 10


def validate_instructor_recipe_size(batch_size: int) -> int:
    """
    Instructor_Recipe is intentionally fixed to 10 slots.

    Keep this guard near the schema so all callers produce the same error and
    do not drift from the pydantic length constraints below.
    """
    if batch_size != INSTRUCTOR_RECIPE_SIZE:
        raise ValueError(
            "Instructor_Recipe requires exactly "
            f"{INSTRUCTOR_RECIPE_SIZE} slots, got {batch_size}."
        )
    return batch_size


class CommandQuery(str, Enum):
    """
    It is slot action type
    """
    GENERATE = "generate"
    RECYCLE = "recycle"

class Vault_Analytics(BaseModel):
    """
    Lightweight summary of the current failure vault.
    This is the tiny telemetry block sent to the Instructor API.
    """
    avg_token_length: float = 0.0
    most_problematic_category: str | None = None
    total_failures_tracked: int = 0
    active_failures: int = 0
    usable_failures: int = 0
    cooling_down_failures: int = 0
    exhausted_failures: int = 0
    retired_failures: int = 0
    avg_loss: float = 0.0
    minimum_entry_loss: float = 0.0

class failure_vault_item(BaseModel):
    """
    Compact reference to one high-loss ledger record.
    It stores only metadata, not the raw sample, to preserve token economy.
    """
    ledger_id: int
    tag: str
    token_length: int
    loss: float
    recycle_count: int = 0
    last_recycled_round: int | None = None
    active: bool = True
    retired_reason: str | None = None
    first_seen_round: int | None = None
    last_seen_round: int | None = None
    last_loss: float | None = None
    best_recycle_loss: float | None = None

class Failure_VaultDB(BaseModel):
    """
    List of features of vault recycle samples
    """
    top_failures: list[failure_vault_item] = Field(default_factory=list)
    analytics: Vault_Analytics = Field(default_factory=Vault_Analytics)

class Evaluation_Output(BaseModel):
    """
    output of evaluator. only part that is fram evaluator is loss and others are from studen and instructor
    """
    sample: str
    instruct: str
    gold_summary: str | None = None
    student_ans: str
    token_length: int
    loss: float
    tag: str
    source: Literal["generated", "recycled"]
    source_ledger_id: int | None = None

class Training_Sample(BaseModel):
    """
    One batch item prepared for local student training.
    It may come from API generation or from recycling a local ledger failure.
    """
    sample: str
    instruct: str
    gold_summary: str | None = None
    source: Literal["generated", "recycled"]
    source_ledger_id: int | None = None
    tag: str

class User_Example(BaseModel):
    """
    A user-provided example stored in pipeline state for generation context.
    """
    sample: str = ""
    output: str
    instruct: str | None = None
    context: str | None = None
    mode: Literal["plain", "contextual"] = "plain"

    @model_validator(mode="after")
    def normalize_example_shape(self):
        if self.context and not self.sample:
            self.sample = self.context

        if self.instruct or self.context:
            self.mode = "contextual"
            self.context = self.context or self.sample
        else:
            self.mode = "plain"

        return self

class learning_record(BaseModel):
    """
    One permanent record saved in data/LearningData.json after local evaluation.
    """
    ledger_id: int
    sample: str
    instruct: str
    gold_summary: str | None = None
    student_ans: str = ""
    token_length: int
    loss: float
    tag: str
    round_id: int
    source: Literal["generated", "recycled"] | None = None
    source_ledger_id: int | None = None

class Instructor_Slot(BaseModel):
    """
    One slot in the 10-slot Instructor recipe.
    It tells the orchestrator whether to generate a new sample
    or recycle an existing failure from the local ledger.
    """
    slot_id: int = Field(ge=1, le=INSTRUCTOR_RECIPE_SIZE)
    command: CommandQuery
    tag: str
    failure_ledger_id: int | None = None

    @model_validator(mode="after")
    def recycle_requires_failure_id(self):
        if self.command == CommandQuery.RECYCLE and self.failure_ledger_id is None:
            raise ValueError("recycle slots must include failure_ledger_id")
        return self

class Instructor_Recipe(BaseModel):
    """
    The full fixed-size strategy returned by the Instructor.

    The prompt/schema currently supports exactly INSTRUCTOR_RECIPE_SIZE slots.
    Use validate_instructor_recipe_size() when accepting a configurable batch
    size before constructing recipes.
    """
    slots: list[Instructor_Slot] = Field(
        min_length=INSTRUCTOR_RECIPE_SIZE,
        max_length=INSTRUCTOR_RECIPE_SIZE,
    )

class Pipeline_State(BaseModel):
    """
    It create a pipeline to let program know where it is and recover if anything goes wrong
    """
    current_round: int = 0
    consecutive_instructor_no_recycle: int = 0
    example_format: Literal["auto", "plain", "contextual"] = "auto"
    task_mode: Literal["auto", "text", "visual", "mixed"] = "auto"
    resolved_task_mode: Literal["text", "visual", "mixed"] = "text"
    user_examples: list[User_Example] = Field(default_factory=list)
    learning_tag_counts: dict[str, int] = Field(default_factory=dict)
    active_recipe: list[Instructor_Slot] = Field(default_factory=list)
    current_training_batch: list[Training_Sample] = Field(default_factory=list)
    processed_samples: list[learning_record] = Field(default_factory=list)
    recycled_failure_ledger_ids: list[int] = Field(default_factory=list)
    planner_state: dict[str, Any] = Field(default_factory=dict)
    planner_history: list[dict[str, Any]] = Field(default_factory=list)
    task_shape_profile: dict[str, Any] = Field(default_factory=dict)
    validation_weakness_profile: dict[str, Any] = Field(default_factory=dict)
    learning_pressure_state: dict[str, Any] = Field(default_factory=dict)
    round_health_history: list[dict[str, Any]] = Field(default_factory=list)
    best_round_id: int | None = None
    best_eval_loss: float | None = None
    best_adapter_path: str | None = None
    raw_best_round_id: int | None = None
    raw_best_eval_loss: float | None = None
    raw_best_adapter_path: str | None = None
    last_rollback_decision: dict[str, Any] = Field(default_factory=dict)
    bad_round_streak: int = 0
    base_adapter_override_path: str | None = None
    section_id: str = ""

class Section_Index(BaseModel):
    """
    Global index that tracks available sections and the active section.
    """
    current_section_id: str | None = None
    sections: list[str] = Field(default_factory=list)

class Orchestrator_Input(BaseModel):
    state: Pipeline_State = Field(default_factory=Pipeline_State)
    ledger: list[learning_record] = Field(default_factory=list)
    failure_vault: Failure_VaultDB = Field(default_factory=Failure_VaultDB)

class Orchestrator_Output(BaseModel):
    state: Pipeline_State
    recipe: Instructor_Recipe
    training_batch: list[Training_Sample]
    evaluations: list[Evaluation_Output]
    updated_ledger: list[learning_record]
    updated_failure_vault: Failure_VaultDB
