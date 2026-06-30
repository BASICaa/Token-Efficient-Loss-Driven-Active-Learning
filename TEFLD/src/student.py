from __future__ import annotations

import json
import os
import inspect
from pathlib import Path
from typing import Any

from .dataschema import (
    Evaluation_Output,
    INSTRUCTOR_RECIPE_SIZE,
    Pipeline_State,
    Training_Sample,
)
from .helper import (
    DATA_DIR,
    append_section_debug_log,
    ensure_current_section,
    save_pipeline_state,
    sync_learning_tag_counts_from_ledger,
)

try:
    from dotenv import load_dotenv
except ModuleNotFoundError:

    def load_dotenv(*_args: Any, **_kwargs: Any) -> bool:
        return False


PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_SAVING_DIR = PROJECT_ROOT / "Models"
_base_model_env = os.environ.get("BASE_MODEL_DIR")
BASE_MODEL_DIR = Path(_base_model_env) if _base_model_env else PROJECT_ROOT / "models_cache"

DEFAULT_MODEL_ID = "gemma-4-E2B-it"
DEFAULT_MAX_LENGTH = 512
EXPECTED_BATCH_SIZE = INSTRUCTOR_RECIPE_SIZE
ADAPTER_COMPLETE_FILENAME = "training_complete.json"
BASE_LORA_TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]
GEMMA4_LORA_TARGET_MODULES = [f"{module}.linear" for module in BASE_LORA_TARGET_MODULES]
GEMMA4_LANGUAGE_LORA_TARGET_MODULES = (
    r"model\.language_model\.layers\.\d+\."
    r"(self_attn\.(q_proj|k_proj|v_proj|o_proj)|"
    r"mlp\.(gate_proj|up_proj|down_proj))$"
)

load_dotenv(DATA_DIR / ".env")
load_dotenv()


def missing_dependency_error(package_name: str) -> ModuleNotFoundError:
    return ModuleNotFoundError(
        f"Student model operations require '{package_name}'. "
        "Install the training dependencies before calling train/evaluate methods."
    )


def import_torch():
    try:
        import torch
    except ModuleNotFoundError as exc:
        raise missing_dependency_error("torch") from exc
    return torch


def import_peft():
    try:
        from peft import (
            LoraConfig,
            PeftModel,
            TaskType,
            get_peft_model,
            prepare_model_for_kbit_training,
        )
    except ModuleNotFoundError as exc:
        raise missing_dependency_error("peft") from exc

    return {
        "LoraConfig": LoraConfig,
        "PeftModel": PeftModel,
        "TaskType": TaskType,
        "get_peft_model": get_peft_model,
        "prepare_model_for_kbit_training": prepare_model_for_kbit_training,
    }


def import_transformers():
    try:
        from transformers import (
            AutoModelForCausalLM,
            AutoProcessor,
            AutoTokenizer,
            BitsAndBytesConfig,
            GenerationConfig,
            Trainer,
            TrainingArguments,
            default_data_collator,
        )
    except ModuleNotFoundError as exc:
        raise missing_dependency_error("transformers") from exc

    return {
        "AutoModelForCausalLM": AutoModelForCausalLM,
        "AutoProcessor": AutoProcessor,
        "AutoTokenizer": AutoTokenizer,
        "BitsAndBytesConfig": BitsAndBytesConfig,
        "GenerationConfig": GenerationConfig,
        "Trainer": Trainer,
        "TrainingArguments": TrainingArguments,
        "default_data_collator": default_data_collator,
    }


def import_safetensors_load_file():
    try:
        from safetensors.torch import load_file
    except ModuleNotFoundError as exc:
        raise missing_dependency_error("safetensors") from exc
    return load_file


def trainer_processor_kwargs(trainer_cls: Any, tokenizer: Any) -> dict[str, Any]:
    parameters = inspect.signature(trainer_cls.__init__).parameters
    if "processing_class" in parameters:
        return {"processing_class": tokenizer}
    if "tokenizer" in parameters:
        return {"tokenizer": tokenizer}
    return {}


def is_gemma4_model_id(model_id: str | Path) -> bool:
    normalized = str(model_id).replace("\\", "/").lower()
    return "gemma-4" in normalized or "gemma4" in normalized


class TrainyModel:
    """
    Student model connector.

    Construction is intentionally lightweight so an orchestrator/evaluator can
    import and instantiate this class without loading torch or the base model.
    Heavy ML dependencies are imported only by train/evaluate/generation paths.
    """

    def __init__(self, model_name: str = DEFAULT_MODEL_ID, *, expected_batch_size: int | None = EXPECTED_BATCH_SIZE, max_length: int = DEFAULT_MAX_LENGTH) -> None:
        self.model_id = model_name
        self.expected_batch_size = expected_batch_size
        self.max_length = max_length

        self.section_id: str = ensure_current_section()
        self.pipeline: Pipeline_State = sync_learning_tag_counts_from_ledger(
            self.section_id
        )
        self.samples: list[Training_Sample] = list(
            self.pipeline.current_training_batch
        )
        self.current_round = self.pipeline.current_round

        self.base_model_location = self.resolve_base_model_location()
        self.output_dir = self.round_adapter_dir(self.current_round)

        self.tokenized_batch: list[dict[str, Any]] = []
        self.tokenizer: Any = None
        self.model: Any = None
        self.trainer: Any = None
        self.used_base_adapter_override = False

    def refresh_from_pipeline(self) -> Pipeline_State:
        self.pipeline = sync_learning_tag_counts_from_ledger(self.section_id)
        self.samples = list(self.pipeline.current_training_batch)
        self.current_round = self.pipeline.current_round
        self.output_dir = self.round_adapter_dir(self.current_round)
        return self.pipeline

    def lenSamples(self) -> int:
        return len(self.samples)

    def getitem(self, index: int) -> dict[str, Any]:
        tokenizer = self.load_tokenizer()
        return self.encode_training_sample(
            self.samples[index],
            tokenizer,
            self.max_length,
        )

    def round_adapter_dir(self, round_id: int) -> Path:
        return MODEL_SAVING_DIR / self.section_id / f"round_{round_id:03d}"

    def adapter_complete_marker(self, adapter_dir: Path | str | None = None) -> Path:
        active_adapter_dir = self.output_dir if adapter_dir is None else Path(adapter_dir)
        return active_adapter_dir / ADAPTER_COMPLETE_FILENAME

    def adapter_is_complete(self, adapter_dir: Path | str | None = None) -> bool:
        active_adapter_dir = self.output_dir if adapter_dir is None else Path(adapter_dir)
        return (
            (active_adapter_dir / "adapter_config.json").exists()
            and self.adapter_complete_marker(active_adapter_dir).exists()
            and self.adapter_matches_current_lora_config(active_adapter_dir)
        )

    def adapter_has_partial_artifacts(self, adapter_dir: Path | str | None = None) -> bool:
        active_adapter_dir = self.output_dir if adapter_dir is None else Path(adapter_dir)
        return (
            (active_adapter_dir / "adapter_config.json").exists()
            and not self.adapter_complete_marker(active_adapter_dir).exists()
        )

    def mark_adapter_complete(self) -> None:
        marker_payload = {
            "section_id": self.section_id,
            "round": self.current_round,
            "adapter_dir": str(self.output_dir),
        }
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.adapter_complete_marker().write_text(
            json.dumps(marker_payload, indent=2),
            encoding="utf-8",
        )

    def adapter_matches_current_lora_config(self, adapter_dir: Path | str) -> bool:
        adapter_config_path = Path(adapter_dir) / "adapter_config.json"
        if not adapter_config_path.exists():
            return False

        try:
            raw_config = json.loads(adapter_config_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return False

        expected_targets = self.lora_config().target_modules
        return raw_config.get("target_modules") == expected_targets

    def assert_adapter_has_lora_updates(self, adapter_dir: Path | str | None = None) -> None:
        active_adapter_dir = self.output_dir if adapter_dir is None else Path(adapter_dir)
        adapter_weights_path = active_adapter_dir / "adapter_model.safetensors"
        if not adapter_weights_path.exists():
            raise FileNotFoundError(f"Missing adapter weights: {adapter_weights_path}")

        load_file = import_safetensors_load_file()
        tensors = load_file(adapter_weights_path)
        lora_b_tensors = [
            tensor
            for name, tensor in tensors.items()
            if "lora_B" in name
        ]
        if not lora_b_tensors:
            raise ValueError(f"No lora_B tensors were saved in {adapter_weights_path}.")

        if not any(float(tensor.detach().abs().sum().cpu().item()) > 0 for tensor in lora_b_tensors):
            raise ValueError(
                "LoRA adapter was saved, but all lora_B tensors are still zero. "
                "This usually means target_modules did not touch the text path "
                "used by training."
            )

    def previous_adapter_dir(self) -> Path | None:
        override_path = self.pipeline.base_adapter_override_path
        self.used_base_adapter_override = False
        if self.current_round > 0 and override_path:
            candidate = Path(override_path)
            if self.adapter_is_complete(candidate):
                self.used_base_adapter_override = True
                return candidate
            print(
                "Configured rollback adapter is not complete; "
                f"falling back to the latest completed round adapter: {candidate}"
            )
            self.pipeline.base_adapter_override_path = None
            save_pipeline_state(self.pipeline, self.section_id)

        for round_id in range(self.current_round - 1, -1, -1):
            candidate = self.round_adapter_dir(round_id)
            if self.adapter_is_complete(candidate):
                return candidate
        return None

    def tokenize_dataset(self) -> list[dict[str, Any]]:
        self.validate_training_batch()
        self.tokenized_batch = [
            self.encode_training_sample(sample, self.load_tokenizer(), self.max_length)
            for sample in self.samples
        ]
        return self.tokenized_batch

    def validate_training_batch(self, samples: list[Training_Sample] | None = None) -> None:
        batch = self.samples if samples is None else samples

        if not batch:
            raise ValueError(
                "Pipeline current_training_batch is empty. Run the instructor first."
            )

        if (
            self.expected_batch_size is not None
            and len(batch) != self.expected_batch_size
        ):
            raise ValueError(
                "Student expects "
                f"{self.expected_batch_size} samples per round, got {len(batch)}."
            )

        missing_gold = [
            index
            for index, sample in enumerate(batch, start=1)
            if not sample.gold_summary or not sample.gold_summary.strip()
        ]
        if missing_gold:
            raise ValueError(
                "Training samples are missing gold_summary at positions: "
                + ", ".join(str(index) for index in missing_gold)
            )

    def create_user_content(self, sample: Training_Sample) -> str:
        content = sample.instruct.strip()
        if sample.sample.strip():
            content = f"{content}\n\nInput:\n{sample.sample.strip()}"
        return content

    def create_prompt(self, sample: Training_Sample, tokenizer: Any | None = None) -> str:
        if (
            is_gemma4_model_id(self.base_model_location)
            and tokenizer is not None
            and hasattr(tokenizer, "apply_chat_template")
        ):
            return tokenizer.apply_chat_template(
                [{"role": "user", "content": self.create_user_content(sample)}],
                tokenize=False,
                add_generation_prompt=True,
            )

        parts = ["### Instruction:", sample.instruct.strip()]

        if sample.sample.strip():
            parts.extend(["", "### Input:", sample.sample.strip()])

        parts.extend(["", "### Output:"])
        return "\n".join(parts) + "\n"

    def resolve_base_model_location(self) -> Path | str:
        model_path = Path(self.model_id)
        if model_path.exists():
            return model_path

        local_model_path = BASE_MODEL_DIR / self.model_id
        if local_model_path.exists():
            return local_model_path

        return self.model_id

    def encode_training_sample(self, sample: Training_Sample, tokenizer: Any, max_length: int = DEFAULT_MAX_LENGTH) -> dict[str, Any]:
        if not sample.gold_summary or not sample.gold_summary.strip():
            raise ValueError("Training samples must include gold_summary.")

        prompt = self.create_prompt(sample, tokenizer)
        target = sample.gold_summary.strip()
        eos_token = tokenizer.eos_token or ""
        full_text = prompt + target + eos_token

        prompt_ids = tokenizer(
            prompt,
            add_special_tokens=False,
        )["input_ids"]
        encoded = tokenizer(
            full_text,
            max_length=max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        input_ids = encoded["input_ids"].squeeze(0)
        attention_mask = encoded["attention_mask"].squeeze(0)
        labels = input_ids.clone()

        prompt_length = min(len(prompt_ids), max_length)
        labels[:prompt_length] = -100
        labels[attention_mask == 0] = -100
        torch_imported = import_torch()

        return {
            "input_ids": input_ids.to(dtype=torch_imported.long),
            "attention_mask": attention_mask,
            "labels": labels.to(dtype=torch_imported.long),
        }

    def load_tokenizer(self):
        if self.tokenizer is not None:
            return self.tokenizer

        transformers = import_transformers()
        if is_gemma4_model_id(self.base_model_location):
            processor = transformers["AutoProcessor"].from_pretrained(
                self.base_model_location,
                trust_remote_code=True,
            )
            tokenizer = getattr(processor, "tokenizer", processor)
        else:
            tokenizer = transformers["AutoTokenizer"].from_pretrained(
                self.base_model_location,
                trust_remote_code=True,
            )

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        tokenizer.padding_side = "right"
        self.tokenizer = tokenizer
        return tokenizer

    def load_base_model(self):
        torch_imported = import_torch()
        transformers = import_transformers()

        load_kwargs: dict[str, Any] = {
            "trust_remote_code": True,
            "low_cpu_mem_usage": True,
        }

        if torch_imported.cuda.is_available():
            load_kwargs.update(
                {
                    "quantization_config": transformers[
                        "BitsAndBytesConfig"
                    ](
                        load_in_4bit=True,
                        bnb_4bit_compute_dtype=torch_imported.bfloat16,
                        bnb_4bit_quant_type="nf4",
                        bnb_4bit_use_double_quant=True,
                    ),
                    "device_map": "cuda:0",
                    "dtype": torch_imported.bfloat16,
                }
            )
        else:
            load_kwargs["dtype"] = torch_imported.float32

        model = transformers["AutoModelForCausalLM"].from_pretrained(
            self.base_model_location,
            **load_kwargs,
        )
        model.config.use_cache = False
        return model

    def lora_config(self):
        peft_imported = import_peft()
        return peft_imported["LoraConfig"](
            r=4,
            lora_alpha=8,
            target_modules=(
                GEMMA4_LANGUAGE_LORA_TARGET_MODULES
                if is_gemma4_model_id(self.base_model_location)
                else BASE_LORA_TARGET_MODULES
            ),
            lora_dropout=0.05,
            bias="none",
            task_type=peft_imported["TaskType"].CAUSAL_LM,
        )

    def make_model_ready(self):
        torch_imported = import_torch()
        peft_imported = import_peft()
        model = self.load_base_model()

        if torch_imported.cuda.is_available():
            model = peft_imported["prepare_model_for_kbit_training"](model)

        if self.current_round == 0:
            if self.adapter_has_partial_artifacts(self.output_dir):
                print(
                    "Found incomplete round-0 adapter artifacts; "
                    "retraining round 0 from a fresh LoRA adapter."
                )
            append_section_debug_log(
                self.section_id,
                "training_base_adapter",
                {
                    "round_id": self.current_round,
                    "base_kind": "fresh_base_model",
                    "base_adapter_path": None,
                    "used_base_adapter_override": False,
                    "output_adapter_path": str(self.output_dir),
                },
            )
            model = peft_imported["get_peft_model"](model, self.lora_config())
        else:
            adapter_dir = self.previous_adapter_dir()
            if adapter_dir is None:
                raise FileNotFoundError(
                    "Round "
                    f"{self.current_round} must continue a previous LoRA adapter, "
                    "but no completed adapter marker was found in earlier "
                    "round folders."
                )
            append_section_debug_log(
                self.section_id,
                "training_base_adapter",
                {
                    "round_id": self.current_round,
                    "base_kind": (
                        "rollback_override"
                        if self.used_base_adapter_override
                        else "previous_completed_round"
                    ),
                    "base_adapter_path": str(adapter_dir),
                    "used_base_adapter_override": self.used_base_adapter_override,
                    "output_adapter_path": str(self.output_dir),
                },
            )
            model = peft_imported["PeftModel"].from_pretrained(
                model,
                adapter_dir,
                is_trainable=True,
            )

        model.train()
        self.model = model
        return model

    def create_trainer(self):
        transformers = import_transformers()
        tokenizer = self.load_tokenizer()
        train_dataset = self.tokenize_dataset()
        model_ready = self.make_model_ready()

        torch_imported = import_torch()
        use_bf16 = torch_imported.cuda.is_available() and torch_imported.cuda.is_bf16_supported()
        use_fp16 = torch_imported.cuda.is_available() and not use_bf16

        args = transformers["TrainingArguments"](
            output_dir=str(self.output_dir),
            per_device_train_batch_size=1,
            gradient_accumulation_steps=4,
            learning_rate=2e-4,
            num_train_epochs=3,
            logging_steps=1,
            save_strategy="no",
            bf16=use_bf16,
            fp16=use_fp16,
            report_to="none",
            remove_unused_columns=False,
            disable_tqdm=True,
        )

        self.trainer = transformers["Trainer"](
            model=model_ready,
            args=args,
            train_dataset=train_dataset,
            data_collator=transformers["default_data_collator"],
            **trainer_processor_kwargs(transformers["Trainer"], tokenizer),
        )
        return self.trainer

    def train(self, *, keep_model: bool = True) -> Path:
        trainer = self.create_trainer()
        trainer.train()

        self.output_dir.mkdir(parents=True, exist_ok=True)
        trainer.model.save_pretrained(self.output_dir)
        self.load_tokenizer().save_pretrained(self.output_dir)
        self.assert_adapter_has_lora_updates()
        self.mark_adapter_complete()
        append_section_debug_log(
            self.section_id,
            "training_complete",
            {
                "round_id": self.current_round,
                "output_adapter_path": str(self.output_dir),
                "used_base_adapter_override": self.used_base_adapter_override,
                "sample_count": len(self.samples),
            },
        )
        trained_model = trainer.model
        self.trainer = None
        del trainer
        if keep_model:
            self.model = trained_model
            self.model.eval()
        else:
            self.model = None

        if self.used_base_adapter_override:
            self.pipeline.base_adapter_override_path = None
            save_pipeline_state(self.pipeline, self.section_id)
            self.used_base_adapter_override = False
        return self.output_dir

    def load_trained_model(self, adapter_dir: Path | str | None = None):
        peft_imported = import_peft()
        tokenizer = self.load_tokenizer()
        model = self.load_base_model()

        adapter_path = Path(adapter_dir) if adapter_dir is not None else self.output_dir
        if not self.adapter_is_complete(adapter_path):
            latest_adapter = self.previous_adapter_dir()
            if latest_adapter is None:
                raise FileNotFoundError(
                    "No completed LoRA adapter found at "
                    f"{adapter_path}. Expected adapter_config.json and "
                    f"{ADAPTER_COMPLETE_FILENAME}."
                )
            adapter_path = latest_adapter

        self.model = peft_imported["PeftModel"].from_pretrained(model, adapter_path)
        self.model.eval()
        self.tokenizer = tokenizer
        return self.model

    def _active_model(self):
        if self.model is None:
            return self.load_trained_model()
        return self.model

    def generate_answer(self, sample: Training_Sample, max_new_tokens: int = 80) -> str:
        torch_imported = import_torch()
        transformers = import_transformers()
        tokenizer = self.load_tokenizer()
        model = self._active_model()
        prompt = self.create_prompt(sample)
        inputs = tokenizer(prompt, return_tensors="pt")
        device = next(model.parameters()).device
        inputs = {key: value.to(device) for key, value in inputs.items()}

        with torch_imported.no_grad():
            generated = model.generate(
                **inputs,
                generation_config=transformers["GenerationConfig"](
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                ),
            )

        answer_ids = generated[0][inputs["input_ids"].shape[-1] :]
        return tokenizer.decode(answer_ids, skip_special_tokens=True).strip()

    def evaluate_sample(
        self,
        sample: Training_Sample,
        *,
        generate_answer: bool = False,
    ) -> Evaluation_Output:
        torch_imported = import_torch()
        tokenizer = self.load_tokenizer()
        model = self._active_model()
        model.eval()
        encoded = self.encode_training_sample(sample, tokenizer, self.max_length)
        device = next(model.parameters()).device
        batch = {
            key: value.unsqueeze(0).to(device)
            for key, value in encoded.items()
        }

        with torch_imported.no_grad():
            outputs = model(**batch)

        student_ans = self.generate_answer(sample) if generate_answer else ""
        token_length = int(batch["attention_mask"].sum().detach().cpu().item())

        return Evaluation_Output(
            sample=sample.sample,
            instruct=sample.instruct,
            gold_summary=sample.gold_summary,
            student_ans=student_ans,
            token_length=token_length,
            loss=float(outputs.loss.detach().cpu().item()),
            tag=sample.tag,
            source=sample.source,
            source_ledger_id=sample.source_ledger_id,
        )

    def evaluate_samples(
        self,
        samples: list[Training_Sample],
        *,
        generate_answers: bool = False,
    ) -> list[Evaluation_Output]:
        self.validate_training_batch(samples)
        return [
            self.evaluate_sample(sample, generate_answer=generate_answers)
            for sample in samples
        ]

    def release_runtime(self, *, release_tokenizer: bool = True) -> None:
        self.trainer = None
        self.model = None
        self.tokenized_batch = []
        if release_tokenizer:
            self.tokenizer = None
