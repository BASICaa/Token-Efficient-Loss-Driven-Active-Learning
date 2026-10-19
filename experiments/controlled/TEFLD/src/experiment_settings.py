"""Validated settings adapted from the application's per-section configuration.

Defaults preserve the completed comparison's recipe. Section state is not read
by controlled experiments; a protocol is the source of settings and identity.
"""

from pydantic import BaseModel, ConfigDict, Field


class TrainingSettings(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    lora_rank: int = Field(default=4, gt=0)
    lora_alpha: int = Field(default=8, gt=0)
    lora_dropout: float = Field(default=0.05, ge=0, lt=1)
    learning_rate: float = Field(default=2e-4, gt=0)
    epochs_per_round: int = Field(default=3, gt=0)
    gradient_accumulation_steps: int = Field(default=4, gt=0)
    max_length: int = Field(default=512, gt=1)

    @classmethod
    def from_protocol(cls, protocol):
        common = protocol["common_training"]
        if common.get("micro_batch_size", 1) != 1:
            raise ValueError(
                "The controlled runtime currently requires micro_batch_size=1."
            )
        expected = {
            "optimizer": "adamw_torch",
            "adam_beta1": 0.9,
            "adam_beta2": 0.999,
            "adam_epsilon": 1e-8,
            "weight_decay": 0.0,
            "max_grad_norm": 1.0,
            "lr_scheduler_type": "linear",
            "warmup_steps": 0,
        }
        contract = common.get("optimizer_contract", {})
        for key, value in expected.items():
            if key in contract and contract[key] != value:
                raise ValueError(f"Unsupported optimizer contract: {key}.")
        return cls.model_validate(
            {key: common[key] for key in cls.model_fields if key in common}
        )
