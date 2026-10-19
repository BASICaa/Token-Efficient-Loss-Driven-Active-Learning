"""Local quantized model loading shared by the isolated experiment runner."""

from __future__ import annotations

import gc
import hashlib
from pathlib import Path
import random


def set_seed(seed):
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def prepare_quantized_model(model, *, checkpointing=True):
    """Freeze the base and stabilize small parameters without expanding huge embeddings.

    PEFT's generic preparation upcasts all floating non-quantized parameters.
    Gemma's large frozen embeddings retain their pretrained dtype here; all
    methods and the base reference share this explicit precision contract.
    """
    import torch

    for parameter in model.parameters():
        parameter.requires_grad_(False)
        if (
            parameter.dtype in (torch.float16, torch.bfloat16)
            and parameter.numel() < 1000000
            and parameter.__class__.__name__ != "Params4bit"
        ):
            parameter.data = parameter.data.float()
    if checkpointing:
        model.enable_input_require_grads()
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
    return model


def load_runtime(
    model_path, *, seed=42, adapter=None, train=True, training_config=None
):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import LoraConfig, PeftModel, get_peft_model
    from .student import (
        GEMMA4_LANGUAGE_LORA_TARGET_MODULES,
        BASE_LORA_TARGET_MODULES,
        is_gemma4_model_id,
    )
    from .experiment_settings import TrainingSettings

    settings = TrainingSettings.model_validate(training_config or {})

    set_seed(seed)
    path = Path(model_path)
    if not path.is_dir():
        raise FileNotFoundError(
            "Use an existing local model; automatic model downloads are disabled."
        )
    tokenizer = AutoTokenizer.from_pretrained(
        path, local_files_only=True, trust_remote_code=False
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    kwargs = {
        "local_files_only": True,
        "trust_remote_code": False,
        "attn_implementation": "eager",
    }
    if torch.cuda.is_available():
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        kwargs.update(
            dtype=dtype,
            device_map={"": "cuda:0"},
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=dtype,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            ),
        )
    else:
        kwargs["dtype"] = torch.float32
    model = AutoModelForCausalLM.from_pretrained(path, **kwargs)
    model.config.use_cache = False
    if torch.cuda.is_available():
        model = prepare_quantized_model(model, checkpointing=train)
    if adapter:
        model = PeftModel.from_pretrained(model, adapter, is_trainable=train)
    elif train:
        targets = (
            GEMMA4_LANGUAGE_LORA_TARGET_MODULES
            if is_gemma4_model_id(path)
            else BASE_LORA_TARGET_MODULES
        )
        model = get_peft_model(
            model,
            LoraConfig(
                r=settings.lora_rank,
                lora_alpha=settings.lora_alpha,
                lora_dropout=settings.lora_dropout,
                target_modules=targets,
                bias="none",
                task_type="CAUSAL_LM",
            ),
        )
    if train:
        model.enable_input_require_grads()
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
    model.eval()
    return model, tokenizer


def cleanup():
    import torch

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def file_fingerprint(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
