"""Explicit teacher routing; credentials never enter protocol/cache identities."""

from urllib.parse import urlsplit
import math
import os
from pathlib import Path
import json


def hibana_profile_for_model(model):
    profiles = {
        "openai/gpt-6.1-sol": "experiment_protocol_hibana.json",
        "openai/gpt-6-sol": "experiment_protocol_hibana_gpt6.json",
    }
    if model not in profiles:
        return None
    path = (
        Path(__file__).resolve().parents[2]
        / "experiment_design_2026-10-18"
        / profiles[model]
    )
    controls = json.loads(path.read_text(encoding="utf-8"))["tefld_controls"]
    if controls["teacher_model_id"] != model:
        raise ValueError(
            "Legacy model differs from the Hibana profile; use the controlled runner with an explicit protocol."
        )
    return validate_provider(controls["provider"])


def validate_provider(config):
    if not config:
        return None
    if config.get("name") != "hibana":
        raise ValueError("Unsupported teacher provider.")
    url = urlsplit(config["base_url"])
    if (
        url.scheme != "https"
        or url.hostname not in {"api-ai.hibanacloud.ir", "api.hibanacloud.ir"}
        or url.path.rstrip("/") != "/v1"
        or url.username
        or url.password
        or url.query
        or url.fragment
    ):
        raise ValueError("Invalid Hibana API endpoint.")
    if (
        config.get("api_key_env") != "HIBANA_API_KEY"
        or config.get("currency") != "toman"
    ):
        raise ValueError(
            "Hibana requires its own credential variable and toman pricing."
        )
    for key in ("input_per_million", "output_per_million", "max_spend"):
        value = config.get(key)
        if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
            raise ValueError(
                "Hibana prices and spend cap must be finite positive values."
            )
    if "model_id" in config and (
        not isinstance(config["model_id"], str) or not config["model_id"].strip()
    ):
        raise ValueError("Provider pricing must name a model.")
    if "request_options" in config:
        options = config["request_options"]
        if not isinstance(options, dict) or set(options) - {
            "reasoning_effort",
            "temperature",
        }:
            raise ValueError("Unsupported provider request options.")
        if "temperature" in options and (
            type(options["temperature"]) not in (int, float)
            or not math.isfinite(options["temperature"])
            or not 0 <= options["temperature"] <= 2
        ):
            raise ValueError("Invalid provider temperature.")
        if "reasoning_effort" in options and options["reasoning_effort"] not in {
            "none",
            "low",
            "medium",
            "high",
        }:
            raise ValueError("Invalid provider reasoning setting.")
    return dict(config)


def client_kwargs(config):
    config = validate_provider(config)
    if config is None:
        return {}
    key = os.environ.get(config["api_key_env"])
    if not key:
        raise ValueError(
            "HIBANA_API_KEY is missing; configure it locally, not in chat."
        )
    return {"base_url": config["base_url"], "api_key": key}
