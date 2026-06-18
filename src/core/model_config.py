import os
from typing import Optional, Tuple


DEFAULT_BEDROCK_LOW_MODEL_ID = "us.anthropic.claude-haiku-4-5-20251001-v1:0"
DEFAULT_BEDROCK_HIGH_MODEL_ID = "us.anthropic.claude-sonnet-4-5-20250929-v1:0"


def is_bedrock_model(model_name: str) -> bool:
    return model_name.startswith("bedrock:") or model_name.startswith("anthropic.claude")


def as_bedrock_model(model_id: Optional[str]) -> Optional[str]:
    if not model_id:
        return None
    return model_id if model_id.startswith("bedrock:") else f"bedrock:{model_id}"


def model_capacity_rank(model_name: str) -> int:
    normalized = model_name.lower()
    if "haiku" in normalized:
        return 1
    if "sonnet" in normalized:
        return 2
    if "opus" in normalized:
        return 3
    return 0


def resolve_model_names(low_model: Optional[str] = None, high_model: Optional[str] = None) -> Tuple[str, str]:
    resolved_low = (
        low_model
        or os.getenv("LOW_MODEL")
        or as_bedrock_model(os.getenv("AWS_BEDROCK_LOW_MODEL_ID"))
    )
    resolved_high = (
        high_model
        or os.getenv("HIGH_MODEL")
        or as_bedrock_model(os.getenv("AWS_BEDROCK_HIGH_MODEL_ID"))
        or as_bedrock_model(os.getenv("AWS_BEDROCK_MODEL_ID"))
    )

    if not resolved_low and resolved_high and is_bedrock_model(resolved_high):
        resolved_low = as_bedrock_model(DEFAULT_BEDROCK_LOW_MODEL_ID)
    if not resolved_high and resolved_low and is_bedrock_model(resolved_low):
        resolved_high = as_bedrock_model(DEFAULT_BEDROCK_HIGH_MODEL_ID)

    return resolved_low or "gpt-4o-mini", resolved_high or "gpt-4o"


def validate_cascade_setup(low_model: str, high_model: str, allow_same_model: bool = False) -> Tuple[bool, list[str]]:
    messages = []
    if low_model == high_model:
        messages.append("LOW_MODEL and HIGH_MODEL are identical.")
        if not allow_same_model:
            return False, messages
        messages.append("Continuing because identical models were explicitly allowed.")

    low_rank = model_capacity_rank(low_model)
    high_rank = model_capacity_rank(high_model)
    if low_rank and high_rank and low_rank >= high_rank:
        messages.append("LOW_MODEL does not appear lighter than HIGH_MODEL (Haiku < Sonnet < Opus).")
    return True, messages


def required_api_keys(low_model: str, high_model: str) -> list[str]:
    required = []
    if "gpt" in low_model.lower() or "gpt" in high_model.lower():
        required.append("OPENAI_API_KEY")
    if (
        ("claude" in low_model.lower() and not is_bedrock_model(low_model))
        or ("claude" in high_model.lower() and not is_bedrock_model(high_model))
    ):
        required.append("ANTHROPIC_API_KEY")
    return list(dict.fromkeys(required))
