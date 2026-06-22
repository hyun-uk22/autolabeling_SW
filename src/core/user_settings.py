import json
import os
from pathlib import Path
from typing import Dict

from dotenv import load_dotenv


ENV_FIELDS = [
    "AWS_REGION",
    "AWS_PROFILE",
    "AWS_ACCESS_KEY_ID",
    "AWS_SECRET_ACCESS_KEY",
    "AWS_SESSION_TOKEN",
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "LOW_MODEL",
    "HIGH_MODEL",
    "PLANNER_MODEL",
    "INTENT_ROUTER_MODEL",
    "CHAT_MODEL",
]

SECRET_FIELDS = {
    "AWS_ACCESS_KEY_ID",
    "AWS_SECRET_ACCESS_KEY",
    "AWS_SESSION_TOKEN",
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
}


def _settings_path() -> Path:
    return Path.home() / ".autolabel" / "settings.json"


def read_user_settings() -> Dict[str, str]:
    path = _settings_path()
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return {key: str(value) for key, value in data.items() if key in ENV_FIELDS and value}


def save_user_settings(values: Dict[str, str]) -> str:
    clean = {
        key: str(value).strip()
        for key, value in values.items()
        if key in ENV_FIELDS and str(value).strip()
    }
    path = _settings_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(clean, ensure_ascii=False, indent=2), encoding="utf-8")
    for key, value in clean.items():
        os.environ[key] = value
    return str(path)


def load_user_environment() -> Dict[str, str]:
    load_dotenv()
    settings = read_user_settings()
    if "AWS_PROFILE" not in settings and (
        settings.get("AWS_ACCESS_KEY_ID") and settings.get("AWS_SECRET_ACCESS_KEY")
    ):
        os.environ.pop("AWS_PROFILE", None)
        os.environ.pop("AWS_DEFAULT_PROFILE", None)
    for key, value in settings.items():
        os.environ.setdefault(key, value)
    return settings
