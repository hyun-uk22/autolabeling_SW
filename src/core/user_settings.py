import os
import tempfile
from pathlib import Path
from typing import Dict, Optional

from dotenv import dotenv_values, load_dotenv, set_key


ENV_FIELDS = (
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
)

SECRET_FIELDS = {
    "AWS_ACCESS_KEY_ID",
    "AWS_SECRET_ACCESS_KEY",
    "AWS_SESSION_TOKEN",
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
}


def user_config_dir() -> Path:
    base = os.getenv("APPDATA") or os.getenv("LOCALAPPDATA") or str(Path.home())
    return Path(base) / "AutoLabel"


def user_env_path() -> Path:
    return user_config_dir() / ".env"


def load_user_environment(path: Optional[Path] = None) -> Path:
    env_path = Path(path) if path else user_env_path()
    if env_path.exists():
        load_dotenv(env_path, override=True)
    return env_path


def read_user_settings(path: Optional[Path] = None) -> Dict[str, str]:
    env_path = Path(path) if path else user_env_path()
    saved = dotenv_values(env_path) if env_path.exists() else {}
    return {
        key: str(saved.get(key) or os.getenv(key, ""))
        for key in ENV_FIELDS
    }


def save_user_settings(values: Dict[str, str], path: Optional[Path] = None) -> Path:
    env_path = Path(path) if path else user_env_path()
    env_path.parent.mkdir(parents=True, exist_ok=True)
    normalized = {key: str(values.get(key, "")).strip() for key in ENV_FIELDS}

    descriptor, temporary_name = tempfile.mkstemp(
        prefix=".env-",
        suffix=".tmp",
        dir=env_path.parent,
        text=True,
    )
    os.close(descriptor)
    temporary_path = Path(temporary_name)
    try:
        for key, value in normalized.items():
            if value:
                set_key(str(temporary_path), key, value, quote_mode="always")
        os.replace(temporary_path, env_path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()

    for key, value in normalized.items():
        if value:
            os.environ[key] = value
        else:
            os.environ.pop(key, None)
    return env_path
