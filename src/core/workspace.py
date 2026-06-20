import json
import os
import tempfile
from pathlib import Path
from typing import Optional, Union

from .user_settings import user_config_dir


PathValue = Union[str, Path]

WORKSPACE_DEFAULTS = {
    "images": "data/raw",
    "labels": "data/labeled",
    "visualized": "data/visualized",
    "converted": "data/converted",
    "ground_truth": "data/ground_truth",
    "reports": "data/reports",
    "plugin_config": "configs/plugins.json",
}


def workspace_config_path() -> Path:
    return user_config_dir() / "workspace.json"


def normalize_workspace(path: PathValue) -> Path:
    return Path(path).expanduser().resolve()


def load_workspace(path: Optional[Path] = None) -> Optional[Path]:
    config_path = Path(path) if path else workspace_config_path()
    if not config_path.exists():
        return None
    try:
        data = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return None
    value = data.get("workspace") if isinstance(data, dict) else None
    return normalize_workspace(value) if value else None


def ensure_workspace_layout(workspace: PathValue) -> Path:
    root = normalize_workspace(workspace)
    root.mkdir(parents=True, exist_ok=True)
    for key, relative_path in WORKSPACE_DEFAULTS.items():
        target = root / relative_path
        directory = target.parent if key == "plugin_config" else target
        directory.mkdir(parents=True, exist_ok=True)
        if key == "plugin_config" and not target.exists():
            target.write_text(
                json.dumps({"plugins": []}, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
    return root


def save_workspace(workspace: PathValue, path: Optional[Path] = None, create_layout: bool = False) -> Path:
    root = ensure_workspace_layout(workspace) if create_layout else normalize_workspace(workspace)
    root.mkdir(parents=True, exist_ok=True)
    config_path = Path(path) if path else workspace_config_path()
    config_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix="workspace-",
        suffix=".tmp",
        dir=config_path.parent,
        text=True,
    )
    os.close(descriptor)
    temporary_path = Path(temporary_name)
    try:
        temporary_path.write_text(
            json.dumps({"workspace": str(root)}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        os.replace(temporary_path, config_path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()
    return root


def resolve_workspace_path(workspace: PathValue, value: PathValue) -> str:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = normalize_workspace(workspace) / path
    return str(path.resolve())


def relative_to_workspace(workspace: PathValue, value: PathValue) -> str:
    path = Path(value).expanduser()
    if not path.is_absolute():
        return path.as_posix()
    try:
        return path.resolve().relative_to(normalize_workspace(workspace)).as_posix()
    except ValueError:
        return str(path.resolve())
