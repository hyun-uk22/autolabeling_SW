import json
from pathlib import Path
from typing import Optional


WORKSPACE_DEFAULTS = {
    "images": "data/raw",
    "labels": "data/labeled",
    "visualized": "data/visualized",
    "converted": "data/converted",
    "ground_truth": "data/ground_truth",
    "reports": "data/reports",
    "plugin_config": "configs/plugins.json",
}


def _config_path() -> Path:
    return Path.home() / ".autolabel" / "workspace.json"


def load_workspace() -> Optional[Path]:
    path = _config_path()
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    workspace = data.get("workspace")
    return Path(workspace) if workspace else None


def save_workspace(workspace: str, create_layout: bool = False) -> Path:
    root = Path(workspace).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    if create_layout:
        for value in WORKSPACE_DEFAULTS.values():
            path = root / value
            if path.suffix:
                path.parent.mkdir(parents=True, exist_ok=True)
            else:
                path.mkdir(parents=True, exist_ok=True)
    config = _config_path()
    config.parent.mkdir(parents=True, exist_ok=True)
    config.write_text(json.dumps({"workspace": str(root)}, ensure_ascii=False, indent=2), encoding="utf-8")
    return root


def resolve_workspace_path(workspace: str, value: str) -> str:
    path = Path(value).expanduser()
    if path.is_absolute():
        return str(path)
    return str((Path(workspace).expanduser().resolve() / path).resolve())
