from pathlib import Path
from typing import Any, Dict

from .models import OperationPlan, WorkflowPlan
from .planner import WorkflowPlanner


def _resolve_workspace_plan(plan: WorkflowPlan, workspace: str) -> Dict[str, Any]:
    root = Path(workspace).expanduser().resolve() if workspace else Path.cwd()
    data = plan.model_dump()
    for operation in data.get("operations", []):
        for key in ("input_path", "img_dir", "out_dir", "vis_dir", "gt_dir", "classes_path"):
            value = operation.get(key)
            if not value:
                continue
            path = Path(value).expanduser()
            if not path.is_absolute():
                operation[key] = str((root / path).resolve())
    return data


def handle_conversation(request: str, workspace: str = "") -> Dict[str, Any]:
    text = request.strip()
    if not text:
        return {"kind": "message", "response": "요청 내용을 입력해주세요."}

    lowered = text.lower()
    if any(token in lowered for token in ["상태", "도움", "help", "사용법"]):
        return {
            "kind": "message",
            "response": (
                "예: `data/failure_cases/bad_yolo_coordinates/labels를 yolo로 변환해줘`처럼 "
                "입력하면 실행 계획을 먼저 보여드립니다."
            ),
        }

    plan = WorkflowPlanner().plan(text)
    plan_data = _resolve_workspace_plan(plan, workspace)

    # Fallback planner may omit useful conversion defaults.
    for operation in plan_data.get("operations", []):
        if operation.get("action") == "convert":
            operation.setdefault("source_format", "auto")
            operation.setdefault("formats", ["yolo"])
            operation.setdefault("img_dir", str((Path(workspace) / "data/raw").resolve()) if workspace else "data/raw")
            operation.setdefault("out_dir", str((Path(workspace) / "data/converted").resolve()) if workspace else "data/converted")

    WorkflowPlan.model_validate(plan_data)
    return {
        "kind": "plan",
        "proposal": {
            "request": text,
            "plan": plan_data,
        },
    }
