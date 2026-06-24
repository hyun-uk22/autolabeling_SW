import copy
import json
import os
import re
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator

from ..core.llm_client import extract_json
from ..core.model_config import resolve_planner_model
from ..utils.label_importer import extract_class_names_from_text
from .conversation_router import _call_model
from .models import OperationPlan


PATCH_SYSTEM_PROMPT = """
You are an execution-plan patch proposer for AutoLabel.
Return JSON only. Do not execute anything.

Given the current plan and the user's revision request, propose conservative field updates.
Only include fields the user explicitly asks to change.
Never change action unless the user clearly asks for a new task; for ordinary revisions, keep action unchanged.

Allowed response shape:
{
  "mode": "patch|cancel|new_plan|clarify",
  "reason": "short Korean reason",
  "updates": {
    "formats": ["yolo|pascal_voc|coco|vision_json"],
    "source_format": "auto|yolo|pascal_voc|coco|vision_json|csv|generic_json|custom_mapping",
    "input_path": "path",
    "img_dir": "path",
    "out_dir": "path",
    "vis_dir": "path",
    "classes_path": "path",
    "task_type": "classification|object_detection|segmentation|pose_estimation|ocr|tracking|all",
    "threshold": 0.75,
    "duplicate_iou": 0.85,
    "strict": true,
    "specialist_consistency_runs": 1,
    "specialist_advisor_mode": "none|low|high|both",
    "llm_consistency_mode": "none|low|high|both",
    "usage_mode": "library|official_repo|custom",
    "framework": "huggingface|mmsegmentation|detectron2|pytorch|custom",
    "dataset_purpose": "training|inference|evaluation",
    "repo_url": "https://github.com/owner/repo",
    "repo_path": "path",
    "prompt": "optional updated generation prompt"
  }
}

Use cancel when the user asks to cancel the pending plan.
Use new_plan when the user asks to discard this plan and start a different task.
Use clarify when the requested change is ambiguous.
"""

ALLOWED_UPDATE_FIELDS = {
    "formats",
    "source_format",
    "input_path",
    "img_dir",
    "out_dir",
    "vis_dir",
    "classes_path",
    "task_type",
    "threshold",
    "duplicate_iou",
    "strict",
    "specialist_consistency_runs",
    "specialist_advisor_mode",
    "llm_consistency_mode",
    "usage_mode",
    "framework",
    "dataset_purpose",
    "repo_url",
    "repo_path",
    "prompt",
}
ALLOWED_FORMATS = {"yolo", "pascal_voc", "coco", "vision_json"}
ALLOWED_SOURCE_FORMATS = ALLOWED_FORMATS | {"auto", "csv", "generic_json", "custom_mapping", "mask_image"}
ALLOWED_TASKS = {
    "classification",
    "object_detection",
    "segmentation",
    "pose_estimation",
    "ocr",
    "tracking",
    "all",
    "semantic_segmentation",
    "instance_segmentation",
    "panoptic_segmentation",
}
ALLOWED_ADVISOR_MODES = {"none", "low", "high", "both"}
ALLOWED_USAGE_MODES = {"library", "official_repo", "custom"}
ALLOWED_FRAMEWORKS = {"huggingface", "mmsegmentation", "detectron2", "pytorch", "custom"}
ALLOWED_PURPOSES = {"training", "inference", "evaluation"}
PATH_FIELDS = {"input_path", "img_dir", "out_dir", "vis_dir", "classes_path", "repo_path"}
TASK_ALIASES = {
    "classification": ("classification", "분류", "이미지 분류", "classify"),
    "object_detection": ("object detection", "object_detection", "detection", "detect", "객체 탐지", "객체 검출", "탐지", "검출"),
    "segmentation": ("segmentation", "세그멘테이션", "분할", "seg"),
    "pose_estimation": ("pose estimation", "pose_estimation", "pose", "포즈", "keypoint", "키포인트"),
    "ocr": ("ocr", "문자 인식", "텍스트 인식", "글자 인식"),
    "tracking": ("tracking", "추적", "track"),
    "all": ("all", "전체 태스크", "모든 태스크"),
}


class PlanPatch(BaseModel):
    mode: str = Field(default="patch")
    reason: str = ""
    updates: Dict[str, Any] = Field(default_factory=dict)

    @field_validator("mode")
    @classmethod
    def validate_mode(cls, value: str) -> str:
        if value not in {"patch", "cancel", "new_plan", "clarify"}:
            raise ValueError(f"Unsupported patch mode: {value}")
        return value


def _relative_or_string(path: str, workspace: Path) -> str:
    try:
        return Path(path).resolve().relative_to(workspace).as_posix()
    except (OSError, ValueError):
        return str(path)


def _resolve_path(value: Any, workspace: Path) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("경로 값이 비어 있습니다.")
    raw = value.strip().strip('"').strip("'")
    candidate = Path(raw).expanduser()
    if not candidate.is_absolute():
        candidate = workspace / candidate
    resolved = candidate.resolve()
    try:
        resolved.relative_to(workspace)
    except ValueError as exc:
        raise ValueError(f"workspace 외부 경로는 plan 수정에서 허용하지 않습니다: {raw}") from exc
    return str(resolved)


def _resolve_any_path(value: Any, workspace: Path) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("경로 값이 비어 있습니다.")
    raw = value.strip().strip('"').strip("'")
    candidate = Path(raw).expanduser()
    if not candidate.is_absolute():
        candidate = workspace / candidate
    return str(candidate.resolve())


def _sanitize_updates(updates: Dict[str, Any], current_operation: Dict[str, Any], workspace: Path) -> Dict[str, Any]:
    if not isinstance(updates, dict):
        raise ValueError("patch updates는 객체여야 합니다.")
    unknown = set(updates) - ALLOWED_UPDATE_FIELDS
    if unknown:
        raise ValueError(f"허용되지 않은 plan 필드 수정입니다: {', '.join(sorted(unknown))}")

    sanitized: Dict[str, Any] = {}
    for key, value in updates.items():
        if value is None:
            continue
        if key == "formats":
            formats = [str(item).strip() for item in (value if isinstance(value, list) else [value]) if str(item).strip()]
            invalid = set(formats) - ALLOWED_FORMATS
            if invalid:
                raise ValueError(f"지원하지 않는 출력 포맷입니다: {', '.join(sorted(invalid))}")
            sanitized[key] = list(dict.fromkeys(formats))
        elif key == "source_format":
            source_format = str(value).strip()
            if source_format not in ALLOWED_SOURCE_FORMATS:
                raise ValueError(f"지원하지 않는 입력 포맷입니다: {source_format}")
            sanitized[key] = source_format
        elif key == "task_type":
            task_type = str(value).strip()
            if task_type not in ALLOWED_TASKS:
                raise ValueError(f"지원하지 않는 task_type입니다: {task_type}")
            sanitized[key] = task_type
        elif key == "specialist_advisor_mode":
            mode = str(value).strip()
            if mode not in ALLOWED_ADVISOR_MODES:
                raise ValueError(f"지원하지 않는 advisor mode입니다: {mode}")
            sanitized[key] = mode
        elif key == "usage_mode":
            mode = str(value).strip()
            if mode not in ALLOWED_USAGE_MODES:
                raise ValueError(f"지원하지 않는 사용 방식입니다: {mode}")
            sanitized[key] = mode
        elif key == "framework":
            framework = str(value).strip()
            if framework not in ALLOWED_FRAMEWORKS:
                raise ValueError(f"지원하지 않는 프레임워크입니다: {framework}")
            sanitized[key] = framework
        elif key == "dataset_purpose":
            purpose = str(value).strip()
            if purpose not in ALLOWED_PURPOSES:
                raise ValueError(f"지원하지 않는 목적입니다: {purpose}")
            sanitized[key] = purpose
        elif key == "repo_url":
            repo_url = str(value).strip()
            if not repo_url.startswith(("https://", "http://")):
                raise ValueError("repo_url은 http:// 또는 https:// URL이어야 합니다.")
            sanitized[key] = repo_url
        elif key == "threshold":
            threshold = float(value)
            if not 0.0 <= threshold <= 1.0:
                raise ValueError("threshold는 0~1 범위여야 합니다.")
            sanitized[key] = threshold
        elif key == "duplicate_iou":
            duplicate_iou = float(value)
            if not 0.0 < duplicate_iou <= 1.0:
                raise ValueError("duplicate_iou는 0보다 크고 1 이하여야 합니다.")
            sanitized[key] = duplicate_iou
        elif key == "specialist_consistency_runs":
            runs = int(value)
            if not 0 <= runs <= 3:
                raise ValueError("specialist_consistency_runs는 0~3 범위여야 합니다.")
            sanitized[key] = runs
        elif key in {"specialist_advisor_mode", "llm_consistency_mode"}:
            mode = str(value).strip().lower()
            if mode not in ALLOWED_ADVISOR_MODES:
                raise ValueError(f"지원하지 않는 모드입니다: {mode}")
            sanitized[key] = mode
        elif key == "strict":
            if isinstance(value, bool):
                sanitized[key] = value
            elif isinstance(value, (int, float)):
                sanitized[key] = value > 0
            else:
                sanitized[key] = str(value).strip().lower() in {"true", "1", "yes", "y", "on", "strict", "엄격"}
        elif key == "repo_path":
            sanitized[key] = _resolve_any_path(value, workspace)
        elif key == "prompt":
            prompt = str(value).strip()
            if not prompt:
                raise ValueError("prompt 값이 비어 있습니다.")
            sanitized[key] = prompt
        elif key in PATH_FIELDS:
            sanitized[key] = _resolve_path(value, workspace)

    if "formats" in sanitized and not sanitized["formats"]:
        raise ValueError("출력 포맷은 최소 1개 이상이어야 합니다.")
    operation_copy = copy.deepcopy(current_operation)
    operation_copy.update(sanitized)
    OperationPlan.model_validate(operation_copy)
    return sanitized


def _call_patch_model(
    request: str,
    current_proposal: Dict[str, Any],
    model_name: Optional[str],
    caller: Optional[Callable[[str], Dict[str, Any]]],
) -> PlanPatch:
    prompt = json.dumps(
        {
            "current_plan": current_proposal.get("plan", {}),
            "current_summary": current_proposal.get("summary", ""),
            "revision_request": request,
        },
        ensure_ascii=False,
        indent=2,
    )
    if caller:
        payload = caller(prompt)
    else:
        if not model_name:
            raise RuntimeError("Plan patcher 모델이 설정되지 않았습니다. PLAN_PATCH_MODEL 또는 INTENT_ROUTER_MODEL을 설정하세요.")
        payload = extract_json(_call_model(model_name, PATCH_SYSTEM_PROMPT, prompt, True))
    return PlanPatch.model_validate(payload)


def _task_type_from_request(request: str) -> Optional[str]:
    lowered = request.lower().replace("-", " ")
    for task_type, aliases in TASK_ALIASES.items():
        if any(alias in lowered for alias in aliases):
            return task_type
    return None


def _summary_with_task(summary: str, task_type: str) -> str:
    base = (summary or "실행 계획").replace(" (수정됨)", "")
    updated = re.sub(r"(이미지\s+\d+개에 대해\s+)\S+(\s+라벨 생성)", rf"\1{task_type}\2", base)
    if updated == base and "라벨 생성" in base:
        updated = re.sub(r"(\s+라벨 생성)", rf" {task_type}\1", base, count=1)
    if updated == base:
        updated = f"{base} (수정됨)"
    else:
        updated = f"{updated} (수정됨)"
    return updated


def _without_class_missing_warnings(warnings: List[str]) -> List[str]:
    return [
        warning
        for warning in warnings
        if "검출 대상 클래스 목록이 필요" not in warning
        and "찾을 클래스명이 필요" not in warning
        and "Grounding DINO/Grounded SAM2 기반 생성" not in warning
    ]


def _prompt_with_request_classes(prompt: str, request: str) -> tuple[str, List[str]]:
    labels = extract_class_names_from_text(request)
    if not labels:
        return prompt, []
    merged_prompt = str(prompt or "").strip()
    if request.strip() and request.strip() not in merged_prompt:
        merged_prompt = f"{merged_prompt}\n{request.strip()}".strip()
    if "classes:" not in merged_prompt.lower() and "names:" not in merged_prompt.lower():
        merged_prompt = f"{merged_prompt}\nclasses: {', '.join(labels)}".strip()
    return merged_prompt, labels


def _deterministic_patch(
    request: str,
    current_proposal: Dict[str, Any],
    workspace: Path,
) -> Optional[Dict[str, Any]]:
    operations = current_proposal.get("plan", {}).get("operations", [])
    if len(operations) != 1:
        return None
    before = operations[0]
    if before.get("action") != "generate":
        return None
    task_type = _task_type_from_request(request)
    if not task_type or task_type == before.get("task_type"):
        return None

    prompt, labels = _prompt_with_request_classes(str(before.get("prompt") or ""), request)
    raw_updates = {"task_type": task_type}
    if labels:
        raw_updates["prompt"] = prompt
    updates = _sanitize_updates(raw_updates, before, workspace)
    revised = copy.deepcopy(current_proposal)
    revised["plan"]["operations"][0].update(updates)
    revised["warnings"] = _without_class_missing_warnings(list(revised.get("warnings", [])))
    if labels:
        revised["warnings"].append(
            "수정 요청에서 검출 대상 클래스를 추출해 specialist labels로 반영했습니다: "
            + ", ".join(labels)
        )
    revised["warnings"].append("요청에서 태스크 변경을 감지해 실행 계획에 반영했습니다.")
    revised["summary"] = _summary_with_task(revised.get("summary", "실행 계획"), task_type)
    return {
        "kind": "patch",
        "reason": "사용자가 라벨 생성 태스크 변경을 요청했습니다.",
        "proposal": revised,
        "changes": _describe_changes(before, updates, workspace),
    }


def _describe_changes(before: Dict[str, Any], updates: Dict[str, Any], workspace: Path) -> List[str]:
    changes = []
    for key, after in updates.items():
        before_value = before.get(key)
        display_before = before_value
        display_after = after
        if key in PATH_FIELDS:
            display_before = _relative_or_string(str(before_value), workspace) if before_value else ""
            display_after = _relative_or_string(str(after), workspace)
        changes.append(f"{key}: `{display_before}` -> `{display_after}`")
    return changes


def revise_pending_proposal(
    request: str,
    current_proposal: Dict[str, Any],
    workspace: str | Path,
    model_name: Optional[str] = None,
    caller: Optional[Callable[[str], Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    workspace_path = Path(workspace).expanduser().resolve()
    deterministic = _deterministic_patch(request, current_proposal, workspace_path)
    if deterministic:
        return deterministic
    patch = _call_patch_model(
        request,
        current_proposal,
        model_name or os.getenv("PLAN_PATCH_MODEL") or os.getenv("INTENT_ROUTER_MODEL") or resolve_planner_model(),
        caller,
    )
    if patch.mode != "patch":
        return {
            "kind": patch.mode,
            "reason": patch.reason,
            "proposal": current_proposal,
            "changes": [],
        }
    operations = current_proposal.get("plan", {}).get("operations", [])
    if len(operations) != 1:
        raise ValueError("현재는 단일 operation plan만 수정할 수 있습니다.")
    before = copy.deepcopy(operations[0])
    updates = _sanitize_updates(patch.updates, before, workspace_path)
    revised = copy.deepcopy(current_proposal)
    revised["plan"]["operations"][0].update(updates)
    revised["warnings"] = list(revised.get("warnings", []))
    revised["warnings"].append("LLM plan patch 제안을 검증한 뒤 실행 계획에 반영했습니다.")
    if before.get("action") == "generate" and "task_type" in updates:
        revised["summary"] = _summary_with_task(revised.get("summary", "실행 계획"), updates["task_type"])
    else:
        summary = (revised.get("summary", "실행 계획") or "실행 계획").replace(" (수정됨)", "")
        revised["summary"] = f"{summary} (수정됨)"
    return {
        "kind": "patch",
        "reason": patch.reason,
        "proposal": revised,
        "changes": _describe_changes(before, updates, workspace_path),
    }
