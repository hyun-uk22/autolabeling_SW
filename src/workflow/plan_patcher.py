import copy
import re
from typing import Any, Dict, List


def _operation(proposal: Dict[str, Any]) -> Dict[str, Any]:
    operations = proposal.get("plan", {}).get("operations") or []
    return operations[0] if operations else {}


def _extract_path(text: str) -> str:
    quoted = re.search(r"[\"']([^\"']+)[\"']", text)
    if quoted:
        return quoted.group(1).strip()
    match = re.search(r"((?:[A-Za-z]:\\|data[\\/])[\w가-힣 ._\\/\-]+)", text)
    if match:
        path = match.group(1).strip().rstrip(".")
        return re.split(r"\s*(?:로|으로|에|에서| 바꿔| 변경| 해줘| 해|$)", path, maxsplit=1)[0].strip()
    return ""


def _extract_formats(text: str) -> List[str]:
    formats = []
    lowered = text.lower()
    for fmt in ("yolo", "pascal_voc", "coco", "vision_json"):
        aliases = {fmt}
        if fmt == "pascal_voc":
            aliases.update({"pascal", "voc", "xml"})
        if fmt == "vision_json":
            aliases.update({"jsonl", "vision json"})
        if any(alias in lowered for alias in aliases):
            formats.append(fmt)
    return list(dict.fromkeys(formats))


def revise_pending_proposal(
    request: str,
    proposal: Dict[str, Any],
    workspace: str,
) -> Dict[str, Any]:
    """Apply simple natural-language edits to a pending workflow proposal."""
    text = str(request or "").strip()
    lowered = text.lower()
    if not text:
        return {"kind": "clarify", "reason": "수정 요청이 비어 있습니다."}
    if any(word in lowered for word in ("취소", "cancel", "그만", "중단")):
        return {"kind": "cancel", "reason": "사용자가 현재 실행 계획 취소를 요청했습니다."}

    revised = copy.deepcopy(proposal)
    operation = _operation(revised)
    if not operation:
        return {"kind": "clarify", "reason": "수정할 실행 계획을 찾지 못했습니다."}

    changes: List[str] = []
    path = _extract_path(text)
    if path:
        if any(keyword in text for keyword in ("출력", "저장", "결과", "out", "output")):
            operation["out_dir"] = path
            changes.append(f"출력 경로를 `{path}`로 변경")
        elif any(keyword in text for keyword in ("이미지", "image", "raw")):
            operation["img_dir"] = path
            changes.append(f"이미지 폴더를 `{path}`로 변경")
        elif any(keyword in text for keyword in ("라벨", "label", "입력")):
            operation["input_path"] = path
            changes.append(f"입력 라벨 경로를 `{path}`로 변경")

    formats = _extract_formats(text)
    if formats and any(keyword in text for keyword in ("포맷", "형식", "변환", "format", "출력")):
        operation["formats"] = formats
        changes.append(f"출력 포맷을 `{', '.join(formats)}`로 변경")

    if "strict" in lowered or "제외" in text:
        operation["strict"] = True
        changes.append("입력 데이터 문제가 있는 데이터는 제외하도록 변경")
    if "포함" in text or "유지" in text:
        operation["strict"] = False
        changes.append("입력 데이터 문제가 있어도 가능한 데이터는 유지하도록 변경")

    if not changes:
        return {
            "kind": "clarify",
            "reason": "지원하는 수정은 출력 경로, 입력 라벨 경로, 이미지 폴더, 출력 포맷, 제외 여부입니다.",
        }

    return {
        "kind": "patch",
        "proposal": revised,
        "changes": changes,
        "reason": "사용자 수정 요청을 현재 실행 계획에 반영했습니다.",
    }
