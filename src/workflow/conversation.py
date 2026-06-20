import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from .models import OperationPlan, WorkflowPlan


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
LABEL_EXTENSIONS = {".txt", ".xml", ".json", ".jsonl", ".csv"}
IGNORED_DIRECTORIES = {
    ".git",
    ".venv",
    "__pycache__",
    "build",
    "dist",
    "dist-installer",
    "logs",
    "visualized",
    "converted",
    "reports",
    "workflow",
}
FORMAT_ALIASES = {
    "coco": ("ms coco", "mscoco", "coco", "코코"),
    "pascal_voc": ("pascal voc", "pascal_voc", "voc", "파스칼"),
    "vision_json": ("vision json", "vision_json", "jsonl"),
    "yolo": ("yolo", "욜로"),
}
SOURCE_FORMAT_ALIASES = {
    **FORMAT_ALIASES,
    "csv": ("csv",),
    "generic_json": ("generic json", "generic_json", "일반 json"),
}


def _relative(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return str(path)


def _is_ignored(path: Path, root: Path) -> bool:
    try:
        parts = path.relative_to(root).parts
    except ValueError:
        return True
    return any(part.lower() in IGNORED_DIRECTORIES for part in parts[:-1])


def _looks_like_yolo(path: Path) -> bool:
    if path.name.lower() == "classes.txt":
        return False
    try:
        lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    except (OSError, UnicodeError):
        return False
    if not lines:
        return False
    for line in lines[:10]:
        parts = line.split()
        if len(parts) != 5:
            return False
        try:
            values = [float(value) for value in parts]
        except ValueError:
            return False
        if int(values[0]) != values[0] or not all(0.0 <= value <= 1.0 for value in values[1:]):
            return False
    return True


def _json_format(path: Path) -> Optional[str]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, ValueError):
        return None
    if isinstance(data, dict) and {"images", "annotations", "categories"}.issubset(data):
        return "coco"
    if isinstance(data, list) and data and all(isinstance(item, dict) for item in data[:10]):
        return "generic_json"
    if isinstance(data, dict) and any(
        key in data for key in ("image", "image_name", "file_name", "boxes", "segments", "objects")
    ):
        return "generic_json"
    return None


def _looks_like_label_csv(path: Path) -> bool:
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            fieldnames = set(csv.DictReader(handle).fieldnames or [])
    except (OSError, UnicodeError, csv.Error):
        return False
    image_fields = {"image", "image_name", "file_name", "filename"}
    box_fields = {"xmin", "ymin", "xmax", "ymax"}
    return bool(fieldnames & image_fields) and box_fields.issubset(fieldnames)


def _label_format(path: Path) -> Optional[str]:
    extension = path.suffix.lower()
    if extension == ".txt":
        return "yolo" if _looks_like_yolo(path) else None
    if extension == ".xml":
        return "pascal_voc"
    if extension == ".json":
        return _json_format(path)
    if extension == ".jsonl":
        return "vision_json"
    if extension == ".csv":
        return "csv" if _looks_like_label_csv(path) else None
    return None


def _candidate_score(candidate: Dict[str, Any]) -> tuple:
    relative = candidate["relative_path"].lower()
    preferred = int(relative == "data/labeled" or relative.startswith("data/labeled/"))
    return preferred, candidate["file_count"], -len(relative)


def discover_workspace(workspace: str | Path, max_files: int = 20000) -> Dict[str, Any]:
    root = Path(workspace).expanduser().resolve()
    if not root.is_dir():
        raise ValueError(f"Workspace가 존재하지 않습니다: {root}")

    image_counts: Counter[Path] = Counter()
    grouped_labels: Dict[tuple, List[Path]] = defaultdict(list)
    files_scanned = 0
    truncated = False

    for path in root.rglob("*"):
        if not path.is_file() or _is_ignored(path, root):
            continue
        files_scanned += 1
        if files_scanned > max_files:
            truncated = True
            break
        extension = path.suffix.lower()
        if extension in IMAGE_EXTENSIONS:
            image_counts[path.parent] += 1
            continue
        if extension not in LABEL_EXTENSIONS:
            continue
        label_format = _label_format(path)
        if not label_format:
            continue
        input_path = path if label_format in {"coco", "generic_json"} else path.parent
        grouped_labels[(input_path, label_format)].append(path)

    image_directories = [
        {
            "path": str(path),
            "relative_path": _relative(path, root),
            "file_count": count,
        }
        for path, count in image_counts.most_common()
    ]
    image_directories.sort(
        key=lambda item: (
            int(item["relative_path"].lower() == "data/raw"),
            item["file_count"],
        ),
        reverse=True,
    )

    label_candidates = [
        {
            "path": str(input_path),
            "relative_path": _relative(input_path, root),
            "format": label_format,
            "file_count": len(paths),
        }
        for (input_path, label_format), paths in grouped_labels.items()
    ]
    label_candidates.sort(key=_candidate_score, reverse=True)
    return {
        "workspace": str(root),
        "files_scanned": min(files_scanned, max_files),
        "truncated": truncated,
        "image_directories": image_directories,
        "label_candidates": label_candidates,
    }


def _contains_any(value: str, tokens: Iterable[str]) -> bool:
    return any(token in value for token in tokens)


def _mentioned_formats(request: str, aliases: Dict[str, Iterable[str]] = FORMAT_ALIASES) -> List[str]:
    lowered = request.lower().replace("-", " ")
    return [
        name
        for name, names in aliases.items()
        if any(alias in lowered for alias in names)
    ]


def _target_formats(request: str) -> List[str]:
    lowered = request.lower().replace("-", " ")
    mentioned = _mentioned_formats(request)
    if len(mentioned) <= 1:
        return mentioned
    if _contains_any(lowered, ("동시에", "두 가지", "둘 다", "모두 출력")):
        return mentioned

    targets = []
    for name, aliases in FORMAT_ALIASES.items():
        for alias in aliases:
            for match in re.finditer(re.escape(alias), lowered):
                suffix = lowered[match.end():match.end() + 18]
                if re.match(r"\s*(?:형식|포맷)?\s*(?:으로|로)", suffix):
                    targets.append(name)
                    break
            if name in targets:
                break
    return list(dict.fromkeys(targets)) or mentioned[-1:]


def _source_format(request: str) -> Optional[str]:
    lowered = request.lower().replace("-", " ")
    for name, aliases in SOURCE_FORMAT_ALIASES.items():
        for alias in aliases:
            for match in re.finditer(re.escape(alias), lowered):
                suffix = lowered[match.end():match.end() + 18]
                if re.match(r"\s*(?:라벨|데이터셋|파일)", suffix):
                    return name
    return None


def _action(request: str) -> str:
    lowered = request.lower()
    if _contains_any(lowered, ("평가", "비교", "리포트", "evaluate", "evaluation")):
        return "evaluate"
    if _contains_any(lowered, ("변환", "바꿔", "바꾸", "변경", "통일", "convert", "format")):
        return "convert"
    if _contains_any(lowered, ("생성", "라벨링", "라벨링해", "만들어", "generate", "label image")):
        return "generate"
    raise ValueError("요청에서 작업 종류를 확인하지 못했습니다. 변환, 라벨 생성 또는 평가 작업을 명시해 주세요.")


def _task_type(request: str) -> str:
    lowered = request.lower()
    mappings = (
        ("segmentation", ("segmentation", "세그멘테이션", "분할")),
        ("pose_estimation", ("pose", "포즈")),
        ("ocr", ("ocr", "문자 인식")),
        ("tracking", ("tracking", "추적")),
        ("classification", ("classification", "분류")),
    )
    for task, aliases in mappings:
        if _contains_any(lowered, aliases):
            return task
    return "object_detection"


def _discover_runs(root: Path) -> Dict[str, str]:
    runs = {}
    for path in root.rglob("run_metrics.csv"):
        if _is_ignored(path, root):
            continue
        name = path.parent.name or f"run-{len(runs) + 1}"
        unique_name = name
        suffix = 2
        while unique_name in runs:
            unique_name = f"{name}-{suffix}"
            suffix += 1
        runs[unique_name] = str(path.parent.resolve())
    return runs


def _explicit_workspace_path(request: str, root: Path) -> Optional[Path]:
    path_pattern = r"(?<![\w.])([\w.-]+(?:[\\/][\w.-]+)+)"
    for raw in re.findall(path_pattern, request):
        candidate = Path(raw.replace("\\", "/"))
        if candidate.is_absolute():
            resolved = candidate.resolve()
        else:
            resolved = (root / candidate).resolve()
        try:
            resolved.relative_to(root)
        except ValueError as exc:
            raise ValueError("Workspace 밖의 경로는 대화형 자동 실행에서 사용할 수 없습니다.") from exc
        if not resolved.exists():
            raise ValueError(f"요청에 지정된 경로가 존재하지 않습니다: {_relative(resolved, root)}")
        return resolved
    return None


def _numeric_option(request: str, name: str) -> Optional[float]:
    match = re.search(rf"(?:{name})[^0-9]{{0,20}}([01](?:\.\d+)?)", request, flags=re.IGNORECASE)
    return float(match.group(1)) if match else None


def _select_label_candidate(
    candidates: List[Dict[str, Any]],
    root: Path,
    explicit_path: Optional[Path],
    requested_format: Optional[str],
) -> Dict[str, Any]:
    selected = candidates
    if explicit_path:
        selected = [
            candidate
            for candidate in selected
            if Path(candidate["path"]).resolve() == explicit_path
            or explicit_path in Path(candidate["path"]).resolve().parents
        ]
        if not selected:
            raise ValueError(f"지정한 경로에서 변환 가능한 라벨을 찾지 못했습니다: {_relative(explicit_path, root)}")
    if requested_format:
        matching = [candidate for candidate in selected if candidate["format"] == requested_format]
        if not matching:
            raise ValueError(f"Workspace에서 요청한 입력 포맷 `{requested_format}` 라벨을 찾지 못했습니다.")
        selected = matching
    return selected[0]


def build_conversation_plan(request: str, workspace: str | Path) -> Dict[str, Any]:
    root = Path(workspace).expanduser().resolve()
    inventory = discover_workspace(root)
    action = _action(request)
    warnings = []

    if inventory["truncated"]:
        warnings.append("파일 수가 탐색 한도를 초과해 일부만 확인했습니다.")

    if action == "convert":
        formats = _target_formats(request)
        if not formats:
            raise ValueError("변환할 출력 포맷을 확인하지 못했습니다. 예: MS COCO 형식으로 바꿔줘")
        candidates = inventory["label_candidates"]
        if not candidates:
            raise ValueError("Workspace에서 변환 가능한 라벨 파일을 찾지 못했습니다.")
        images = inventory["image_directories"]
        if not images:
            raise ValueError("Workspace에서 라벨과 연결할 이미지 디렉터리를 찾지 못했습니다.")
        requested_source_format = _source_format(request)
        explicit_path = _explicit_workspace_path(request, root)
        selected = _select_label_candidate(
            candidates,
            root,
            explicit_path,
            requested_source_format,
        )
        duplicate_iou = _numeric_option(request, "iou")
        if len(candidates) > 1:
            warnings.append(
                f"라벨 후보 {len(candidates)}개 중 {selected['relative_path']}을 우선 선택했습니다."
            )
        operation = OperationPlan(
            action="convert",
            input_path=selected["path"],
            img_dir=images[0]["path"],
            out_dir=str((root / "data" / "converted").resolve()),
            formats=formats,
            source_format="auto",
            duplicate_iou=duplicate_iou if duplicate_iou is not None else 0.85,
            strict=_contains_any(request.lower(), ("strict", "엄격")),
            require_approval=False,
        )
        summary = (
            f"{selected['format']} 라벨 {selected['file_count']}개를 "
            f"{', '.join(formats)} 형식으로 변환"
        )
    elif action == "generate":
        images = inventory["image_directories"]
        if not images:
            raise ValueError("Workspace에서 라벨링할 이미지를 찾지 못했습니다.")
        task_type = _task_type(request)
        formats = _target_formats(request) or (["yolo"] if task_type == "object_detection" else ["vision_json"])
        threshold = _numeric_option(request, "threshold|신뢰도")
        plugin_config = (root / "configs" / "plugins.json").resolve()
        operation = OperationPlan(
            action="generate",
            task_type=task_type,
            img_dir=images[0]["path"],
            out_dir=str((root / "data" / "labeled").resolve()),
            vis_dir=str((root / "data" / "visualized").resolve()),
            formats=formats,
            threshold=threshold if threshold is not None else 0.75,
            prompt=request,
            plugin_config=str(plugin_config) if plugin_config.is_file() else None,
            require_approval=True,
        )
        summary = f"이미지 {images[0]['file_count']}개에 대해 {task_type} 라벨 생성"
    else:
        if _contains_any(request.lower(), ("원본 이미지랑 잘 맞", "이미지와 잘 맞", "image alignment")):
            raise ValueError(
                "라벨과 원본 이미지의 공간 정합성 검증은 현재 대화형 평가에서 지원하지 않습니다. "
                "현재 평가는 ground truth 비교 또는 run_metrics.csv 실험 비교를 지원합니다."
            )
        runs = _discover_runs(root)
        if not runs:
            for candidate in inventory["label_candidates"]:
                if candidate["format"] == "yolo":
                    runs[candidate["relative_path"]] = candidate["path"]
        if not runs:
            raise ValueError("Workspace에서 평가 가능한 run_metrics.csv를 찾지 못했습니다.")
        ground_truth = (root / "data" / "ground_truth").resolve()
        operation = OperationPlan(
            action="evaluate",
            gt_dir=str(ground_truth) if ground_truth.is_dir() else None,
            out_dir=str((root / "data" / "reports").resolve()),
            runs=runs,
            require_approval=False,
        )
        summary = f"발견한 실험 결과 {len(runs)}개를 비교 평가"

    plan = WorkflowPlan(request_summary=request[:200], operations=[operation])
    return {
        "summary": summary,
        "plan": plan.model_dump(),
        "warnings": warnings,
        "inventory": inventory,
    }


def describe_plan(proposal: Dict[str, Any], workspace: str | Path) -> str:
    root = Path(workspace).expanduser().resolve()
    operation = proposal["plan"]["operations"][0]
    lines = [f"**실행 계획:** {proposal['summary']}"]
    for key, label in (
        ("input_path", "입력 라벨"),
        ("img_dir", "이미지 위치"),
        ("out_dir", "출력 위치"),
    ):
        value = operation.get(key)
        if value:
            lines.append(f"- {label}: `{_relative(Path(value), root)}`")
    if operation.get("formats"):
        lines.append(f"- 출력 포맷: `{', '.join(operation['formats'])}`")
    if operation.get("duplicate_iou") != 0.85:
        lines.append(f"- 중복 IoU: `{operation['duplicate_iou']}`")
    if operation.get("strict"):
        lines.append("- 검증 이슈 레코드: `제외(strict)`")
    if operation.get("threshold") != 0.75:
        lines.append(f"- 신뢰도 기준: `{operation['threshold']}`")
    for warning in proposal.get("warnings", []):
        lines.append(f"- 확인 사항: {warning}")
    lines.append("이 계획으로 실행할까요?")
    return "\n".join(lines)


def describe_result(result: Dict[str, Any], workspace: str | Path) -> str:
    if result.get("status") != "completed":
        errors = result.get("errors") or ["알 수 없는 오류"]
        return "작업에 실패했습니다.\n\n" + "\n".join(f"- {error}" for error in errors)
    outputs = result.get("operation_outputs", [])
    lines = ["작업이 완료되었습니다."]
    for output in outputs:
        action = output.get("action")
        if action == "convert":
            lines.append(
                f"- 변환: {output.get('records_converted', 0)}/{output.get('records_read', 0)}개 레코드"
            )
            lines.append(f"- 입력 포맷: `{output.get('resolved_source_format', 'unknown')}`")
            lines.append(f"- 출력 포맷: `{', '.join(output.get('target_formats', []))}`")
        elif action == "generate":
            lines.append(f"- 처리 이미지: {output.get('images', 0)}개")
            lines.append(f"- 생성 라벨: {output.get('total_labels', 0)}개")
        elif action == "evaluate":
            lines.append(f"- 평가 결과: {len(output.get('rows', []))}개")
        report_path = output.get("report_path") or output.get("summary_path")
        if report_path:
            lines.append(f"- 결과 파일: `{_relative(Path(report_path), Path(workspace).resolve())}`")
    return "\n".join(lines)
