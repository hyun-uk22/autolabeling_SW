import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from ..utils.custom_label_mapper import (
    CUSTOM_MAPPING_FORMAT,
    infer_custom_mapping_spec_from_sample,
    sample_custom_label_file,
)
from ..utils import json_io
from ..utils.label_importer import extract_class_names_from_text
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
PREFERRED_LABEL_PATHS = (
    "data/labeled",
    "data/labels",
    "labels",
    "data/annotations",
    "annotations",
    "data/external_labels",
)
PREFERRED_IMAGE_PATHS = (
    "data/raw",
    "data/images",
    "images",
    "image",
    "data/img",
    "img",
    "JPEGImages",
)
TASK_PATH_HINTS = {
    "classification": ("classification", "classify", "image_classification", "cls"),
    "object_detection": ("detection", "object_detection", "detect", "det"),
    "segmentation": ("segmentation", "segment", "seg"),
    "pose_estimation": ("pose_estimation", "pose", "keypoint", "keypoints"),
    "ocr": ("ocr", "text", "text_recognition", "receipt", "document"),
    "tracking": ("tracking", "track", "video"),
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
    "custom_mapping": ("custom mapping", "custom_mapping", "custom json", "커스텀 json", "커스텀 포맷", "사용자 정의 포맷"),
    "mask_image": ("mask image", "mask images", "mask", "마스크 이미지", "마스크"),
}
MODEL_DATASET_ALIASES = {
    "segformer": ("segformer", "seg former"),
    "maskdino": ("maskdino", "mask dino"),
    "mask2former": ("mask2former", "mask 2 former"),
    "deeplabv3+": ("deeplabv3+", "deeplab", "deep lab"),
}
FRAMEWORK_ALIASES = {
    "huggingface": ("hugging face", "huggingface", "hf", "허깅페이스"),
    "mmsegmentation": ("mmsegmentation", "mmseg", "mmsegmentation", "mmseg멘테이션"),
    "detectron2": ("detectron2", "detectron"),
    "pytorch": ("pytorch", "torch", "파이토치"),
    "custom": ("custom", "커스텀", "직접"),
}
USAGE_MODE_ALIASES = {
    "official_repo": ("official repo", "official repository", "공식 repo", "공식 repository", "공식 레포", "git clone", "clone", "깃클론"),
    "library": ("library", "라이브러리", "패키지", "pip", "huggingface", "mmsegmentation", "detectron2"),
    "custom": ("custom", "커스텀", "직접"),
}
PURPOSE_ALIASES = {
    "training": ("training", "train", "학습", "훈련"),
    "inference": ("inference", "infer", "추론"),
    "evaluation": ("evaluation", "evaluate", "평가"),
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
    if _is_coco_like(data):
        return "coco"
    if isinstance(data, list) and data and all(isinstance(item, dict) for item in data[:10]):
        return "generic_json"
    if isinstance(data, dict) and any(
        key in data for key in ("image", "image_name", "file_name", "boxes", "bbox", "segments", "objects")
    ):
        return "generic_json"
    if any(isinstance(item, dict) and "bbox" in item for item in iter_dicts(data)):
        return "generic_json"
    return None


def _is_coco_like(data: Any) -> bool:
    if not isinstance(data, dict) or not {"images", "annotations", "categories"}.issubset(data):
        return False
    images = data.get("images") or []
    annotations = data.get("annotations") or []
    categories = data.get("categories") or []
    if not isinstance(images, list) or not isinstance(annotations, list) or not isinstance(categories, list):
        return False
    if images and not all(isinstance(item, dict) and "id" in item and "file_name" in item for item in images[:10]):
        return False
    if annotations and not all(
        isinstance(item, dict) and {"image_id", "category_id", "bbox"}.issubset(item) for item in annotations[:10]
    ):
        return False
    if categories and not all(isinstance(item, dict) and "id" in item and "name" in item for item in categories[:10]):
        return False
    return True


def iter_dicts(value):
    if isinstance(value, dict):
        yield value
        for child in value.values():
            yield from iter_dicts(child)
    elif isinstance(value, list):
        for child in value:
            yield from iter_dicts(child)


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
    preferred_rank = 0
    for index, prefix in enumerate(PREFERRED_LABEL_PATHS):
        if relative == prefix.lower() or relative.startswith(prefix.lower() + "/"):
            preferred_rank = len(PREFERRED_LABEL_PATHS) - index
            break
    return preferred_rank, candidate["file_count"], -len(relative)


def _image_candidate_score(image: Dict[str, Any], label: Dict[str, Any], root: Path) -> tuple:
    image_path = Path(image["path"]).resolve()
    label_path = Path(label["path"]).resolve()
    image_relative = image["relative_path"].lower()
    label_relative = label["relative_path"].lower()

    preferred_rank = 0
    for index, prefix in enumerate(PREFERRED_IMAGE_PATHS):
        if image_relative == prefix.lower() or image_relative.startswith(prefix.lower() + "/"):
            preferred_rank = len(PREFERRED_IMAGE_PATHS) - index
            break

    relation_rank = 0
    if image_path.parent == label_path.parent:
        relation_rank = 5
    elif image_path.parent == label_path.parent.parent:
        relation_rank = 4
    elif label_path.parent == image_path.parent:
        relation_rank = 3
    else:
        try:
            label_parts = label_path.relative_to(root).parts
            image_parts = image_path.relative_to(root).parts
        except ValueError:
            label_parts = ()
            image_parts = ()
        if label_parts and image_parts and label_parts[0] == image_parts[0]:
            relation_rank = 2

    label_tokens = {"label", "labels", "labeled", "annotation", "annotations"}
    image_tokens = {"image", "images", "raw", "img", "jpegimages"}
    sibling_rank = 0
    if label_path.name.lower() in label_tokens and image_path.name.lower() in image_tokens:
        if label_path.parent == image_path.parent:
            sibling_rank = 5
        elif label_path.parent.parent == image_path.parent.parent:
            sibling_rank = 3

    return sibling_rank, relation_rank, preferred_rank, image["file_count"], -len(image_relative)


def _select_image_for_label(
    images: List[Dict[str, Any]],
    label: Dict[str, Any],
    root: Path,
    formats: Optional[Iterable[str]] = None,
) -> Dict[str, Any]:
    if not images:
        return {
            "path": label["path"] if Path(label["path"]).is_dir() else str(Path(label["path"]).parent),
            "relative_path": label["relative_path"],
            "file_count": 0,
            "missing": True,
            "requires_image_size": bool(set(formats or []) & {"coco", "pascal_voc"}),
        }
    return max(images, key=lambda image: _image_candidate_score(image, label, root))


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


def _mentioned_model(request: str) -> Optional[str]:
    lowered = request.lower().replace("-", " ")
    for model, aliases in MODEL_DATASET_ALIASES.items():
        if any(alias in lowered for alias in aliases):
            return model
    return None


def _mentioned_framework(request: str) -> Optional[str]:
    lowered = request.lower().replace("-", " ")
    for framework, aliases in FRAMEWORK_ALIASES.items():
        if any(alias in lowered for alias in aliases):
            return framework
    return None


def _mentioned_usage_mode(request: str) -> str:
    lowered = request.lower().replace("-", " ")
    for mode, aliases in USAGE_MODE_ALIASES.items():
        if any(alias in lowered for alias in aliases):
            return mode
    return "library"


def _mentioned_purpose(request: str) -> Optional[str]:
    lowered = request.lower().replace("-", " ")
    for purpose, aliases in PURPOSE_ALIASES.items():
        if any(alias in lowered for alias in aliases):
            return purpose
    return None


def _repo_url_from_request(request: str) -> Optional[str]:
    match = re.search(r"https?://[^\s'\"`]+", request)
    if not match:
        return None
    return match.group(0).rstrip(".,)")


def _repo_path_from_request(request: str, root: Path) -> Optional[str]:
    matches = []
    matches.extend(
        match.group(1).strip().strip("'\"")
        for match in re.finditer(r"(?im)^\s*(?:repo|repository|레포|저장소)\s*:\s*(.+?)\s*$", request)
    )
    matches.extend(
        match.group(1).strip().strip("'\"")
        for match in re.finditer(
            r"(?:repo|repository|레포|저장소)\s*(?:경로|위치|path)?\s*(?:은|는|을|를|로|:|=)?\s*([A-Za-z]:[\\/][^\r\n]+|[\w.-]+(?:[\\/][\w.-]+)+)",
            request,
            re.IGNORECASE,
        )
    )
    for raw in matches:
        if raw.startswith(("http://", "https://")):
            continue
        candidate = Path(raw)
        resolved = candidate.resolve() if candidate.is_absolute() else (root / candidate).resolve()
        return str(resolved)
    return None


def _model_task(model_name: Optional[str], request: str) -> str:
    lowered = request.lower()
    if _contains_any(lowered, ("panoptic", "파노픽")):
        return "panoptic_segmentation"
    if _contains_any(lowered, ("instance", "인스턴스")) or model_name in {"maskdino", "mask2former"}:
        return "instance_segmentation"
    return "semantic_segmentation"


def _output_path_from_request(request: str, root: Path) -> Optional[Path]:
    matches = []
    matches.extend(
        match.group(1).strip().strip("'\"")
        for match in re.finditer(r"(?im)^\s*(?:out|output|출력|저장)\s*:\s*(.+?)\s*$", request)
    )
    matches.extend(
        match.group(1).strip().strip("'\"")
        for match in re.finditer(
            r"(?:출력|저장)\s*(?:위치|경로|폴더)?\s*(?:은|는|을|를|로|:|=)?\s*([A-Za-z]:[\\/][^\r\n]+|[\w.-]+(?:[\\/][\w.-]+)+)",
            request,
        )
    )
    matches.extend(
        match.group(1).strip().strip("'\"")
        for match in re.finditer(r"(?:출력|저장|output)[^A-Za-z0-9가-힣]{0,10}(?:위치|경로|폴더|dir)?[^A-Za-z0-9가-힣]{0,10}([A-Za-z]:[\\/][^\r\n]+|[\w.-]+(?:[\\/][\w.-]+)+)", request, re.IGNORECASE)
    )
    for raw in matches:
        candidate = Path(raw)
        resolved = candidate.resolve() if candidate.is_absolute() else (root / candidate).resolve()
        try:
            resolved.relative_to(root)
        except ValueError as exc:
            raise ValueError("Workspace 밖의 출력 경로는 사용할 수 없습니다.") from exc
        return resolved
    return None


def _split_policy(request: str) -> Dict[str, Optional[float]]:
    lowered = request.lower()
    match = re.search(r"train\D{0,10}([0-9]+(?:\.[0-9]+)?)\D{0,10}val\D{0,10}([0-9]+(?:\.[0-9]+)?)", lowered)
    if not match:
        match = re.search(r"학습\D{0,10}([0-9]+(?:\.[0-9]+)?)\D{0,10}(?:검증|val)\D{0,10}([0-9]+(?:\.[0-9]+)?)", lowered)
    if match:
        train = float(match.group(1))
        val = float(match.group(2))
        total = train + val
        if total > 1.0:
            train /= total
            val /= total
        return {"train": train, "val": val, "test": None}
    ratio = re.search(r"([0-9]+)\s*[:/]\s*([0-9]+)(?:\s*[:/]\s*([0-9]+))?", lowered)
    if ratio:
        values = [float(value) for value in ratio.groups() if value is not None]
        total = sum(values)
        return {
            "train": values[0] / total,
            "val": values[1] / total if len(values) > 1 else None,
            "test": values[2] / total if len(values) > 2 else None,
        }
    return {"train": None, "val": None, "test": None}


def diagnose_model_dataset_request(request: str, workspace: str | Path) -> Optional[Dict[str, Any]]:
    root = Path(workspace).expanduser().resolve()
    model_name = _mentioned_model(request)
    lowered = request.lower()
    layout_requested = _contains_any(lowered, ("데이터셋", "dataset", "폴더", "구조", "양식", "학습용", "training"))
    if not model_name or not layout_requested:
        return None
    usage_mode = _mentioned_usage_mode(request)
    framework = _mentioned_framework(request)
    purpose = _mentioned_purpose(request)
    source_formats = _mentioned_formats(request, SOURCE_FORMAT_ALIASES)
    out_dir = _output_path_from_request(request, root)
    repo_url = _repo_url_from_request(request)
    repo_path = _repo_path_from_request(request, root)
    split = _split_policy(request)
    task_type = _model_task(model_name, request)

    missing = []
    if not framework:
        missing.append("framework")
    if not purpose:
        missing.append("purpose")
    if not source_formats:
        missing.append("source_label_format")
    if not out_dir:
        missing.append("output_dir")
    if usage_mode == "official_repo" and not (repo_url or repo_path):
        missing.append("repo_source")
    if purpose == "training" and (split["train"] is None or split["val"] is None):
        missing.append("split_policy")

    questions = {
        "framework": "사용 프레임워크를 알려주세요. 예: Hugging Face, MMSegmentation, Detectron2, PyTorch, custom",
        "purpose": "목적을 알려주세요. 예: training, inference, evaluation",
        "source_label_format": "현재 라벨 형식을 알려주세요. 예: COCO, Pascal VOC, YOLO, vision_json, mask image",
        "output_dir": "출력 위치를 알려주세요. 예: datasets/segformer",
        "repo_source": "공식 repo를 사용할 경우 clone URL 또는 로컬 repo 경로를 알려주세요. 예: repo: https://github.com/... 또는 repo: external/MaskDINO",
        "split_policy": "학습/검증/테스트 분할 방식을 알려주세요. 예: train 8, val 2",
    }
    source_format = source_formats[0] if source_formats else None
    incompatible = []
    if model_name == "segformer" and source_format == "yolo":
        incompatible.append("SegFormer semantic segmentation 학습에는 bbox-only YOLO 라벨만으로는 정확한 class mask를 만들 수 없습니다. polygon/mask 라벨이 필요합니다.")
    return {
        "model_name": model_name,
        "usage_mode": usage_mode,
        "framework": framework,
        "purpose": purpose,
        "repo_url": repo_url,
        "repo_path": repo_path,
        "task_type": task_type,
        "source_label_format": source_format,
        "out_dir": str(out_dir) if out_dir else None,
        "split": split,
        "missing_required_info": missing,
        "questions": [questions[item] for item in missing],
        "incompatible_warnings": incompatible,
        "can_create_plan": not missing and not incompatible,
    }


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
                if re.match(r"\s*(?:라벨|데이터셋|파일|형식|포맷|규격)", suffix):
                    if re.match(r"\s*(?:라벨|데이터셋|파일)", suffix):
                        return name
                    if re.match(
                        r"\s*(?:형식|포맷|규격)\s*(?:이야|입니다|이고|이며|이다|임|은|는|라고|로 되어|으로 되어)",
                        suffix,
                    ):
                        return name
    return None


def _action(request: str) -> str:
    lowered = request.lower()
    if _mentioned_model(request) and _contains_any(
        lowered,
        ("데이터셋", "dataset", "폴더", "구조", "양식", "학습용", "training"),
    ):
        return "prepare_model_dataset"
    if _contains_any(lowered, ("평가", "비교", "리포트", "evaluate", "evaluation")):
        return "evaluate"
    if _contains_any(lowered, ("변환", "바꿔", "바꾸", "변경", "통일", "convert", "format")):
        return "convert"
    if _contains_any(lowered, ("생성", "라벨링", "라벨링해", "만들어", "검출", "탐지", "진행", "generate", "detect", "segment", "label image")):
        return "generate"
    raise ValueError("요청에서 작업 종류를 확인하지 못했습니다. 변환, 라벨 생성 또는 평가 작업을 명시해 주세요.")


def _is_label_editor_request(request: str) -> bool:
    lowered = request.lower()
    label_context = _contains_any(lowered, ("라벨", "label", "annotation", "어노테이션", "데이터셋"))
    edit_context = _contains_any(
        lowered,
        (
            "편집",
            "수정",
            "고치",
            "검수",
            "리뷰",
            "라벨 에디터",
            "label editor",
            "edit label",
            "edit annotation",
        ),
    )
    return label_context and edit_context


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


def _explicit_workspace_path(request: str, root: Path, allow_external: bool = False) -> Optional[Path]:
    raw_paths = []
    raw_paths.extend(
        match.group(1).strip().strip("'\"")
        for match in re.finditer(r"(?im)^\s*path\s*:\s*(.+?)\s*$", request)
    )
    raw_paths.extend(
        match.group(1).strip().strip("'\"")
        for match in re.finditer(r"(?<![\w.])([A-Za-z]:[\\/][^\r\n]+)", request)
    )
    raw_paths.extend(re.findall(r"(?<![\w.])([\w.-]+(?:[\\/][\w.-]+)+)", request))

    for raw in raw_paths:
        candidate = Path(raw)
        if candidate.is_absolute():
            resolved = candidate.resolve()
        else:
            resolved = (root / candidate).resolve()
        if not allow_external:
            try:
                resolved.relative_to(root)
            except ValueError as exc:
                raise ValueError("Workspace 밖의 경로는 대화형 자동 실행에서 사용할 수 없습니다.") from exc
        if not resolved.exists():
            raise ValueError(f"요청에 지정된 경로가 존재하지 않습니다: {_relative(resolved, root)}")
        return resolved
    return None


def _image_directory_from_path(path: Path, root: Path) -> Dict[str, Any]:
    directory = path if path.is_dir() else path.parent
    if not directory.is_dir():
        raise ValueError(f"지정한 이미지 경로가 디렉터리가 아닙니다: {_relative(directory, root)}")

    def allowed_image(item: Path) -> bool:
        if not item.is_file() or item.suffix.lower() not in IMAGE_EXTENSIONS:
            return False
        try:
            item.relative_to(root)
        except ValueError:
            return True
        return not _is_ignored(item, root)

    image_files = [
        item
        for item in directory.rglob("*")
        if allowed_image(item)
    ]
    if not image_files:
        raise ValueError(f"지정한 경로에서 라벨링할 이미지를 찾지 못했습니다: {_relative(directory, root)}")
    return {
        "path": str(directory),
        "relative_path": _relative(directory, root),
        "file_count": len(image_files),
    }


def _select_image_directory(
    images: List[Dict[str, Any]],
    root: Path,
    explicit_path: Optional[Path],
    task_type: str = "object_detection",
) -> Dict[str, Any]:
    selected = images
    if explicit_path:
        matching = [
            image
            for image in selected
            if Path(image["path"]).resolve() == explicit_path
            or Path(image["path"]).resolve() in explicit_path.parents
            or explicit_path in Path(image["path"]).resolve().parents
        ]
        if not matching:
            raise ValueError(f"지정한 경로에서 라벨링할 이미지를 찾지 못했습니다: {_relative(explicit_path, root)}")
        selected = matching

    hints = TASK_PATH_HINTS.get(task_type, ())

    def score(image: Dict[str, Any]) -> tuple:
        relative = image["relative_path"].lower().replace("\\", "/")
        parts = set(Path(relative).parts)
        task_rank = 0
        for index, hint in enumerate(hints):
            normalized_hint = hint.lower()
            if normalized_hint in parts:
                task_rank = len(hints) + 2 - index
                break
            if normalized_hint in relative:
                task_rank = len(hints) + 1 - index
                break
        preferred_rank = 0
        for index, prefix in enumerate(PREFERRED_IMAGE_PATHS):
            normalized_prefix = prefix.lower()
            if relative == normalized_prefix or relative.startswith(normalized_prefix + "/"):
                preferred_rank = len(PREFERRED_IMAGE_PATHS) - index
                break
        return task_rank, preferred_rank, image["file_count"], -len(relative)

    return max(selected, key=score)


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
        if requested_format == CUSTOM_MAPPING_FORMAT:
            matching = [
                candidate
                for candidate in selected
                if candidate["format"] in {"generic_json", "coco"}
            ]
        else:
            matching = [candidate for candidate in selected if candidate["format"] == requested_format]
        if not matching:
            raise ValueError(f"Workspace에서 요청한 입력 포맷 `{requested_format}` 라벨을 찾지 못했습니다.")
        selected = matching
    return selected[0]


def _custom_label_field_from_request(request: str) -> Optional[str]:
    patterns = (
        r'라벨[^\r\n]{0,80}"[^"]+"\s*(?:의|에서|안의)\s*"([^"]+)"',
        r"라벨[^\r\n]{0,80}'[^']+'\s*(?:의|에서|안의)\s*'([^']+)'",
        r'라벨[^\r\n]{0,80}"([A-Za-z_][A-Za-z0-9_]*)"',
        r"라벨[^\r\n]{0,80}'([A-Za-z_][A-Za-z0-9_]*)'",
        r"(?:label\s*(?:name|field)?)[^\r\n]{0,60}`?([A-Za-z_][A-Za-z0-9_]*)`?",
    )
    for pattern in patterns:
        match = re.search(pattern, request, flags=re.IGNORECASE)
        if match:
            return match.group(1)
    return None


def _custom_mapping_for_path(input_path: str, request: str = "") -> tuple[str, Optional[str]]:
    sample_path = sample_custom_label_file(input_path)
    spec = infer_custom_mapping_spec_from_sample(sample_path)
    label_field = _custom_label_field_from_request(request)
    if label_field:
        spec["label_path"] = f"@.{label_field}"
    return json_io.dumps(spec, ensure_ascii=False, indent=2), label_field


def _should_use_custom_mapping(
    selected: Dict[str, Any],
    requested_source_format: Optional[str],
) -> bool:
    if requested_source_format == CUSTOM_MAPPING_FORMAT:
        return True
    if requested_source_format:
        return False
    return selected["format"] == "generic_json"


def build_label_editor_request(
    request: str,
    workspace: str | Path,
    intent_overrides: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    overrides = intent_overrides or {}
    if not _is_label_editor_request(request) and overrides.get("action") != "open_editor":
        return None

    root = Path(workspace).expanduser().resolve()
    inventory = discover_workspace(root)
    candidates = inventory["label_candidates"]
    if not candidates:
        raise ValueError("Workspace에서 편집 가능한 라벨 파일을 찾지 못했습니다.")

    requested_source_format = overrides.get("source_format") or _source_format(request)
    source_path = overrides.get("source_path")
    explicit_path = _explicit_workspace_path(source_path, root) if source_path else _explicit_workspace_path(request, root)
    selected = _select_label_candidate(candidates, root, explicit_path, requested_source_format)
    selected_images = _select_image_for_label(inventory["image_directories"], selected, root)
    task_type = overrides.get("task_type") or _task_type(request)

    output_path = Path(selected["path"])
    output_dir = output_path if output_path.is_dir() else output_path.parent
    source_format = requested_source_format or "auto"
    context = {
        "label_path": selected["path"],
        "image_dir": selected_images["path"],
        "source_format": source_format,
        "classes_path": "",
        "task_type": task_type,
        "output_dir": str(output_dir.resolve()),
    }

    warnings = []
    if len(candidates) > 1:
        warnings.append(f"라벨 후보 {len(candidates)}개 중 {selected['relative_path']}을 편집 대상으로 선택했습니다.")
    if selected_images.get("missing"):
        warnings.append("이미지 폴더를 찾지 못해 라벨 경로를 이미지 경로로 임시 지정했습니다. 이미지 시각화 없이 표 편집 위주로 동작할 수 있습니다.")
    elif len(inventory["image_directories"]) > 1:
        warnings.append(f"이미지 후보 {len(inventory['image_directories'])}개 중 {selected_images['relative_path']}을 연결했습니다.")

    lines = [
        "라벨 편집 탭으로 이동합니다.",
        "",
        f"- 편집 라벨: `{selected['relative_path']}`",
        f"- 이미지 폴더: `{selected_images['relative_path']}`",
        f"- 라벨 형식: `{source_format}`",
        f"- 편집 태스크: `{task_type}`",
    ]
    if warnings:
        lines.append("")
        lines.append("확인 사항:")
        lines.extend(f"- {warning}" for warning in warnings)

    return {
        "kind": "open_editor",
        "response": "\n".join(lines),
        "editor_context": context,
        "warnings": warnings,
        "inventory": inventory,
    }


def build_conversation_plan(
    request: str,
    workspace: str | Path,
    intent_overrides: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    root = Path(workspace).expanduser().resolve()
    inventory = discover_workspace(root)
    overrides = intent_overrides or {}
    action = overrides.get("action") or _action(request)
    warnings = []

    if inventory["truncated"]:
        warnings.append("파일 수가 탐색 한도를 초과해 일부만 확인했습니다.")

    if action == "convert":
        formats = overrides.get("target_formats") or _target_formats(request)
        if not formats:
            raise ValueError("변환할 출력 포맷을 확인하지 못했습니다. 예: MS COCO 형식으로 바꿔줘")
        candidates = inventory["label_candidates"]
        if not candidates:
            raise ValueError("Workspace에서 변환 가능한 라벨 파일을 찾지 못했습니다.")
        requested_source_format = overrides.get("source_format") or _source_format(request)
        source_path = overrides.get("source_path")
        explicit_path = _explicit_workspace_path(source_path, root) if source_path else _explicit_workspace_path(request, root)
        selected = _select_label_candidate(
            candidates,
            root,
            explicit_path,
            requested_source_format,
        )
        selected_images = _select_image_for_label(inventory["image_directories"], selected, root, formats)
        duplicate_iou = overrides.get("duplicate_iou")
        if duplicate_iou is None:
            duplicate_iou = _numeric_option(request, "iou")
        if len(candidates) > 1:
            warnings.append(
                f"라벨 후보 {len(candidates)}개 중 {selected['relative_path']}을 우선 선택했습니다."
            )
        if len(inventory["image_directories"]) > 1:
            warnings.append(
                f"이미지 후보 {len(inventory['image_directories'])}개 중 {selected_images['relative_path']}을 라벨과 연결했습니다."
            )
        if selected_images.get("missing"):
            if selected_images.get("requires_image_size"):
                warnings.append(
                    "원본 이미지 디렉터리를 찾지 못했습니다. COCO/Pascal VOC 출력은 이미지 크기가 필요하므로 "
                    "실행은 가능하지만 해당 포맷 출력이 누락되고 리포트에 missing_image가 기록될 수 있습니다."
                )
            else:
                warnings.append(
                    "원본 이미지 디렉터리를 찾지 못했습니다. YOLO/Vision JSON처럼 이미지 크기가 필요 없는 출력은 계속 진행합니다."
                )
        use_custom_mapping = _should_use_custom_mapping(selected, requested_source_format)
        source_format = CUSTOM_MAPPING_FORMAT if use_custom_mapping else (requested_source_format or "auto")
        custom_label_mapping = None
        input_path = selected["path"]
        if use_custom_mapping:
            selected_path = Path(selected["path"])
            if explicit_path and explicit_path.is_dir():
                input_path = str(explicit_path)
            elif selected_path.is_file():
                input_path = str(selected_path.parent)
            custom_label_mapping, label_field_override = _custom_mapping_for_path(input_path, request)
            if requested_source_format == CUSTOM_MAPPING_FORMAT:
                warnings.append("커스텀 라벨 샘플을 분석해 custom_mapping 매핑 스펙을 자동 생성했습니다.")
            else:
                warnings.append("generic JSON 라벨을 custom_mapping으로 자동 분석해 변환합니다.")
            if label_field_override:
                warnings.append(f"프롬프트에서 지정한 라벨 필드 `{label_field_override}`를 custom_mapping에 반영했습니다.")
        operation = OperationPlan(
            action="convert",
            input_path=input_path,
            img_dir=selected_images["path"],
            out_dir=str((root / "data" / "converted").resolve()),
            formats=formats,
            source_format=source_format,
            custom_label_mapping=custom_label_mapping,
            duplicate_iou=duplicate_iou if duplicate_iou is not None else 0.85,
            strict=bool(overrides.get("strict")) or _contains_any(request.lower(), ("strict", "엄격")),
            require_approval=False,
        )
        summary = (
            f"{selected['format']} 라벨 {selected['file_count']}개를 "
            f"{', '.join(formats)} 형식으로 변환"
        )
    elif action == "generate":
        source_path = overrides.get("source_path")
        explicit_path = (
            _explicit_workspace_path(source_path, root, allow_external=True)
            if source_path
            else _explicit_workspace_path(request, root, allow_external=True)
        )
        task_type = overrides.get("task_type") or _task_type(request)
        images = inventory["image_directories"]
        if explicit_path:
            selected_images = _image_directory_from_path(explicit_path, root)
        else:
            if not images:
                raise ValueError(
                    "Workspace에서 라벨링할 이미지를 찾지 못했습니다. "
                    "이미지를 workspace의 data/raw에 넣거나 프롬프트에 `path: 이미지_폴더`를 지정하세요."
                )
            selected_images = _select_image_directory(images, root, explicit_path, task_type)
        formats = overrides.get("target_formats") or _target_formats(request) or (["yolo"] if task_type == "object_detection" else ["vision_json"])
        threshold = overrides.get("threshold")
        if threshold is None:
            threshold = _numeric_option(request, "threshold|신뢰도")
        plugin_config = (root / "configs" / "plugins.json").resolve()
        normalized_class_labels = [
            str(label).strip()
            for label in overrides.get("normalized_class_labels", [])
            if str(label).strip()
        ]
        generation_prompt = request
        if normalized_class_labels:
            generation_prompt = (
                f"{request}\n"
                "classes: " + ", ".join(dict.fromkeys(normalized_class_labels))
            )
            warnings.append(
                "LLM이 프롬프트의 검출 대상 클래스를 영어 specialist labels로 정규화했습니다: "
                + ", ".join(dict.fromkeys(normalized_class_labels))
            )
        if overrides.get("class_normalizer_error"):
            warnings.append(f"클래스명 LLM 정규화에 실패해 규칙 기반 후보만 사용합니다: {overrides['class_normalizer_error']}")
        operation = OperationPlan(
            action="generate",
            task_type=task_type,
            img_dir=selected_images["path"],
            out_dir=str((root / "data" / "labeled").resolve()),
            vis_dir=str((root / "data" / "visualized").resolve()),
            formats=formats,
            threshold=threshold if threshold is not None else 0.75,
            prompt=generation_prompt,
            specialist_consistency_runs=0,
            specialist_advisor_mode="none",
            plugin_config=str(plugin_config) if plugin_config.is_file() else None,
            require_approval=True,
        )
        if task_type in {"object_detection", "segmentation", "tracking"} and not extract_class_names_from_text(generation_prompt):
            warnings.append(
                "Grounding DINO/Grounded SAM2 기반 생성은 찾을 클래스명이 필요합니다. "
                "자유 문장으로 `person, car만 찾아줘`, `'person', 'car' 기반으로 진행해줘`, "
                "`사람 차 찾아줘`처럼 입력하거나 classes.txt/data.yaml을 지정하세요."
            )
        summary = f"이미지 {selected_images['file_count']}개에 대해 {task_type} 라벨 생성"
    else:
        if action == "prepare_model_dataset":
            diagnosis = diagnose_model_dataset_request(request, root)
            if not diagnosis:
                raise ValueError("모델 데이터셋 준비 요청을 해석하지 못했습니다.")
            if diagnosis["missing_required_info"] or diagnosis["incompatible_warnings"]:
                messages = []
                messages.extend(diagnosis["questions"])
                messages.extend(diagnosis["incompatible_warnings"])
                raise ValueError("모델 데이터셋 준비에 필요한 정보가 부족합니다: " + " / ".join(messages))
            operation = OperationPlan(
                action="prepare_model_dataset",
                model_name=diagnosis["model_name"],
                usage_mode=diagnosis["usage_mode"],
                framework=diagnosis["framework"],
                dataset_purpose=diagnosis["purpose"],
                repo_url=diagnosis["repo_url"],
                repo_path=diagnosis["repo_path"],
                task_type=diagnosis["task_type"],
                source_format=diagnosis["source_label_format"] or "auto",
                out_dir=diagnosis["out_dir"],
                output_layout=f"{diagnosis['model_name']}_{diagnosis['usage_mode']}_{diagnosis['framework']}",
                split_train=diagnosis["split"].get("train"),
                split_val=diagnosis["split"].get("val"),
                split_test=diagnosis["split"].get("test"),
                require_approval=True,
            )
            summary = (
                f"{diagnosis['model_name']} {diagnosis['usage_mode']} {diagnosis['framework']} "
                f"{diagnosis['purpose']}용 데이터셋 구조 생성"
            )
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
    if operation.get("action") == "prepare_model_dataset":
        lines.append(f"- 모델: `{operation.get('model_name')}`")
        lines.append(f"- 사용 방식: `{operation.get('usage_mode', 'library')}`")
        lines.append(f"- 프레임워크: `{operation.get('framework')}`")
        if operation.get("repo_url"):
            lines.append(f"- 공식 repo URL: `{operation.get('repo_url')}`")
        if operation.get("repo_path"):
            lines.append(f"- 로컬 repo 경로: `{_relative(Path(operation['repo_path']), root)}`")
        lines.append(f"- 목적: `{operation.get('dataset_purpose')}`")
        lines.append(f"- 태스크: `{operation.get('task_type')}`")
        lines.append(f"- 입력 라벨 형식: `{operation.get('source_format')}`")
        split_parts = [
            f"train={operation.get('split_train'):.2f}" if operation.get("split_train") is not None else None,
            f"val={operation.get('split_val'):.2f}" if operation.get("split_val") is not None else None,
            f"test={operation.get('split_test'):.2f}" if operation.get("split_test") is not None else None,
        ]
        split_text = ", ".join(part for part in split_parts if part)
        if split_text:
            lines.append(f"- 분할: `{split_text}`")
    if operation.get("duplicate_iou") != 0.85:
        lines.append(f"- 중복 IoU: `{operation['duplicate_iou']}`")
    if operation.get("strict"):
        lines.append("- 검증 이슈 레코드: `제외(strict)`")
    if operation.get("threshold") != 0.75:
        lines.append(f"- 신뢰도 기준: `{operation['threshold']}`")
    if operation.get("action") == "generate":
        lines.append(f"- 태스크: `{operation.get('task_type', 'object_detection')}`")
        specialist_runs = int(operation.get("specialist_consistency_runs", 0) or 0)
        advisor_mode = operation.get("specialist_advisor_mode", "none")
        if specialist_runs > 0:
            lines.append(f"- Specialist 재추론: `{specialist_runs}회`")
        if advisor_mode and advisor_mode != "none":
            lines.append(f"- 재추론 Advisor: `{advisor_mode}`")
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
        elif action == "prepare_model_dataset":
            lines.append(f"- 모델 데이터셋 구조: `{output.get('layout', '-')}`")
            lines.append(f"- 생성 폴더: {len(output.get('created_directories', []))}개")
        report_path = output.get("report_path") or output.get("summary_path")
        if report_path:
            lines.append(f"- 결과 파일: `{_relative(Path(report_path), Path(workspace).resolve())}`")
    return "\n".join(lines)
