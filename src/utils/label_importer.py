import csv
import json
import os
import ast
import xml.etree.ElementTree as ET
from collections import Counter
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

from ..core.llm_client import normalize_confidence, normalize_coordinate
from ..core.models import BoundingBox, DetectionResult, Point, PolygonSegment


IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".webp", ".bmp")
YOLO_DATASET_FILENAMES = ("data.yaml", "data.yml", "dataset.yaml", "dataset.yml")
SOURCE_PRIORITY = {
    "coco": 60,
    "pascal_voc": 50,
    "yolo": 40,
    "vision_json": 30,
    "csv": 20,
    "generic_json": 10,
}
CLASS_LIST_PRIORITY = {
    "yolo": 100,
    "coco": 80,
    "pascal_voc": 70,
    "vision_json": 60,
    "csv": 50,
    "generic_json": 40,
}


@dataclass(frozen=True)
class LabelSource:
    path: str
    format: str
    search_root: Optional[str] = None


@dataclass
class LabelImportBatch:
    records: List[Tuple[str, DetectionResult]]
    report: Dict[str, Any]


def find_image_path(image_dir: str, image_name: str) -> str:
    candidate = os.path.join(image_dir, image_name)
    if os.path.exists(candidate):
        return candidate
    stem = os.path.splitext(image_name)[0]
    for ext in IMAGE_EXTS:
        candidate = os.path.join(image_dir, stem + ext)
        if os.path.exists(candidate):
            return candidate
    return os.path.join(image_dir, image_name)


def infer_label_format(input_path: str) -> str:
    if os.path.isdir(input_path):
        files = os.listdir(input_path)
        if any(name.lower().endswith(".xml") for name in files):
            return "pascal_voc"
        if any(name.lower().endswith(".txt") for name in files):
            return "yolo"
        if any(name.lower().endswith(".jsonl") for name in files):
            return "vision_json"
    ext = os.path.splitext(input_path)[1].lower()
    if ext == ".xml":
        return "pascal_voc"
    if ext == ".jsonl":
        return "vision_json"
    if ext == ".csv":
        return "csv"
    if ext == ".txt":
        return "yolo"
    if ext == ".json":
        with open(input_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict) and {"images", "annotations", "categories"}.issubset(data.keys()):
            return "coco"
        return "generic_json"
    raise ValueError(f"Cannot infer label format from {input_path}")


def _strip_yaml_value(value: str) -> str:
    value = value.strip().strip(",")
    if "#" in value:
        value = value.split("#", 1)[0].strip()
    if (
        (value.startswith('"') and value.endswith('"'))
        or (value.startswith("'") and value.endswith("'"))
    ):
        return value[1:-1]
    return value


def _parse_inline_yaml_names(value: str) -> List[str]:
    value = value.strip()
    if not value:
        return []
    if value.startswith("["):
        try:
            parsed = ast.literal_eval(value)
            if isinstance(parsed, list):
                return [str(item).strip() for item in parsed if str(item).strip()]
        except (SyntaxError, ValueError):
            inner = value.strip("[]")
            return [_strip_yaml_value(item) for item in inner.split(",") if _strip_yaml_value(item)]
    if value.startswith("{"):
        inner = value.strip("{}")
        entries = []
        for item in inner.split(","):
            if ":" not in item:
                continue
            key, label = item.split(":", 1)
            try:
                index = int(_strip_yaml_value(key))
            except ValueError:
                index = len(entries)
            entries.append((index, _strip_yaml_value(label)))
        return [label for _, label in sorted(entries) if label]
    return [_strip_yaml_value(value)]


def load_yolo_yaml_classes(yaml_path: Optional[str]) -> List[str]:
    if not yaml_path or not os.path.exists(yaml_path):
        return []
    with open(yaml_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    for index, line in enumerate(lines):
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or not stripped.startswith("names:"):
            continue
        indent = len(line) - len(line.lstrip())
        inline_value = stripped.split(":", 1)[1].strip()
        if inline_value:
            return _parse_inline_yaml_names(inline_value)

        ordered_entries = []
        list_entries = []
        for child in lines[index + 1:]:
            child_stripped = child.strip()
            if not child_stripped or child_stripped.startswith("#"):
                continue
            child_indent = len(child) - len(child.lstrip())
            if child_indent <= indent:
                break
            if child_stripped.startswith("-"):
                label = _strip_yaml_value(child_stripped[1:])
                if label:
                    list_entries.append(label)
                continue
            if ":" in child_stripped:
                key, label = child_stripped.split(":", 1)
                try:
                    class_id = int(_strip_yaml_value(key))
                except ValueError:
                    class_id = len(ordered_entries)
                label = _strip_yaml_value(label)
                if label:
                    ordered_entries.append((class_id, label))
        if ordered_entries:
            return [label for _, label in sorted(ordered_entries)]
        return list_entries
    return []


def _candidate_yolo_class_paths(
    source_path: str,
    classes_path: Optional[str] = None,
    search_root: Optional[str] = None,
) -> List[str]:
    if classes_path:
        return [classes_path]
    base_dir = source_path if os.path.isdir(source_path) else os.path.dirname(source_path)
    search_dirs = []
    current = os.path.abspath(base_dir)
    root = os.path.abspath(search_root) if search_root else current
    while True:
        search_dirs.append(current)
        if current == root or os.path.dirname(current) == current:
            break
        try:
            common = os.path.commonpath([current, root])
        except ValueError:
            break
        if common != root:
            break
        current = os.path.dirname(current)

    candidates = []
    for directory in search_dirs:
        candidates.extend(os.path.join(directory, name) for name in YOLO_DATASET_FILENAMES)
    for directory in search_dirs:
        if not os.path.isdir(directory):
            continue
        candidates.extend(
            os.path.join(directory, name)
            for name in sorted(os.listdir(directory))
            if name.lower().endswith((".yaml", ".yml"))
            and name.lower() not in YOLO_DATASET_FILENAMES
        )
    candidates.extend(os.path.join(directory, "classes.txt") for directory in search_dirs)
    return candidates


def load_classes(classes_path: Optional[str]) -> List[str]:
    if not classes_path or not os.path.exists(classes_path):
        return []
    if classes_path.lower().endswith((".yaml", ".yml")):
        return load_yolo_yaml_classes(classes_path)
    with open(classes_path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def load_yolo_classes(
    source_path: str,
    classes_path: Optional[str] = None,
    search_root: Optional[str] = None,
) -> List[str]:
    classes, _ = resolve_yolo_class_mapping(source_path, classes_path, search_root)
    return classes


def resolve_yolo_class_mapping(
    source_path: str,
    classes_path: Optional[str] = None,
    search_root: Optional[str] = None,
) -> Tuple[List[str], Optional[str]]:
    for candidate in _candidate_yolo_class_paths(source_path, classes_path, search_root):
        class_list = load_classes(candidate)
        if class_list:
            return class_list, os.path.abspath(candidate)
    return [], None


def _append_unique(labels: List[str], values: Iterable[str]) -> None:
    seen = set(labels)
    for value in values:
        label = str(value).strip()
        if not label or label in seen:
            continue
        labels.append(label)
        seen.add(label)


def _labels_from_result(result: DetectionResult) -> List[str]:
    labels: List[str] = []
    _append_unique(labels, (item.label for item in result.boxes))
    _append_unique(labels, (item.label for item in result.segments))
    _append_unique(labels, (item.label for item in result.classifications))
    _append_unique(labels, (item.label for item in result.poses))
    _append_unique(labels, (item.label for item in result.tracks))
    return labels


def _classes_for_source(source: LabelSource, classes_path: Optional[str]) -> List[str]:
    if source.format == "yolo":
        return load_yolo_classes(source.path, classes_path, source.search_root)
    if source.format == "coco":
        with open(source.path, "r", encoding="utf-8") as f:
            data = json.load(f)
        categories = sorted(data.get("categories", []), key=lambda item: item.get("id", 0))
        return [item.get("name", str(item.get("id"))) for item in categories]
    if source.format == "pascal_voc":
        labels: List[str] = []
        files = [source.path]
        if os.path.isdir(source.path):
            files = [
                os.path.join(source.path, name)
                for name in sorted(os.listdir(source.path))
                if name.lower().endswith(".xml")
            ]
        for path in files:
            root = ET.parse(path).getroot()
            _append_unique(labels, (obj.findtext("name") or "object" for obj in root.findall("object")))
        return labels
    return []


def _source_metadata(source: LabelSource, classes_path: Optional[str]) -> Dict[str, Any]:
    metadata: Dict[str, Any] = {}
    if source.format == "yolo":
        class_list, mapping_path = resolve_yolo_class_mapping(source.path, classes_path, source.search_root)
        metadata["class_mapping"] = {
            "status": "found" if class_list else "missing",
            "path": mapping_path,
            "classes": len(class_list),
            "searched": [
                os.path.abspath(path)
                for path in _candidate_yolo_class_paths(source.path, classes_path, source.search_root)
            ],
        }
    return metadata


def label_for_class_id(class_id: int, class_list: List[str]) -> str:
    if 0 <= class_id < len(class_list):
        return class_list[class_id]
    return str(class_id)


def normalize_pixel_box(xmin, ymin, xmax, ymax, width: float, height: float) -> dict:
    if width <= 0 or height <= 0:
        raise ValueError("Image width and height must be positive for pixel coordinate conversion")
    return {
        "xmin": normalize_coordinate(float(xmin) / width),
        "ymin": normalize_coordinate(float(ymin) / height),
        "xmax": normalize_coordinate(float(xmax) / width),
        "ymax": normalize_coordinate(float(ymax) / height),
    }


def parse_yolo_file(path: str, image_name: str, class_list: List[str]) -> Tuple[str, DetectionResult]:
    result = DetectionResult(task_type="object_detection")
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            class_id = int(float(parts[0]))
            x_center, y_center, width, height = map(float, parts[1:])
            result.boxes.append(
                BoundingBox(
                    label=label_for_class_id(class_id, class_list),
                    xmin=normalize_coordinate(x_center - width / 2),
                    ymin=normalize_coordinate(y_center - height / 2),
                    xmax=normalize_coordinate(x_center + width / 2),
                    ymax=normalize_coordinate(y_center + height / 2),
                    confidence=1.0,
                )
            )
    return image_name, result


def import_yolo(
    input_path: str,
    image_dir: str,
    classes_path: Optional[str] = None,
    search_root: Optional[str] = None,
) -> List[Tuple[str, DetectionResult]]:
    class_list = load_yolo_classes(input_path, classes_path, search_root)
    files = []
    if os.path.isdir(input_path):
        files = [
            os.path.join(input_path, name)
            for name in os.listdir(input_path)
            if name.endswith(".txt") and name != "classes.txt"
        ]
    else:
        files = [input_path]

    records = []
    for path in sorted(files):
        image_name = os.path.splitext(os.path.basename(path))[0] + ".jpg"
        records.append(parse_yolo_file(path, image_name, class_list))
    return records


def import_pascal_voc(input_path: str) -> List[Tuple[str, DetectionResult]]:
    files = []
    if os.path.isdir(input_path):
        files = [os.path.join(input_path, name) for name in os.listdir(input_path) if name.lower().endswith(".xml")]
    else:
        files = [input_path]

    records = []
    for path in sorted(files):
        root = ET.parse(path).getroot()
        image_name = root.findtext("filename") or os.path.splitext(os.path.basename(path))[0] + ".jpg"
        width = float(root.findtext("size/width") or 1)
        height = float(root.findtext("size/height") or 1)
        result = DetectionResult(task_type="object_detection")
        for obj in root.findall("object"):
            label = obj.findtext("name") or "object"
            box = obj.find("bndbox")
            if box is None:
                continue
            coords = normalize_pixel_box(
                box.findtext("xmin") or 0,
                box.findtext("ymin") or 0,
                box.findtext("xmax") or width,
                box.findtext("ymax") or height,
                width,
                height,
            )
            result.boxes.append(BoundingBox(label=label, confidence=1.0, **coords))
        records.append((image_name, result))
    return records


def import_coco(input_path: str) -> List[Tuple[str, DetectionResult]]:
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    images = {item["id"]: item for item in data.get("images", [])}
    categories = {item["id"]: item.get("name", str(item["id"])) for item in data.get("categories", [])}
    grouped: Dict[int, DetectionResult] = {}
    for annotation in data.get("annotations", []):
        image_id = annotation.get("image_id")
        image = images.get(image_id)
        if not image:
            continue
        result = grouped.setdefault(image_id, DetectionResult(task_type="object_detection"))
        width = float(image.get("width") or 1)
        height = float(image.get("height") or 1)
        label = categories.get(annotation.get("category_id"), str(annotation.get("category_id")))
        bbox = annotation.get("bbox") or []
        if len(bbox) == 4:
            x, y, w, h = bbox
            coords = normalize_pixel_box(x, y, x + w, y + h, width, height)
            result.boxes.append(BoundingBox(label=label, confidence=normalize_confidence(annotation.get("score", 1.0)), **coords))
        for segmentation in annotation.get("segmentation", []) or []:
            if not isinstance(segmentation, list) or len(segmentation) < 6:
                continue
            points = [
                Point(x=normalize_coordinate(segmentation[idx] / width), y=normalize_coordinate(segmentation[idx + 1] / height))
                for idx in range(0, len(segmentation) - 1, 2)
            ]
            result.segments.append(PolygonSegment(label=label, polygon=points, confidence=normalize_confidence(annotation.get("score", 1.0))))
            result.task_type = "segmentation"

    return [(images[image_id].get("file_name", f"{image_id}.jpg"), result) for image_id, result in grouped.items()]


def import_vision_json(input_path: str) -> List[Tuple[str, DetectionResult]]:
    files = []
    if os.path.isdir(input_path):
        files = [os.path.join(input_path, name) for name in os.listdir(input_path) if name.lower().endswith(".jsonl")]
    else:
        files = [input_path]

    records = []
    for path in files:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                data = json.loads(line)
                result = DetectionResult(task_type=data.get("task_type", "object_detection"))
                for item in data.get("boxes", []):
                    norm = item.get("normalized", item)
                    result.boxes.append(BoundingBox(label=item.get("label", "object"), confidence=normalize_confidence(item.get("confidence", 1.0)), **norm))
                for item in data.get("segments", []):
                    points = [Point(x=normalize_coordinate(point.get("x", 0.0)), y=normalize_coordinate(point.get("y", 0.0))) for point in item.get("polygon", [])]
                    if len(points) >= 3:
                        result.segments.append(PolygonSegment(label=item.get("label", "object"), polygon=points, confidence=normalize_confidence(item.get("confidence", 1.0))))
                records.append((data.get("image_name", "image.jpg"), result))
    return records


def import_csv(input_path: str) -> List[Tuple[str, DetectionResult]]:
    grouped: Dict[str, DetectionResult] = {}
    with open(input_path, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            image_name = row.get("image") or row.get("image_name") or row.get("file_name") or row.get("filename")
            if not image_name:
                continue
            label = row.get("label") or row.get("class") or row.get("class_name") or row.get("category") or "object"
            result = grouped.setdefault(image_name, DetectionResult(task_type="object_detection"))
            if all(key in row for key in ["xmin", "ymin", "xmax", "ymax"]):
                result.boxes.append(
                    BoundingBox(
                        label=label,
                        xmin=normalize_coordinate(row["xmin"]),
                        ymin=normalize_coordinate(row["ymin"]),
                        xmax=normalize_coordinate(row["xmax"]),
                        ymax=normalize_coordinate(row["ymax"]),
                        confidence=normalize_confidence(row.get("confidence", 1.0)),
                    )
                )
    return list(grouped.items())


def iter_dicts(value):
    if isinstance(value, dict):
        yield value
        for child in value.values():
            yield from iter_dicts(child)
    elif isinstance(value, list):
        for child in value:
            yield from iter_dicts(child)


def import_generic_json(input_path: str) -> List[Tuple[str, DetectionResult]]:
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    grouped: Dict[str, DetectionResult] = {}
    for item in iter_dicts(data):
        if not all(key in item for key in ["xmin", "ymin", "xmax", "ymax"]):
            continue
        image_name = (
            item.get("image")
            or item.get("image_name")
            or item.get("file_name")
            or item.get("filename")
            or os.path.splitext(os.path.basename(input_path))[0] + ".jpg"
        )
        label = item.get("label") or item.get("class") or item.get("class_name") or item.get("category") or "object"
        result = grouped.setdefault(image_name, DetectionResult(task_type="object_detection"))
        result.boxes.append(
            BoundingBox(
                label=label,
                xmin=normalize_coordinate(item["xmin"]),
                ymin=normalize_coordinate(item["ymin"]),
                xmax=normalize_coordinate(item["xmax"]),
                ymax=normalize_coordinate(item["ymax"]),
                confidence=normalize_confidence(item.get("confidence", 1.0)),
            )
        )
    return list(grouped.items())


def _import_with_format(
    input_path: str,
    image_dir: str,
    source_format: str,
    classes_path: Optional[str] = None,
    search_root: Optional[str] = None,
) -> List[Tuple[str, DetectionResult]]:
    fmt = source_format
    if fmt == "yolo":
        return import_yolo(input_path, image_dir, classes_path=classes_path, search_root=search_root)
    if fmt == "pascal_voc":
        return import_pascal_voc(input_path)
    if fmt == "coco":
        return import_coco(input_path)
    if fmt == "vision_json":
        return import_vision_json(input_path)
    if fmt == "csv":
        return import_csv(input_path)
    if fmt == "generic_json":
        return import_generic_json(input_path)
    raise ValueError(f"Unsupported source format: {fmt}")


def _looks_like_yolo(path: str, image_dir: str) -> bool:
    with open(path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]
    if not lines:
        image_name = os.path.splitext(os.path.basename(path))[0] + ".jpg"
        return os.path.exists(find_image_path(image_dir, image_name))
    for line in lines:
        parts = line.split()
        if len(parts) != 5:
            return False
        try:
            float(parts[0])
            coordinates = [float(value) for value in parts[1:]]
        except ValueError:
            return False
        if not all(0.0 <= value <= 1.0 for value in coordinates):
            return False
    return True


def _detect_label_file(path: str, image_dir: str) -> Tuple[Optional[str], Optional[str]]:
    name = os.path.basename(path).lower()
    ext = os.path.splitext(name)[1]
    if name == "classes.txt":
        return None, None
    try:
        if ext == ".xml":
            root = ET.parse(path).getroot()
            if root.tag.lower().endswith("annotation"):
                return "pascal_voc", None
            return None, "unrecognized_xml_schema"
        if ext == ".txt":
            return ("yolo", None) if _looks_like_yolo(path, image_dir) else (None, "unrecognized_txt_schema")
        if ext == ".jsonl":
            with open(path, "r", encoding="utf-8") as f:
                first = next((line for line in f if line.strip()), None)
            if first is None:
                return "vision_json", None
            data = json.loads(first)
            if isinstance(data, dict) and ("image_name" in data or "task_type" in data):
                return "vision_json", None
            return None, "unrecognized_jsonl_schema"
        if ext == ".json":
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict) and {"images", "annotations", "categories"}.issubset(data):
                return "coco", None
            return None, "unrecognized_json_schema"
        if ext == ".csv":
            with open(path, "r", encoding="utf-8-sig", newline="") as f:
                fieldnames = set(csv.DictReader(f).fieldnames or [])
            image_fields = {"image", "image_name", "file_name", "filename"}
            box_fields = {"xmin", "ymin", "xmax", "ymax"}
            if fieldnames & image_fields and box_fields.issubset(fieldnames):
                return "csv", None
            return None, "unrecognized_csv_schema"
    except (OSError, ValueError, json.JSONDecodeError, ET.ParseError) as exc:
        return None, f"schema_read_failed:{exc}"
    return None, None


def discover_label_sources(input_path: str, image_dir: str) -> Tuple[List[LabelSource], Dict[str, Any]]:
    if not os.path.isdir(input_path):
        fmt = infer_label_format(input_path)
        return [LabelSource(
            path=input_path,
            format=fmt,
            search_root=os.path.dirname(os.path.abspath(input_path)),
        )], {
            "files_scanned": 1,
            "skipped_files": [],
        }

    sources = []
    skipped = []
    files_scanned = 0
    candidate_extensions = {".xml", ".txt", ".json", ".jsonl", ".csv"}
    for root, _, names in os.walk(input_path):
        for name in sorted(names):
            path = os.path.join(root, name)
            files_scanned += 1
            ext = os.path.splitext(name)[1].lower()
            if ext not in candidate_extensions or name.lower() == "classes.txt":
                continue
            fmt, reason = _detect_label_file(path, image_dir)
            if fmt:
                sources.append(LabelSource(path=path, format=fmt, search_root=os.path.abspath(input_path)))
            elif reason:
                skipped.append({"path": os.path.abspath(path), "reason": reason})
    sources.sort(key=lambda item: (item.format, item.path.lower()))
    return sources, {"files_scanned": files_scanned, "skipped_files": skipped}


def _box_values(item) -> List[float]:
    return [item.xmin, item.ymin, item.xmax, item.ymax]


def _spatial_iou(values1: List[float], values2: List[float]) -> float:
    xmin = max(values1[0], values2[0])
    ymin = max(values1[1], values2[1])
    xmax = min(values1[2], values2[2])
    ymax = min(values1[3], values2[3])
    intersection = max(0.0, xmax - xmin) * max(0.0, ymax - ymin)
    if intersection <= 0.0:
        return 0.0
    area1 = max(0.0, values1[2] - values1[0]) * max(0.0, values1[3] - values1[1])
    area2 = max(0.0, values2[2] - values2[0]) * max(0.0, values2[3] - values2[1])
    return intersection / (area1 + area2 - intersection)


def _segment_values(item: PolygonSegment) -> List[float]:
    xs = [point.x for point in item.polygon]
    ys = [point.y for point in item.polygon]
    return [min(xs), min(ys), max(xs), max(ys)]


def _prefer_new(existing, existing_format: str, new, new_format: str) -> bool:
    if new.confidence != existing.confidence:
        return new.confidence > existing.confidence
    return SOURCE_PRIORITY.get(new_format, 0) > SOURCE_PRIORITY.get(existing_format, 0)


def _is_numeric_label(label: str) -> bool:
    text = str(label).strip()
    return text.isdigit()


def _merge_spatial_items(
    target: list,
    target_sources: List[str],
    incoming: Iterable,
    incoming_format: str,
    value_getter,
    duplicate_iou: float,
    image_name: str,
    item_type: str,
    conflicts: List[Dict[str, Any]],
    label_normalizations: List[Dict[str, Any]],
) -> int:
    duplicates = 0
    for item in incoming:
        item_values = value_getter(item)
        duplicate_index = None
        normalized = False
        for index, existing in enumerate(target):
            overlap = _spatial_iou(value_getter(existing), item_values)
            if overlap < duplicate_iou:
                continue
            if existing.label == item.label:
                duplicate_index = index
                break
            existing_numeric = _is_numeric_label(existing.label)
            incoming_numeric = _is_numeric_label(item.label)
            if existing_numeric != incoming_numeric:
                duplicates += 1
                existing_format = target_sources[index]
                chosen_label = item.label if existing_numeric else existing.label
                numeric_label = existing.label if existing_numeric else item.label
                if existing_numeric:
                    target[index] = item.model_copy(deep=True)
                    target_sources[index] = incoming_format
                label_normalizations.append({
                    "image": image_name,
                    "type": item_type,
                    "numeric_label": numeric_label,
                    "canonical_label": chosen_label,
                    "iou": round(overlap, 6),
                    "existing_format": existing_format,
                    "incoming_format": incoming_format,
                })
                normalized = True
                break
            conflicts.append({
                "image": image_name,
                "type": item_type,
                "existing_label": existing.label,
                "incoming_label": item.label,
                "iou": round(overlap, 6),
                "existing_format": target_sources[index],
                "incoming_format": incoming_format,
            })
        if normalized:
            continue
        if duplicate_index is None:
            target.append(item.model_copy(deep=True))
            target_sources.append(incoming_format)
            continue
        duplicates += 1
        existing = target[duplicate_index]
        if _prefer_new(existing, target_sources[duplicate_index], item, incoming_format):
            target[duplicate_index] = item.model_copy(deep=True)
            target_sources[duplicate_index] = incoming_format
    return duplicates


def _canonical_image_key(image_name: str) -> str:
    normalized = image_name.replace("\\", "/").rstrip("/")
    return os.path.splitext(os.path.basename(normalized))[0].casefold()


def _apply_inferred_numeric_label_mapping(
    records: List[Tuple[str, DetectionResult]],
    label_normalizations: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    candidates: Dict[str, set] = {}
    for item in label_normalizations:
        numeric_label = str(item.get("numeric_label", "")).strip()
        canonical_label = str(item.get("canonical_label", "")).strip()
        if numeric_label.isdigit() and canonical_label and not canonical_label.isdigit():
            candidates.setdefault(numeric_label, set()).add(canonical_label)

    inferred = {
        numeric_label: next(iter(labels))
        for numeric_label, labels in candidates.items()
        if len(labels) == 1
    }
    if not inferred:
        return []

    propagated: List[Dict[str, Any]] = []
    fields = ("boxes", "segments", "classifications", "poses", "tracks")
    for image_name, result in records:
        for field in fields:
            for item in getattr(result, field):
                label = getattr(item, "label", None)
                if str(label) not in inferred:
                    continue
                numeric_label = str(label)
                canonical_label = inferred[numeric_label]
                item.label = canonical_label
                propagated.append({
                    "image": image_name,
                    "type": f"{field}_global_numeric_label",
                    "numeric_label": numeric_label,
                    "canonical_label": canonical_label,
                    "reason": "inferred_from_matching_named_label",
                })
    return propagated


def merge_label_records(
    records: List[Tuple[str, DetectionResult, LabelSource]],
    duplicate_iou: float = 0.85,
) -> Tuple[List[Tuple[str, DetectionResult]], Dict[str, Any]]:
    grouped: Dict[str, Dict[str, Any]] = {}
    duplicates_removed = 0
    conflicts: List[Dict[str, Any]] = []
    label_normalizations: List[Dict[str, Any]] = []
    identity_collisions = []

    for image_name, result, source in records:
        key = _canonical_image_key(image_name)
        if key not in grouped:
            grouped[key] = {
                "image_name": image_name,
                "result": DetectionResult(task_type=result.task_type),
                "formats": [],
                "paths": [],
                "box_sources": [],
                "segment_sources": [],
                "aliases": {image_name},
            }
        entry = grouped[key]
        if image_name not in entry["aliases"]:
            entry["aliases"].add(image_name)
            identity_collisions.append({
                "image_key": key,
                "names": sorted(entry["aliases"]),
            })
        merged = entry["result"]
        entry["formats"].append(source.format)
        entry["paths"].append(os.path.abspath(source.path))
        duplicates_removed += _merge_spatial_items(
            merged.boxes,
            entry["box_sources"],
            result.boxes,
            source.format,
            _box_values,
            duplicate_iou,
            image_name,
            "box_label_conflict",
            conflicts,
            label_normalizations,
        )
        duplicates_removed += _merge_spatial_items(
            merged.segments,
            entry["segment_sources"],
            result.segments,
            source.format,
            _segment_values,
            duplicate_iou,
            image_name,
            "segment_label_conflict",
            conflicts,
            label_normalizations,
        )

        existing_classes = {item.label: index for index, item in enumerate(merged.classifications)}
        for item in result.classifications:
            index = existing_classes.get(item.label)
            if index is None:
                merged.classifications.append(item.model_copy(deep=True))
                existing_classes[item.label] = len(merged.classifications) - 1
            else:
                duplicates_removed += 1
                if item.confidence > merged.classifications[index].confidence:
                    merged.classifications[index] = item.model_copy(deep=True)

        for field in ("poses", "texts", "tracks"):
            target_items = getattr(merged, field)
            signatures = {json.dumps(item.model_dump(), sort_keys=True) for item in target_items}
            for item in getattr(result, field):
                signature = json.dumps(item.model_dump(), sort_keys=True)
                if signature in signatures:
                    duplicates_removed += 1
                    continue
                target_items.append(item.model_copy(deep=True))
                signatures.add(signature)

        if merged.task_type != result.task_type:
            merged.task_type = "all"

    output = []
    for entry in grouped.values():
        result = entry["result"]
        result.plugin_metadata = dict(result.plugin_metadata)
        result.plugin_metadata["conversion_sources"] = {
            "formats": sorted(set(entry["formats"])),
            "paths": sorted(set(entry["paths"])),
        }
        output.append((entry["image_name"], result))
    output.sort(key=lambda item: item[0].casefold())
    global_normalizations = _apply_inferred_numeric_label_mapping(output, label_normalizations)
    label_normalizations.extend(global_normalizations)
    return output, {
        "duplicate_iou": duplicate_iou,
        "duplicates_removed": duplicates_removed,
        "conflicts": conflicts,
        "label_normalizations": label_normalizations,
        "global_label_normalizations": global_normalizations,
        "image_identity_collisions": identity_collisions,
    }


def import_labels_with_report(
    input_path: str,
    image_dir: str,
    source_format: str = "auto",
    classes_path: Optional[str] = None,
    duplicate_iou: float = 0.85,
) -> LabelImportBatch:
    discovery = {"files_scanned": 1, "skipped_files": []}
    if source_format == "auto" and os.path.isdir(input_path):
        sources, discovery = discover_label_sources(input_path, image_dir)
    else:
        fmt = infer_label_format(input_path) if source_format == "auto" else source_format
        search_root = input_path if os.path.isdir(input_path) else os.path.dirname(os.path.abspath(input_path))
        sources = [LabelSource(path=input_path, format=fmt, search_root=os.path.abspath(search_root))]

    imported = []
    processed_sources = []
    failed_sources = []
    class_sources = []
    for source in sources:
        try:
            source_records = _import_with_format(
                source.path,
                image_dir,
                source.format,
                classes_path=classes_path,
                search_root=source.search_root,
            )
            source_classes = _classes_for_source(source, classes_path)
            metadata = _source_metadata(source, classes_path)
            has_unmapped_yolo = (
                source.format == "yolo"
                and metadata.get("class_mapping", {}).get("status") == "missing"
            )
            if not source_classes and not has_unmapped_yolo:
                for _, result in source_records:
                    _append_unique(source_classes, _labels_from_result(result))
            processed_sources.append({
                "path": os.path.abspath(source.path),
                "format": source.format,
                "records": len(source_records),
                "classes": source_classes,
                **metadata,
            })
            class_sources.append((source.format, os.path.abspath(source.path), source_classes))
            imported.extend((image_name, result, source) for image_name, result in source_records)
        except Exception as exc:
            failed_sources.append({
                "path": os.path.abspath(source.path),
                "format": source.format,
                "error": str(exc),
            })

    merged, merge_report = merge_label_records(imported, duplicate_iou=duplicate_iou)
    class_list: List[str] = []
    for _, _, labels in sorted(
        class_sources,
        key=lambda item: (-CLASS_LIST_PRIORITY.get(item[0], 0), item[1].lower()),
    ):
        _append_unique(class_list, labels)
    for _, result in merged:
        _append_unique(class_list, _labels_from_result(result))
    format_counts = Counter(item["format"] for item in processed_sources)
    report = {
        "mode": "mixed_auto" if source_format == "auto" and os.path.isdir(input_path) else "single_format",
        "files_scanned": discovery["files_scanned"],
        "sources_discovered": len(sources),
        "sources_processed": len(processed_sources),
        "sources_failed": len(failed_sources),
        "formats": dict(sorted(format_counts.items())),
        "records_before_merge": len(imported),
        "records_after_merge": len(merged),
        "class_list": class_list,
        "processed_files": processed_sources,
        "failed_files": failed_sources,
        "skipped_files": discovery["skipped_files"],
        "merge": merge_report,
    }
    return LabelImportBatch(records=merged, report=report)


def import_labels(
    input_path: str,
    image_dir: str,
    source_format: str = "auto",
    classes_path: Optional[str] = None,
) -> List[Tuple[str, DetectionResult]]:
    return import_labels_with_report(
        input_path,
        image_dir,
        source_format=source_format,
        classes_path=classes_path,
    ).records
