import csv
import json
import os
import re
import xml.etree.ElementTree as ET
from collections import Counter
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

from ..core.llm_client import normalize_confidence, normalize_coordinate
from ..core.models import BoundingBox, DetectionResult, Point, PolygonSegment


IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".webp", ".bmp")
SOURCE_PRIORITY = {
    "coco": 60,
    "pascal_voc": 50,
    "yolo": 40,
    "vision_json": 30,
    "csv": 20,
    "generic_json": 10,
}


@dataclass(frozen=True)
class LabelSource:
    path: str
    format: str


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


def load_classes(classes_path: Optional[str]) -> List[str]:
    if not classes_path or not os.path.exists(classes_path):
        return []
    with open(classes_path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


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
    with open(path, "r", encoding="utf-8-sig") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            class_id = int(float(parts[0]))
            x_center, y_center, width, height = map(float, parts[1:])
            result.boxes.append(
                BoundingBox(
                    label=label_for_class_id(class_id, class_list),
                    xmin=x_center - width / 2,
                    ymin=y_center - height / 2,
                    xmax=x_center + width / 2,
                    ymax=y_center + height / 2,
                    confidence=1.0,
                )
            )
    return image_name, result


def import_yolo(input_path: str, image_dir: str, classes_path: Optional[str] = None) -> List[Tuple[str, DetectionResult]]:
    default_classes = os.path.join(input_path if os.path.isdir(input_path) else os.path.dirname(input_path), "classes.txt")
    class_list = load_classes(classes_path or default_classes)
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
        with open(path, "r", encoding="utf-8-sig") as f:
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
) -> List[Tuple[str, DetectionResult]]:
    fmt = source_format
    if fmt == "yolo":
        return import_yolo(input_path, image_dir, classes_path=classes_path)
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
    with open(path, "r", encoding="utf-8-sig") as f:
        lines = [line.strip() for line in f if line.strip()]
    if not lines:
        return True
    for line in lines:
        parts = line.split()
        if len(parts) != 5:
            return False
        try:
            float(parts[0])
            [float(value) for value in parts[1:]]
        except ValueError:
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
            with open(path, "r", encoding="utf-8-sig") as f:
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
        if ext in {".yaml", ".yml"}:
            return None, _audit_yaml_config(path)
    except (OSError, ValueError, json.JSONDecodeError, ET.ParseError) as exc:
        return None, f"schema_read_failed:{exc}"
    return None, None


def _audit_yaml_config(path: str) -> str:
    text = open(path, "r", encoding="utf-8-sig").read()
    if not re.search(r"(?m)^\s*names\s*:", text):
        return "unrecognized_yaml_schema"

    invalid_keys = []
    for key in re.findall(r"(?m)^\s{2,}([^:\s]+)\s*:", text):
        try:
            if int(key) < 0:
                invalid_keys.append(key)
        except ValueError:
            invalid_keys.append(key)
    if invalid_keys:
        return f"yaml_invalid_class_id:{','.join(invalid_keys[:5])}"

    base_path = None
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("path:"):
            base_path = stripped.split(":", 1)[1].strip().strip("'\"")
            break
    if base_path and not os.path.exists(base_path):
        return f"yaml_path_missing:{base_path}"
    return "yaml_config_not_label_source"


def discover_label_sources(input_path: str, image_dir: str) -> Tuple[List[LabelSource], Dict[str, Any]]:
    if not os.path.isdir(input_path):
        fmt = infer_label_format(input_path)
        return [LabelSource(path=input_path, format=fmt)], {
            "files_scanned": 1,
            "skipped_files": [],
        }

    sources = []
    skipped = []
    files_scanned = 0
    candidate_extensions = {".xml", ".txt", ".json", ".jsonl", ".csv", ".yaml", ".yml"}
    for root, _, names in os.walk(input_path):
        for name in sorted(names):
            path = os.path.join(root, name)
            files_scanned += 1
            ext = os.path.splitext(name)[1].lower()
            if ext not in candidate_extensions or name.lower() == "classes.txt":
                continue
            fmt, reason = _detect_label_file(path, image_dir)
            if fmt:
                sources.append(LabelSource(path=path, format=fmt))
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
) -> int:
    duplicates = 0
    for item in incoming:
        item_values = value_getter(item)
        duplicate_index = None
        for index, existing in enumerate(target):
            overlap = _spatial_iou(value_getter(existing), item_values)
            if overlap < duplicate_iou:
                continue
            if existing.label == item.label:
                duplicate_index = index
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


def merge_label_records(
    records: List[Tuple[str, DetectionResult, LabelSource]],
    duplicate_iou: float = 0.85,
) -> Tuple[List[Tuple[str, DetectionResult]], Dict[str, Any]]:
    grouped: Dict[str, Dict[str, Any]] = {}
    duplicates_removed = 0
    conflicts: List[Dict[str, Any]] = []
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
    return output, {
        "duplicate_iou": duplicate_iou,
        "duplicates_removed": duplicates_removed,
        "conflicts": conflicts,
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
        sources = [LabelSource(path=input_path, format=fmt)]

    imported = []
    processed_sources = []
    failed_sources = []
    for source in sources:
        try:
            source_records = _import_with_format(
                source.path,
                image_dir,
                source.format,
                classes_path=classes_path,
            )
            processed_sources.append({
                "path": os.path.abspath(source.path),
                "format": source.format,
                "records": len(source_records),
            })
            imported.extend((image_name, result, source) for image_name, result in source_records)
        except Exception as exc:
            failed_sources.append({
                "path": os.path.abspath(source.path),
                "format": source.format,
                "error": str(exc),
            })

    merged, merge_report = merge_label_records(imported, duplicate_iou=duplicate_iou)
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
