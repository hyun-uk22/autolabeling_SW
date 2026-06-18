import csv
import json
import os
import xml.etree.ElementTree as ET
from typing import Dict, Iterable, List, Optional, Tuple

from ..core.llm_client import normalize_confidence, normalize_coordinate
from ..core.models import BoundingBox, DetectionResult, Point, PolygonSegment


IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".webp", ".bmp")


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


def import_yolo(input_path: str, image_dir: str, classes_path: Optional[str] = None) -> List[Tuple[str, DetectionResult]]:
    class_list = load_classes(classes_path or os.path.join(input_path, "classes.txt"))
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


def import_labels(input_path: str, image_dir: str, source_format: str = "auto", classes_path: Optional[str] = None) -> List[Tuple[str, DetectionResult]]:
    fmt = infer_label_format(input_path) if source_format == "auto" else source_format
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
