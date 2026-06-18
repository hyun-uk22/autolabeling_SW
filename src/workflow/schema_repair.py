import json
import os
from copy import deepcopy
from typing import List

from ..core.models import DetectionResult


def schema_candidates(input_path: str, current_format: str = "auto") -> List[str]:
    candidates = []
    if current_format != "auto":
        candidates.append(current_format)
    if os.path.isdir(input_path):
        names = os.listdir(input_path)
        if any(name.lower().endswith(".xml") for name in names):
            candidates.append("pascal_voc")
        if any(name.lower().endswith(".txt") for name in names):
            candidates.append("yolo")
        if any(name.lower().endswith(".jsonl") for name in names):
            candidates.append("vision_json")
    else:
        ext = os.path.splitext(input_path)[1].lower()
        mapping = {".xml": "pascal_voc", ".txt": "yolo", ".jsonl": "vision_json", ".csv": "csv"}
        if ext in mapping:
            candidates.append(mapping[ext])
        if ext == ".json":
            try:
                with open(input_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, dict) and {"images", "annotations", "categories"}.issubset(data):
                    candidates.extend(["coco", "generic_json"])
                else:
                    candidates.extend(["generic_json", "coco"])
            except Exception:
                candidates.extend(["generic_json", "coco"])
    if current_format == "auto":
        candidates.insert(0, "auto")
    return list(dict.fromkeys(candidates))


def repair_result(result: DetectionResult) -> DetectionResult:
    repaired = deepcopy(result)
    repaired.classifications = [item for item in repaired.classifications if item.label]
    repaired.boxes = [
        item for item in repaired.boxes
        if item.label
        and 0.0 <= item.xmin < item.xmax <= 1.0
        and 0.0 <= item.ymin < item.ymax <= 1.0
    ]
    repaired.segments = [
        item for item in repaired.segments
        if item.label
        and len(item.polygon) >= 3
        and all(0.0 <= point.x <= 1.0 and 0.0 <= point.y <= 1.0 for point in item.polygon)
    ]
    repaired.poses = [
        item for item in repaired.poses
        if item.keypoints
        and all(point.name and 0.0 <= point.x <= 1.0 and 0.0 <= point.y <= 1.0 for point in item.keypoints)
    ]
    repaired.texts = [
        item for item in repaired.texts
        if item.text
        and 0.0 <= item.xmin < item.xmax <= 1.0
        and 0.0 <= item.ymin < item.ymax <= 1.0
    ]
    repaired.tracks = [
        item for item in repaired.tracks
        if item.track_id and item.label
        and 0.0 <= item.xmin < item.xmax <= 1.0
        and 0.0 <= item.ymin < item.ymax <= 1.0
    ]
    return repaired
