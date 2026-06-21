import json
import os
import xml.etree.ElementTree as ET
from typing import Dict, List


AGGREGATE_FORMATS = {"coco", "vision_json", "classes"}


class ArtifactAuditor:
    def audit_record(self, paths: Dict[str, str]) -> List[str]:
        issues = []
        for fmt, path in paths.items():
            if fmt not in AGGREGATE_FORMATS:
                issues.extend(self._audit(fmt, path))
        return issues

    def audit_artifacts(self, artifacts: Dict[str, str]) -> List[str]:
        issues = []
        for fmt, path in artifacts.items():
            issues.extend(self._audit(fmt, path))
        return issues

    def _audit(self, fmt: str, path: str) -> List[str]:
        if not path:
            return [f"{fmt}:missing_output_path"]
        if not os.path.exists(path):
            return [f"{fmt}:missing_output_file:{path}"]
        if os.path.getsize(path) == 0:
            return [f"{fmt}:empty_output_file:{path}"]
        try:
            method = getattr(self, f"_audit_{fmt}", None)
            return method(path) if method else []
        except Exception as exc:
            return [f"{fmt}:output_parse_failed:{exc}"]

    def _audit_yolo(self, path: str) -> List[str]:
        with open(path, "r", encoding="utf-8") as handle:
            rows = [line.split() for line in handle if line.strip()]
        if not rows:
            return [f"yolo:no_label_rows:{path}"]
        for row in rows:
            if len(row) != 5:
                return [f"yolo:invalid_row_shape:{path}"]
            try:
                class_id = int(float(row[0]))
                x_center, y_center, width, height = map(float, row[1:])
            except ValueError:
                return [f"yolo:non_numeric_row:{path}"]
            if class_id < 0:
                return [f"yolo:negative_class_id:{path}"]
            if not all(0.0 <= value <= 1.0 for value in (x_center, y_center, width, height)):
                return [f"yolo:coordinate_out_of_range:{path}"]
            if width <= 0.0 or height <= 0.0:
                return [f"yolo:invalid_box_size:{path}"]
        return []

    def _audit_pascal_voc(self, path: str) -> List[str]:
        root = ET.parse(path).getroot()
        objects = root.findall("object")
        if not objects:
            return [f"pascal_voc:no_objects:{path}"]
        for item in objects:
            box = item.find("bndbox")
            if not item.findtext("name") or box is None:
                return [f"pascal_voc:invalid_object:{path}"]
            try:
                xmin, ymin, xmax, ymax = [
                    float(box.findtext(name) or "")
                    for name in ("xmin", "ymin", "xmax", "ymax")
                ]
            except ValueError:
                return [f"pascal_voc:non_numeric_box:{path}"]
            if xmin >= xmax or ymin >= ymax:
                return [f"pascal_voc:invalid_box_order:{path}"]
        return []

    def _audit_coco(self, path: str) -> List[str]:
        with open(path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
        if not data.get("images"):
            return [f"coco:no_images:{path}"]
        if not data.get("annotations"):
            return [f"coco:no_annotations:{path}"]
        if not data.get("categories"):
            return [f"coco:no_categories:{path}"]
        for annotation in data["annotations"]:
            bbox = annotation.get("bbox")
            if not isinstance(bbox, list) or len(bbox) != 4:
                return [f"coco:invalid_bbox:{path}"]
            try:
                _, _, width, height = map(float, bbox)
            except (TypeError, ValueError):
                return [f"coco:non_numeric_bbox:{path}"]
            if width <= 0.0 or height <= 0.0:
                return [f"coco:invalid_bbox_size:{path}"]
            if annotation.get("category_id") is None:
                return [f"coco:missing_category_id:{path}"]
        return []

    def _audit_vision_json(self, path: str) -> List[str]:
        valid = 0
        with open(path, "r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                record = json.loads(line)
                valid += int(any(record.get(key) for key in (
                    "classifications", "boxes", "segments", "poses", "texts", "tracks",
                )))
        return [] if valid else [f"vision_json:no_label_records:{path}"]

    def _audit_classes(self, path: str) -> List[str]:
        with open(path, "r", encoding="utf-8") as handle:
            labels = [line.strip() for line in handle if line.strip()]
        return [] if labels else [f"classes:no_labels:{path}"]
