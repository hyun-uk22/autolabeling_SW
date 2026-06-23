import json
import os
import xml.etree.ElementTree as ET
from typing import Dict, List

from .conversion_preflight import build_conversion_preflight
from .issue_reporter import build_user_action_report


class ArtifactAuditor:
    def audit_record(self, paths: Dict[str, str]) -> List[str]:
        issues: List[str] = []
        for fmt, path in paths.items():
            if not path:
                issues.append(f"missing_output_path:{fmt}")
                continue
            if fmt in {"coco", "vision_json"}:
                continue
            issues.extend(self._audit_file(fmt, path))
        return issues

    def audit_artifacts(self, artifacts: Dict[str, str]) -> List[str]:
        issues: List[str] = []
        for fmt, path in artifacts.items():
            issues.extend(self._audit_file(fmt, path))
        return issues

    def _audit_file(self, fmt: str, path: str) -> List[str]:
        if not os.path.exists(path):
            return [f"missing_output_file:{path}"]
        if os.path.getsize(path) == 0:
            return [f"empty_output_file:{path}"]
        if fmt == "yolo":
            return self._audit_yolo(path)
        if fmt == "pascal_voc":
            return self._audit_pascal_voc(path)
        if fmt == "coco":
            return self._audit_coco(path)
        if fmt == "vision_json":
            return self._audit_vision_json(path)
        return []

    @staticmethod
    def _audit_yolo(path: str) -> List[str]:
        issues: List[str] = []
        rows = [line.strip() for line in open(path, encoding="utf-8") if line.strip()]
        if not rows:
            return [f"no_label_rows:{path}"]
        for index, row in enumerate(rows, start=1):
            parts = row.split()
            if len(parts) != 5:
                issues.append(f"invalid_row_shape:{path}:{index}")
                continue
            try:
                class_id = int(float(parts[0]))
                values = [float(value) for value in parts[1:]]
            except ValueError:
                issues.append(f"non_numeric_row:{path}:{index}")
                continue
            if class_id < 0:
                issues.append(f"negative_class_id:{path}:{index}")
            x_center, y_center, width, height = values
            if any(value < 0 or value > 1 for value in values):
                issues.append(f"coordinate_out_of_range:{path}:{index}")
            if width <= 0 or height <= 0:
                issues.append(f"invalid_box_size:{path}:{index}")
            if x_center - width / 2 < 0 or x_center + width / 2 > 1:
                issues.append(f"coordinate_out_of_range:{path}:{index}")
            if y_center - height / 2 < 0 or y_center + height / 2 > 1:
                issues.append(f"coordinate_out_of_range:{path}:{index}")
        return issues

    @staticmethod
    def _audit_pascal_voc(path: str) -> List[str]:
        try:
            root = ET.parse(path).getroot()
        except ET.ParseError:
            return [f"output_parse_failed:{path}"]
        objects = root.findall("object")
        if not objects:
            return [f"no_objects:{path}"]
        issues: List[str] = []
        for index, item in enumerate(objects, start=1):
            if item.findtext("name") is None or item.find("bndbox") is None:
                issues.append(f"invalid_object:{path}:{index}")
                continue
            box = item.find("bndbox")
            try:
                xmin = float(box.findtext("xmin", ""))
                ymin = float(box.findtext("ymin", ""))
                xmax = float(box.findtext("xmax", ""))
                ymax = float(box.findtext("ymax", ""))
            except ValueError:
                issues.append(f"non_numeric_box:{path}:{index}")
                continue
            if xmin >= xmax or ymin >= ymax:
                issues.append(f"invalid_box_size:{path}:{index}")
        return issues

    @staticmethod
    def _audit_coco(path: str) -> List[str]:
        try:
            data = json.load(open(path, encoding="utf-8"))
        except (OSError, ValueError):
            return [f"output_parse_failed:{path}"]
        issues: List[str] = []
        if not data.get("images"):
            issues.append(f"no_images:{path}")
        if not data.get("annotations"):
            issues.append(f"no_annotations:{path}")
        if not data.get("categories"):
            issues.append(f"no_categories:{path}")
        for index, annotation in enumerate(data.get("annotations", []), start=1):
            bbox = annotation.get("bbox")
            if not isinstance(bbox, list) or len(bbox) != 4:
                issues.append(f"invalid_bbox:{path}:{index}")
                continue
            try:
                _, _, width, height = [float(value) for value in bbox]
            except (TypeError, ValueError):
                issues.append(f"non_numeric_bbox:{path}:{index}")
                continue
            if width <= 0 or height <= 0:
                issues.append(f"invalid_bbox_size:{path}:{index}")
            if "category_id" not in annotation:
                issues.append(f"missing_category_id:{path}:{index}")
        return issues

    @staticmethod
    def _audit_vision_json(path: str) -> List[str]:
        try:
            rows = [json.loads(line) for line in open(path, encoding="utf-8") if line.strip()]
        except (OSError, ValueError):
            return [f"output_parse_failed:{path}"]
        label_keys = ("boxes", "segments", "poses", "texts", "tracks", "classifications")
        if not any(any(record.get(key) for key in label_keys) for record in rows):
            return [f"no_label_records:{path}"]
        return []


def build_generation_performance(
    image_count: int,
    elapsed_sec: float,
    low_api_attempts: int,
    high_api_attempts: int,
    escalation_count: int,
) -> Dict[str, float | int | str]:
    avg_elapsed = elapsed_sec / image_count if image_count else 0.0
    total_attempts = low_api_attempts + high_api_attempts
    escalation_rate = escalation_count / image_count if image_count else 0.0
    return {
        "image_count": image_count,
        "elapsed_sec": elapsed_sec,
        "avg_elapsed_sec": avg_elapsed,
        "low_api_attempts": low_api_attempts,
        "high_api_attempts": high_api_attempts,
        "total_api_attempts": total_attempts,
        "escalation_count": escalation_count,
        "escalation_rate": escalation_rate,
        "estimation_notice": "API 시도 횟수 기반 추정값입니다.",
    }


__all__ = [
    "ArtifactAuditor",
    "build_conversion_preflight",
    "build_generation_performance",
    "build_user_action_report",
]
