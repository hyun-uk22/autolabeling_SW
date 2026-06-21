import json
import os
import xml.etree.ElementTree as ET
from typing import Dict, Iterable, List

from ..core.models import DetectionResult
from ..utils.format_converter import normalize_label_formats
from ..utils.label_validator import validate_result
from ..utils.result_metrics import count_result_labels
from ..workflow.schema_repair import repair_result


AGGREGATE_FORMATS = {"coco", "vision_json", "classes"}


class ConversionQualityAgent:
    """
    Guards label format conversion against silent bad outputs.

    The converter can technically create files even when no labels were parsed.
    This agent treats empty parsed results and empty exported artifacts as
    conversion failures so they are visible in reports instead of being accepted.
    """

    def validate_input(self, result: DetectionResult, image_path: str) -> List[str]:
        issues = validate_result(result, image_path)
        if count_result_labels(result) == 0 and "empty_result" not in issues:
            issues.append("empty_result")
        return issues

    def blocking_input_issues(self, issues: Iterable[str], strict: bool = False) -> List[str]:
        blocking = []
        for issue in issues:
            if (
                issue == "empty_result"
                or issue.startswith("missing_image:")
                or issue.startswith("image_open_failed:")
                or issue == "invalid_image_size"
            ):
                blocking.append(issue)
            elif strict:
                blocking.append(issue)
        return blocking

    def repair_detection_result(self, result: DetectionResult) -> DetectionResult:
        return repair_result(result)

    def resolve_export_formats(
        self,
        result: DetectionResult,
        requested_formats: Iterable[str] | str,
        task_type: str | None = None,
    ) -> List[str]:
        requested = normalize_label_formats(requested_formats)
        selected: List[str] = []
        has_boxes = bool(result.boxes)
        has_segments = bool(result.segments)
        has_image_level = bool(result.classifications)
        has_texts = bool(result.texts)
        has_poses = bool(result.poses)
        has_tracks = bool(result.tracks)
        needs_lossless = (
            task_type in {"classification", "segmentation", "pose_estimation", "ocr", "tracking", "all"}
            or has_segments
            or has_image_level
            or has_texts
            or has_poses
            or has_tracks
        )

        for fmt in requested:
            if fmt in {"yolo", "pascal_voc"} and not has_boxes:
                continue
            if fmt == "coco" and not (has_boxes or has_segments):
                continue
            selected.append(fmt)

        if needs_lossless and "vision_json" not in selected:
            selected.append("vision_json")

        if not selected and (has_boxes or has_segments or has_image_level or has_texts or has_poses or has_tracks):
            selected.append("vision_json")

        return list(dict.fromkeys(selected))

    def summarize_recovery(self, requested: Iterable[str] | str, resolved: Iterable[str], issues: Iterable[str]) -> Dict[str, object]:
        requested_list = normalize_label_formats(requested)
        resolved_list = list(dict.fromkeys(resolved))
        return {
            "requested_formats": requested_list,
            "resolved_formats": resolved_list,
            "fallback_applied": requested_list != resolved_list,
            "issues": list(issues),
        }

    def audit_record_exports(self, paths: Dict[str, str]) -> List[str]:
        issues = []
        for fmt, path in paths.items():
            if fmt in AGGREGATE_FORMATS:
                continue
            issues.extend(self._audit_path(fmt, path))
        return issues

    def audit_final_artifacts(self, artifacts: Dict[str, str]) -> List[str]:
        issues = []
        for fmt, path in artifacts.items():
            issues.extend(self._audit_path(fmt, path))
        return issues

    def _audit_path(self, fmt: str, path: str) -> List[str]:
        if not path:
            return [f"{fmt}:missing_output_path"]
        if not os.path.exists(path):
            return [f"{fmt}:missing_output_file:{path}"]
        if os.path.getsize(path) == 0:
            return [f"{fmt}:empty_output_file:{path}"]

        try:
            if fmt == "yolo":
                return self._audit_yolo(path)
            if fmt == "pascal_voc":
                return self._audit_pascal_voc(path)
            if fmt == "coco":
                return self._audit_coco(path)
            if fmt == "vision_json":
                return self._audit_vision_json(path)
            if fmt == "classes":
                return self._audit_classes(path)
        except Exception as exc:
            return [f"{fmt}:output_parse_failed:{exc}"]
        return []

    def _audit_yolo(self, path: str) -> List[str]:
        with open(path, "r", encoding="utf-8") as f:
            rows = [line.strip().split() for line in f if line.strip()]
        if not rows:
            return [f"yolo:no_label_rows:{path}"]
        if any(len(row) != 5 for row in rows):
            return [f"yolo:invalid_row_shape:{path}"]
        for row in rows:
            try:
                class_id = int(float(row[0]))
                x_center, y_center, width, height = map(float, row[1:])
            except ValueError:
                return [f"yolo:non_numeric_row:{path}"]
            if class_id < 0:
                return [f"yolo:negative_class_id:{path}"]
            if not all(0.0 <= value <= 1.0 for value in [x_center, y_center, width, height]):
                return [f"yolo:coordinate_out_of_range:{path}"]
            if width <= 0.0 or height <= 0.0:
                return [f"yolo:invalid_box_size:{path}"]
        return []

    def _audit_pascal_voc(self, path: str) -> List[str]:
        root = ET.parse(path).getroot()
        objects = root.findall("object")
        if not objects:
            return [f"pascal_voc:no_objects:{path}"]
        for obj in objects:
            box = obj.find("bndbox")
            if not obj.findtext("name") or box is None:
                return [f"pascal_voc:invalid_object:{path}"]
            try:
                xmin = float(box.findtext("xmin") or "")
                ymin = float(box.findtext("ymin") or "")
                xmax = float(box.findtext("xmax") or "")
                ymax = float(box.findtext("ymax") or "")
            except ValueError:
                return [f"pascal_voc:non_numeric_box:{path}"]
            if xmin >= xmax or ymin >= ymax:
                return [f"pascal_voc:invalid_box_order:{path}"]
        return []

    def _audit_coco(self, path: str) -> List[str]:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
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
                _, _, width, height = [float(value) for value in bbox]
            except (TypeError, ValueError):
                return [f"coco:non_numeric_bbox:{path}"]
            if width <= 0.0 or height <= 0.0:
                return [f"coco:invalid_bbox_size:{path}"]
            if annotation.get("category_id") is None:
                return [f"coco:missing_category_id:{path}"]
        return []

    def _audit_vision_json(self, path: str) -> List[str]:
        valid_records = 0
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                record = json.loads(line)
                label_count = sum(
                    len(record.get(key, []))
                    for key in ["classifications", "boxes", "segments", "poses", "texts", "tracks"]
                )
                if label_count > 0:
                    valid_records += 1
        if valid_records == 0:
            return [f"vision_json:no_label_records:{path}"]
        return []

    def _audit_classes(self, path: str) -> List[str]:
        with open(path, "r", encoding="utf-8") as f:
            labels = [line.strip() for line in f if line.strip()]
        if not labels:
            return [f"classes:no_labels:{path}"]
        return []
