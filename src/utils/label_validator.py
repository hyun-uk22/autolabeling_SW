import os
from typing import Dict, List, Optional

from PIL import Image

from ..core.models import DetectionResult


def validate_result(result: DetectionResult, image_path: Optional[str] = None) -> List[str]:
    issues = []
    label_count = (
        len(result.classifications)
        + len(result.boxes)
        + len(result.segments)
        + len(result.poses)
        + len(result.texts)
        + len(result.tracks)
    )
    if label_count == 0:
        issues.append("empty_result")

    if image_path and not os.path.exists(image_path):
        issues.append(f"missing_image:{image_path}")

    width = height = None
    if image_path and os.path.exists(image_path):
        try:
            with Image.open(image_path) as img:
                width, height = img.size
            if width <= 0 or height <= 0:
                issues.append("invalid_image_size")
        except Exception as exc:
            issues.append(f"image_open_failed:{exc}")

    def valid_coord(value):
        return 0.0 <= value <= 1.0

    def validate_box(prefix, item):
        if not getattr(item, "label", getattr(item, "text", None)):
            issues.append(f"{prefix}:missing_label")
        if not all(valid_coord(value) for value in [item.xmin, item.ymin, item.xmax, item.ymax]):
            issues.append(f"{prefix}:coordinate_out_of_range")
        if item.xmin >= item.xmax or item.ymin >= item.ymax:
            issues.append(f"{prefix}:invalid_box_order")

    for idx, item in enumerate(result.classifications):
        if not item.label:
            issues.append(f"classification[{idx}]:missing_label")
        if not 0.0 <= item.confidence <= 1.0:
            issues.append(f"classification[{idx}]:confidence_out_of_range")

    for idx, box in enumerate(result.boxes):
        validate_box(f"box[{idx}]", box)

    for idx, segment in enumerate(result.segments):
        if not segment.label:
            issues.append(f"segment[{idx}]:missing_label")
        if len(segment.polygon) < 3:
            issues.append(f"segment[{idx}]:too_few_points")
        for point_idx, point in enumerate(segment.polygon):
            if not valid_coord(point.x) or not valid_coord(point.y):
                issues.append(f"segment[{idx}].point[{point_idx}]:coordinate_out_of_range")

    for idx, pose in enumerate(result.poses):
        if not pose.keypoints:
            issues.append(f"pose[{idx}]:empty_keypoints")
        for point_idx, point in enumerate(pose.keypoints):
            if not point.name:
                issues.append(f"pose[{idx}].keypoint[{point_idx}]:missing_name")
            if not valid_coord(point.x) or not valid_coord(point.y):
                issues.append(f"pose[{idx}].keypoint[{point_idx}]:coordinate_out_of_range")

    for idx, text in enumerate(result.texts):
        validate_box(f"text[{idx}]", text)
        if not text.text:
            issues.append(f"text[{idx}]:missing_text")

    for idx, track in enumerate(result.tracks):
        validate_box(f"track[{idx}]", track)
        if not track.track_id:
            issues.append(f"track[{idx}]:missing_track_id")

    return issues


def summarize_validation(records: List[Dict[str, object]]) -> Dict[str, object]:
    total = len(records)
    failed = [record for record in records if record.get("issues")]
    issue_counts = {}
    for record in failed:
        for issue in record.get("issues", []):
            key = str(issue).split(":", 1)[0]
            issue_counts[key] = issue_counts.get(key, 0) + 1
    return {
        "total_records": total,
        "valid_records": total - len(failed),
        "failed_records": len(failed),
        "issue_counts": issue_counts,
    }
