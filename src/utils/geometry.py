import math
import unicodedata
from typing import Callable, Iterable, List, Sequence

import numpy as np
from PIL import Image, ImageDraw

from ..core.models import BoundingBox, DetectionResult


POLYGON_MASK_SIZE = 512
POSE_SIGMA = 0.1


def calculate_iou(box_a, box_b):
    """Calculate IoU for boxes represented as [xmin, ymin, xmax, ymax]."""
    xmin = max(box_a[0], box_b[0])
    ymin = max(box_a[1], box_b[1])
    xmax = min(box_a[2], box_b[2])
    ymax = min(box_a[3], box_b[3])
    intersection = max(0.0, xmax - xmin) * max(0.0, ymax - ymin)
    if intersection == 0:
        return 0.0

    area_a = max(0.0, box_a[2] - box_a[0]) * max(0.0, box_a[3] - box_a[1])
    area_b = max(0.0, box_b[2] - box_b[0]) * max(0.0, box_b[3] - box_b[1])
    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.0


def _greedy_collection_score(
    left: Sequence,
    right: Sequence,
    similarity: Callable[[object, object], float],
    compatible: Callable[[object, object], bool],
    minimum: float = 0.0,
) -> float:
    if not left and not right:
        return 1.0
    if not left or not right:
        return 0.0

    candidates = []
    for left_index, left_item in enumerate(left):
        for right_index, right_item in enumerate(right):
            if not compatible(left_item, right_item):
                continue
            score = float(similarity(left_item, right_item))
            if score >= minimum and score > 0.0:
                candidates.append((score, left_index, right_index))

    matched_left = set()
    matched_right = set()
    total = 0.0
    for score, left_index, right_index in sorted(candidates, reverse=True):
        if left_index in matched_left or right_index in matched_right:
            continue
        matched_left.add(left_index)
        matched_right.add(right_index)
        total += score
    return total / max(len(left), len(right))


def _box_values(item) -> List[float]:
    return [item.xmin, item.ymin, item.xmax, item.ymax]


def compute_pairwise_consistency(
    boxes1: List[BoundingBox],
    boxes2: List[BoundingBox],
    iou_threshold: float = 0.5,
) -> float:
    return _greedy_collection_score(
        boxes1,
        boxes2,
        lambda left, right: calculate_iou(_box_values(left), _box_values(right)),
        lambda left, right: left.label == right.label,
        minimum=iou_threshold,
    )


def _polygon_mask(segment, size: int = POLYGON_MASK_SIZE) -> np.ndarray:
    mask = Image.new("1", (size, size), 0)
    if len(segment.polygon) < 3:
        return np.zeros((size, size), dtype=bool)
    points = [
        (
            int(round(min(1.0, max(0.0, point.x)) * (size - 1))),
            int(round(min(1.0, max(0.0, point.y)) * (size - 1))),
        )
        for point in segment.polygon
    ]
    ImageDraw.Draw(mask).polygon(points, fill=1)
    return np.asarray(mask, dtype=bool)


def polygon_mask_iou(left, right, size: int = POLYGON_MASK_SIZE) -> float:
    left_mask = _polygon_mask(left, size)
    right_mask = _polygon_mask(right, size)
    union = np.logical_or(left_mask, right_mask).sum()
    if union == 0:
        return 0.0
    intersection = np.logical_and(left_mask, right_mask).sum()
    return float(intersection / union)


def compute_segmentation_consistency(segments1, segments2, iou_threshold: float = 0.5) -> float:
    masks = {
        id(segment): _polygon_mask(segment)
        for segment in [*segments1, *segments2]
    }

    def similarity(left, right):
        left_mask = masks[id(left)]
        right_mask = masks[id(right)]
        union = np.logical_or(left_mask, right_mask).sum()
        if union == 0:
            return 0.0
        return float(np.logical_and(left_mask, right_mask).sum() / union)

    return _greedy_collection_score(
        segments1,
        segments2,
        similarity,
        lambda left, right: left.label == right.label,
        minimum=iou_threshold,
    )


def _visible_keypoints(pose) -> dict:
    return {point.name: point for point in pose.keypoints if point.visible and point.name}


def _pose_scale(points: Iterable) -> float:
    points = list(points)
    if not points:
        return 0.1
    width = max(point.x for point in points) - min(point.x for point in points)
    height = max(point.y for point in points) - min(point.y for point in points)
    return max(width, height, 0.1)


def pose_oks_similarity(left, right, sigma: float = POSE_SIGMA) -> float:
    left_points = _visible_keypoints(left)
    right_points = _visible_keypoints(right)
    names = set(left_points) | set(right_points)
    common = set(left_points) & set(right_points)
    if not names or not common:
        return 0.0

    scale = (_pose_scale(left_points.values()) + _pose_scale(right_points.values())) / 2
    denominator = 2 * (scale * sigma) ** 2
    total = 0.0
    for name in common:
        left_point = left_points[name]
        right_point = right_points[name]
        distance_squared = (left_point.x - right_point.x) ** 2 + (left_point.y - right_point.y) ** 2
        total += math.exp(-distance_squared / denominator)
    return total / len(names)


def compute_pose_consistency(poses1, poses2) -> float:
    return _greedy_collection_score(
        poses1,
        poses2,
        pose_oks_similarity,
        lambda left, right: left.label == right.label,
        minimum=0.01,
    )


def normalize_ocr_text(value: str) -> str:
    normalized = unicodedata.normalize("NFC", value or "").casefold()
    return " ".join(normalized.split())


def levenshtein_distance(left: str, right: str) -> int:
    if len(left) < len(right):
        left, right = right, left
    previous = list(range(len(right) + 1))
    for left_index, left_character in enumerate(left, start=1):
        current = [left_index]
        for right_index, right_character in enumerate(right, start=1):
            current.append(min(
                current[-1] + 1,
                previous[right_index] + 1,
                previous[right_index - 1] + (left_character != right_character),
            ))
        previous = current
    return previous[-1]


def character_similarity(left: str, right: str) -> float:
    left = normalize_ocr_text(left)
    right = normalize_ocr_text(right)
    length = max(len(left), len(right))
    if length == 0:
        return 1.0
    return 1.0 - (levenshtein_distance(left, right) / length)


def ocr_region_similarity(left, right) -> float:
    spatial_iou = calculate_iou(_box_values(left), _box_values(right))
    if spatial_iou <= 0.0:
        return 0.0
    return 0.4 * spatial_iou + 0.6 * character_similarity(left.text, right.text)


def compute_ocr_consistency(texts1, texts2) -> float:
    return _greedy_collection_score(
        texts1,
        texts2,
        ocr_region_similarity,
        lambda _left, _right: True,
        minimum=0.01,
    )


def jaccard_similarity(values1, values2) -> float:
    set1 = set(values1)
    set2 = set(values2)
    if not set1 and not set2:
        return 1.0
    if not set1 or not set2:
        return 0.0
    return len(set1 & set2) / len(set1 | set2)


def segment_to_box(segment) -> BoundingBox:
    xs = [point.x for point in segment.polygon]
    ys = [point.y for point in segment.polygon]
    return BoundingBox(
        label=segment.label,
        xmin=min(xs),
        ymin=min(ys),
        xmax=max(xs),
        ymax=max(ys),
        confidence=segment.confidence,
    )


def get_result_labels(result: DetectionResult) -> List[str]:
    labels = []
    labels.extend(item.label for item in result.classifications)
    labels.extend(item.label for item in result.boxes)
    labels.extend(item.label for item in result.segments)
    labels.extend(item.label for item in result.poses)
    labels.extend(item.text for item in result.texts)
    labels.extend(f"{item.track_id}:{item.label}" for item in result.tracks)
    for pose in result.poses:
        labels.extend(point.name for point in pose.keypoints if point.visible)
    return labels


def consistency_metric_name(task_type: str) -> str:
    return {
        "object_detection": "bounding_box_iou",
        "segmentation": "polygon_mask_iou",
        "pose_estimation": "pose_oks",
        "ocr": "ocr_region_iou_character_similarity",
        "classification": "label_jaccard",
        "tracking": "label_jaccard",
        "all": "multimodal_consistency",
    }.get(task_type, "label_jaccard")


def _multimodal_consistency(result1: DetectionResult, result2: DetectionResult) -> float:
    scores = []
    if result1.classifications or result2.classifications:
        scores.append(jaccard_similarity(
            [item.label for item in result1.classifications],
            [item.label for item in result2.classifications],
        ))
    if result1.boxes or result2.boxes:
        scores.append(compute_pairwise_consistency(result1.boxes, result2.boxes))
    if result1.segments or result2.segments:
        scores.append(compute_segmentation_consistency(result1.segments, result2.segments))
    if result1.poses or result2.poses:
        scores.append(compute_pose_consistency(result1.poses, result2.poses))
    if result1.texts or result2.texts:
        scores.append(compute_ocr_consistency(result1.texts, result2.texts))
    if result1.tracks or result2.tracks:
        scores.append(jaccard_similarity(
            [f"{item.track_id}:{item.label}" for item in result1.tracks],
            [f"{item.track_id}:{item.label}" for item in result2.tracks],
        ))
    return sum(scores) / len(scores) if scores else 1.0


def compute_result_consistency(result1: DetectionResult, result2: DetectionResult) -> float:
    if result1.task_type == "all" or result2.task_type == "all":
        return _multimodal_consistency(result1, result2)
    if result1.segments and result2.segments:
        return compute_segmentation_consistency(result1.segments, result2.segments)
    if result1.poses and result2.poses:
        return compute_pose_consistency(result1.poses, result2.poses)
    if result1.texts and result2.texts:
        return compute_ocr_consistency(result1.texts, result2.texts)
    if result1.boxes and result2.boxes:
        return compute_pairwise_consistency(result1.boxes, result2.boxes)
    if result1.boxes and result2.segments:
        return compute_pairwise_consistency(
            result1.boxes,
            [segment_to_box(segment) for segment in result2.segments],
        )
    if result1.segments and result2.boxes:
        return compute_pairwise_consistency(
            [segment_to_box(segment) for segment in result1.segments],
            result2.boxes,
        )
    return jaccard_similarity(get_result_labels(result1), get_result_labels(result2))


def get_consistency_score(detections: List[DetectionResult]) -> float:
    """Average pairwise self-consistency across repeated inferences."""
    if len(detections) < 2:
        return 1.0

    consistencies = []
    for left_index in range(len(detections)):
        for right_index in range(left_index + 1, len(detections)):
            consistencies.append(compute_result_consistency(
                detections[left_index],
                detections[right_index],
            ))
    return float(np.mean(consistencies)) if consistencies else 0.0
