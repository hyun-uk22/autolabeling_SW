import numpy as np
from typing import List
from ..core.models import DetectionResult, BoundingBox

def calculate_iou(boxA, boxB):
    """Calculates Intersection over Union (IoU) for two boxes [xmin, ymin, xmax, ymax]"""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)
    if interArea == 0:
        return 0.0

    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou

def compute_pairwise_consistency(boxes1: List[BoundingBox], boxes2: List[BoundingBox], iou_threshold=0.5) -> float:
    """
    Computes consistency between two sets of detections using greedy matching.
    Penalizes missing or extra boxes.
    """
    if not boxes1 and not boxes2:
        return 1.0
    if not boxes1 or not boxes2:
        return 0.0
        
    matched = set()
    total_iou = 0.0

    # Greedy matching logic
    for b1 in boxes1:
        best_iou = 0
        best_j = -1
        for j, b2 in enumerate(boxes2):
            if j in matched or b1.label != b2.label:
                continue
            iou = calculate_iou(
                [b1.xmin, b1.ymin, b1.xmax, b1.ymax],
                [b2.xmin, b2.ymin, b2.xmax, b2.ymax]
            )
            if iou > best_iou:
                best_iou = iou
                best_j = j
                
        if best_iou >= iou_threshold:
            matched.add(best_j)
            total_iou += best_iou
            
    max_boxes = max(len(boxes1), len(boxes2))
    if max_boxes == 0: 
        return 1.0
        
    # Consistency is avg IoU of matches, naturally penalized by unmatched boxes
    # because we divide by the maximum number of boxes present in either set.
    return total_iou / max_boxes

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

def compute_result_consistency(result1: DetectionResult, result2: DetectionResult) -> float:
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
    if result1.segments and result2.segments:
        boxes1 = [segment_to_box(segment) for segment in result1.segments]
        boxes2 = [segment_to_box(segment) for segment in result2.segments]
        return compute_pairwise_consistency(boxes1, boxes2)
    return jaccard_similarity(get_result_labels(result1), get_result_labels(result2))

def get_consistency_score(detections: List[DetectionResult]) -> float:
    """
    Calculates overall consistency across multiple inferences.
    This corresponds to the 'Uncertainty Measurement' in the paper.
    """
    if len(detections) < 2:
        return 1.0
        
    consistencies = []
    # Compare all pairs (e.g., 0-1, 0-2, 1-2 for 3 inferences)
    for i in range(len(detections)):
        for j in range(i + 1, len(detections)):
            score = compute_result_consistency(detections[i], detections[j])
            consistencies.append(score)
            
    return np.mean(consistencies) if consistencies else 0.0


def consistency_metric_name(task_type: str) -> str:
    if task_type in {"object_detection", "segmentation", "tracking"}:
        return "pairwise_iou_consistency"
    if task_type == "classification":
        return "label_jaccard_consistency"
    if task_type in {"pose_estimation", "ocr"}:
        return "semantic_label_consistency"
    return "mixed_consistency"
