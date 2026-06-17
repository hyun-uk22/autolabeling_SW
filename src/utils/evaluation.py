import os
from typing import Dict, List, Tuple

from .geometry import calculate_iou

YoloBox = Tuple[int, float, float, float, float]


def read_yolo_file(path: str) -> List[YoloBox]:
    boxes = []
    if not os.path.exists(path):
        return boxes

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            class_id = int(float(parts[0]))
            x_center, y_center, width, height = map(float, parts[1:])
            xmin = x_center - width / 2
            ymin = y_center - height / 2
            xmax = x_center + width / 2
            ymax = y_center + height / 2
            boxes.append((class_id, xmin, ymin, xmax, ymax))
    return boxes


def evaluate_yolo_dirs(pred_dir: str, gt_dir: str, iou_threshold: float = 0.5) -> Dict[str, float]:
    pred_files = [name for name in os.listdir(pred_dir) if name.endswith(".txt") and name != "classes.txt"]

    true_positive = 0
    false_positive = 0
    false_negative = 0
    matched_ious = []

    for file_name in pred_files:
        pred_boxes = read_yolo_file(os.path.join(pred_dir, file_name))
        gt_boxes = read_yolo_file(os.path.join(gt_dir, file_name))
        matched_gt = set()

        for pred in pred_boxes:
            best_iou = 0.0
            best_idx = -1
            for idx, gt in enumerate(gt_boxes):
                if idx in matched_gt or pred[0] != gt[0]:
                    continue
                iou = calculate_iou(pred[1:], gt[1:])
                if iou > best_iou:
                    best_iou = iou
                    best_idx = idx

            if best_iou >= iou_threshold:
                true_positive += 1
                matched_gt.add(best_idx)
                matched_ious.append(best_iou)
            else:
                false_positive += 1

        false_negative += len(gt_boxes) - len(matched_gt)

    precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
    recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
    mean_iou = sum(matched_ious) / len(matched_ious) if matched_ious else 0.0

    return {
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "precision": precision,
        "recall": recall,
        "mean_iou": mean_iou,
    }
