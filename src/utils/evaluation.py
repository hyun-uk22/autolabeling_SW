import csv
import json
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
    pred_files = {
        name for name in os.listdir(pred_dir)
        if name.endswith(".txt") and name != "classes.txt"
    }
    gt_files = {
        name for name in os.listdir(gt_dir)
        if name.endswith(".txt") and name != "classes.txt"
    }
    all_files = sorted(pred_files | gt_files)

    true_positive = 0
    false_positive = 0
    false_negative = 0
    matched_ious = []

    for file_name in all_files:
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
    f1 = (2 * precision * recall / (precision + recall)) if precision + recall else 0.0
    mean_iou = sum(matched_ious) / len(matched_ious) if matched_ious else 0.0

    return {
        "images": len(all_files),
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "mean_iou": mean_iou,
    }


def summarize_run_metrics(metrics_csv_path: str) -> Dict[str, float]:
    if not os.path.exists(metrics_csv_path):
        return {
            "processed_images": 0,
            "total_labels": 0,
            "avg_elapsed_sec": 0.0,
            "low_api_attempts": 0,
            "high_api_attempts": 0,
            "escalation_rate": 0.0,
            "avg_consistency": 0.0,
            "avg_uncertainty": 0.0,
        }

    with open(metrics_csv_path, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    def to_float(row, key):
        try:
            value = row.get(key, "")
            return float(value) if value not in ("", None) else 0.0
        except ValueError:
            return 0.0

    processed = len(rows)
    total_labels = sum(to_float(row, "objects") for row in rows)
    elapsed = sum(to_float(row, "elapsed_sec") for row in rows)
    low_attempts = sum(to_float(row, "low_api_attempts") for row in rows)
    high_attempts = sum(to_float(row, "high_api_attempts") for row in rows)
    escalated = sum(1 for row in rows if row.get("status") == "Escalated")
    consistency_values = [to_float(row, "consistency_score") for row in rows if row.get("consistency_score") not in ("", None)]
    uncertainty_values = [to_float(row, "uncertainty_score") for row in rows if row.get("uncertainty_score") not in ("", None)]

    return {
        "processed_images": processed,
        "total_labels": total_labels,
        "avg_elapsed_sec": elapsed / processed if processed else 0.0,
        "total_elapsed_sec": elapsed,
        "low_api_attempts": low_attempts,
        "high_api_attempts": high_attempts,
        "escalation_rate": escalated / processed if processed else 0.0,
        "avg_consistency": sum(consistency_values) / len(consistency_values) if consistency_values else 0.0,
        "avg_uncertainty": sum(uncertainty_values) / len(uncertainty_values) if uncertainty_values else 0.0,
    }


def build_experiment_report(
    runs: Dict[str, str],
    gt_dir: str | None = None,
    iou_threshold: float = 0.5,
    manual_time_per_image: float = 45.0,
    low_unit_cost: float = 1.0,
    high_unit_cost: float = 10.0,
) -> List[Dict[str, float]]:
    report = []
    for name, run_dir in runs.items():
        row = {"method": name, "run_dir": run_dir}
        row.update(summarize_run_metrics(os.path.join(run_dir, "run_metrics.csv")))
        if gt_dir:
            row.update(evaluate_yolo_dirs(run_dir, gt_dir, iou_threshold))

        processed = row.get("processed_images", 0)
        manual_time = processed * manual_time_per_image
        total_elapsed = row.get("total_elapsed_sec", 0.0)
        row["manual_time_sec"] = manual_time
        row["time_saved_pct"] = ((manual_time - total_elapsed) / manual_time * 100) if manual_time else 0.0
        row["estimated_relative_cost"] = (
            row.get("low_api_attempts", 0.0) * low_unit_cost
            + row.get("high_api_attempts", 0.0) * high_unit_cost
        )
        report.append(row)
    return report


def save_experiment_report(report: List[Dict[str, float]], output_dir: str) -> Dict[str, str]:
    os.makedirs(output_dir, exist_ok=True)
    json_path = os.path.join(output_dir, "experiment_report.json")
    csv_path = os.path.join(output_dir, "experiment_report.csv")
    md_path = os.path.join(output_dir, "experiment_report.md")

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    fieldnames = sorted({key for row in report for key in row.keys()})
    if fieldnames:
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(report)

    preferred = [
        "method", "precision", "recall", "f1", "mean_iou",
        "avg_elapsed_sec", "time_saved_pct", "low_api_attempts",
        "high_api_attempts", "escalation_rate", "estimated_relative_cost",
    ]
    with open(md_path, "w", encoding="utf-8") as f:
        columns = [col for col in preferred if any(col in row for row in report)]
        f.write("| " + " | ".join(columns) + " |\n")
        f.write("| " + " | ".join("---" for _ in columns) + " |\n")
        for row in report:
            values = []
            for col in columns:
                value = row.get(col, "")
                if isinstance(value, float):
                    value = f"{value:.4f}"
                values.append(str(value))
            f.write("| " + " | ".join(values) + " |\n")

    return {"json": json_path, "csv": csv_path, "markdown": md_path}
