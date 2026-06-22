from copy import deepcopy
from typing import Iterable, List, Tuple

from ..core.models import DetectionResult
from ..utils.geometry import calculate_iou, compute_result_consistency
from ..utils.result_metrics import mean_result_confidence
from .base import VisionTaskPlugin


def _box_iou(left, right) -> float:
    return calculate_iou(
        [left.xmin, left.ymin, left.xmax, left.ymax],
        [right.xmin, right.ymin, right.xmax, right.ymax],
    )


def _merge_boxes(target, incoming, match_iou: float = 0.35, min_confidence: float = 0.0):
    for candidate in incoming:
        if candidate.confidence < min_confidence:
            continue
        best_index = -1
        best_iou = 0.0
        for index, current in enumerate(target):
            if current.label != candidate.label:
                continue
            iou = _box_iou(current, candidate)
            if iou > best_iou:
                best_iou = iou
                best_index = index
        if best_index >= 0 and best_iou >= match_iou:
            current = target[best_index]
            candidate.confidence = (current.confidence + candidate.confidence) / 2
            target[best_index] = candidate
        else:
            target.append(candidate)


def _dedupe_boxes(boxes, nms_iou: float = 0.6):
    kept = []
    for candidate in sorted(boxes, key=lambda item: item.confidence, reverse=True):
        duplicate = False
        for current in kept:
            if current.label == candidate.label and _box_iou(current, candidate) >= nms_iou:
                duplicate = True
                break
        if not duplicate:
            kept.append(candidate)
    boxes[:] = kept


def _merge_by_key(target, incoming, key):
    indexes = {key(item): index for index, item in enumerate(target)}
    for item in incoming:
        item_key = key(item)
        if item_key in indexes:
            current = target[indexes[item_key]]
            if getattr(item, "confidence", 0.0) >= getattr(current, "confidence", 0.0):
                target[indexes[item_key]] = item
        else:
            indexes[item_key] = len(target)
            target.append(item)


def merge_results(
    base: DetectionResult,
    incoming: DetectionResult,
    match_iou: float = 0.35,
    nms_iou: float = 0.6,
    min_confidence: float = 0.0,
) -> DetectionResult:
    result = deepcopy(base)
    _merge_by_key(result.classifications, incoming.classifications, lambda item: item.label)
    result.boxes = [box for box in result.boxes if box.confidence >= min_confidence]
    _merge_boxes(result.boxes, incoming.boxes, match_iou=match_iou, min_confidence=min_confidence)
    _dedupe_boxes(result.boxes, nms_iou=nms_iou)
    _merge_by_key(
        result.segments,
        incoming.segments,
        lambda item: (
            item.label,
            round(sum(point.x for point in item.polygon) / len(item.polygon), 1) if item.polygon else 0,
            round(sum(point.y for point in item.polygon) / len(item.polygon), 1) if item.polygon else 0,
        ),
    )
    _merge_by_key(
        result.poses,
        incoming.poses,
        lambda item: (
            item.label,
            round(sum(point.x for point in item.keypoints) / len(item.keypoints), 1) if item.keypoints else 0,
            round(sum(point.y for point in item.keypoints) / len(item.keypoints), 1) if item.keypoints else 0,
        ),
    )
    _merge_by_key(result.texts, incoming.texts, lambda item: (item.text, round(item.xmin, 2), round(item.ymin, 2)))
    _merge_by_key(result.tracks, incoming.tracks, lambda item: (item.frame_id, item.track_id))
    return result


class TaskPluginOrchestrator:
    def __init__(self, plugins: Iterable[VisionTaskPlugin], fail_fast: bool = False):
        self.plugins = list(plugins)
        self.fail_fast = fail_fast

    @property
    def names(self) -> List[str]:
        return [plugin.plugin_name for plugin in self.plugins]

    def prepare(self, task_type: str) -> List[dict]:
        records = []
        for plugin in self.plugins:
            try:
                record = plugin.prepare(task_type)
                records.append(record)
            except Exception as exc:
                records.append({"plugin": plugin.plugin_name, "status": "error", "error": str(exc)})
                if self.fail_fast:
                    raise
        return records

    def process(
        self,
        image_path: str,
        prompt: str,
        task_type: str,
        seed_result: DetectionResult,
        config_overrides: dict | None = None,
    ) -> Tuple[DetectionResult, List[dict]]:
        result = deepcopy(seed_result)
        records = []
        weighted_scores = []
        config_overrides = config_overrides or {}

        for plugin in self.plugins:
            if not plugin.supports(task_type):
                continue
            try:
                original_config = plugin.config
                try:
                    if config_overrides:
                        plugin.config = {**plugin.config, **dict(config_overrides.get(plugin.plugin_name) or {})}
                    match_iou = float(plugin.config.get("merge_iou", 0.35))
                    nms_iou = float(plugin.config.get("nms_iou", 0.6))
                    min_confidence = float(plugin.config.get("min_confidence", 0.0))
                    output = plugin.refine(image_path, prompt, result)
                finally:
                    plugin.config = original_config
                agreement = compute_result_consistency(result, output.result)
                score = output.score if output.score is not None else agreement
                weight = float(plugin.config.get("weight", 1.0))
                weighted_scores.append((score, weight))
                result = merge_results(
                    result,
                    output.result,
                    match_iou=match_iou,
                    nms_iou=nms_iou,
                    min_confidence=min_confidence,
                )
                result.plugin_scores[plugin.plugin_name] = score
                result.plugin_metadata[plugin.plugin_name] = output.metadata
                records.append({"plugin": plugin.plugin_name, "status": "ok", "score": score, "agreement": agreement})
            except Exception as exc:
                records.append({"plugin": plugin.plugin_name, "status": "error", "error": str(exc)})
                if self.fail_fast:
                    raise

        if weighted_scores:
            plugin_score = sum(score * weight for score, weight in weighted_scores) / sum(weight for _, weight in weighted_scores)
            previous = result.consistency_score if result.consistency_score is not None else plugin_score
            result.consistency_score = (previous + plugin_score) / 2
            result.mean_confidence = mean_result_confidence(result)
            result.uncertainty_score = 1.0 - ((result.consistency_score + result.mean_confidence) / 2)
            used = [record["plugin"] for record in records if record["status"] == "ok"]
            if used:
                prefix = f"{result.source_model}+" if result.source_model else ""
                result.source_model = f"{prefix}{'+'.join(used)}"

        return result, records
