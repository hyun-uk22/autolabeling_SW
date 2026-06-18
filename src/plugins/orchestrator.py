from copy import deepcopy
from typing import Iterable, List, Tuple

from ..core.models import DetectionResult
from ..utils.geometry import calculate_iou, compute_result_consistency
from .base import VisionTaskPlugin


def _box_iou(left, right) -> float:
    return calculate_iou(
        [left.xmin, left.ymin, left.xmax, left.ymax],
        [right.xmin, right.ymin, right.xmax, right.ymax],
    )


def _merge_boxes(target, incoming, match_iou: float = 0.5):
    for candidate in incoming:
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


def merge_results(base: DetectionResult, incoming: DetectionResult) -> DetectionResult:
    result = deepcopy(base)
    _merge_by_key(result.classifications, incoming.classifications, lambda item: item.label)
    _merge_boxes(result.boxes, incoming.boxes)
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


def mean_result_confidence(result: DetectionResult) -> float:
    values = []
    for field in ["classifications", "boxes", "segments", "poses", "texts", "tracks"]:
        values.extend(getattr(item, "confidence", 1.0) for item in getattr(result, field))
    return sum(values) / len(values) if values else 0.0


class TaskPluginOrchestrator:
    def __init__(self, plugins: Iterable[VisionTaskPlugin], fail_fast: bool = False):
        self.plugins = list(plugins)
        self.fail_fast = fail_fast

    @property
    def names(self) -> List[str]:
        return [plugin.plugin_name for plugin in self.plugins]

    def process(
        self,
        image_path: str,
        prompt: str,
        task_type: str,
        seed_result: DetectionResult,
    ) -> Tuple[DetectionResult, List[dict]]:
        result = deepcopy(seed_result)
        records = []
        weighted_scores = []

        for plugin in self.plugins:
            if not plugin.supports(task_type):
                continue
            try:
                output = plugin.refine(image_path, prompt, result)
                agreement = compute_result_consistency(result, output.result)
                score = output.score if output.score is not None else agreement
                weight = float(plugin.config.get("weight", 1.0))
                weighted_scores.append((score, weight))
                result = merge_results(result, output.result)
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
                result.source_model = f"{result.source_model}+{'+'.join(used)}"

        return result, records
