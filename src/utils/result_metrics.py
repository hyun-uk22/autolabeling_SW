from ..core.models import DetectionResult


def result_confidences(result: DetectionResult) -> list[float]:
    values = []
    values.extend(item.confidence for item in result.classifications)
    values.extend(item.confidence for item in result.boxes)
    values.extend(item.confidence for item in result.segments)
    values.extend(item.confidence for item in result.poses)
    values.extend(item.confidence for item in result.texts)
    values.extend(item.confidence for item in result.tracks)
    for pose in result.poses:
        values.extend(point.confidence for point in pose.keypoints)
    return values


def mean_result_confidence(result: DetectionResult) -> float:
    values = result_confidences(result)
    return sum(values) / len(values) if values else 0.0


def count_result_labels(result: DetectionResult) -> int:
    return (
        len(result.classifications)
        + len(result.boxes)
        + len(result.segments)
        + len(result.poses)
        + len(result.texts)
        + len(result.tracks)
    )


def uncertainty_score(consistency: float, confidence: float) -> float:
    return 1.0 - ((consistency + confidence) / 2)
