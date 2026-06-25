from typing import Any, Dict, List

from ..core.models import Point


def to_float(value: Any, default: float = 0.0) -> float:
    try:
        if value in ("", None):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def clamp_unit(value: Any) -> float:
    return max(0.0, min(1.0, to_float(value)))


def object_polygon_points(obj: Dict[str, Any], width: int, height: int) -> List[Point]:
    left = to_float(obj.get("left"))
    top = to_float(obj.get("top"))
    scale_x = to_float(obj.get("scaleX"), 1.0)
    scale_y = to_float(obj.get("scaleY"), 1.0)
    path_offset = obj.get("pathOffset") or {}
    offset_x = to_float(path_offset.get("x"), 0.0) if isinstance(path_offset, dict) else 0.0
    offset_y = to_float(path_offset.get("y"), 0.0) if isinstance(path_offset, dict) else 0.0
    points = obj.get("points") or []
    if not points and obj.get("path"):
        points = []
        for command in obj.get("path") or []:
            if not isinstance(command, (list, tuple)) or len(command) < 3:
                continue
            command_type = str(command[0]).upper()
            if command_type not in {"M", "L"}:
                continue
            points.append({"x": command[1], "y": command[2], "_path": True})

    raw_points = [
        (to_float(point.get("x")), to_float(point.get("y")))
        for point in points
        if isinstance(point, dict)
    ]
    absolute_like = False
    normalized_like = False
    if raw_points:
        min_x = min(x for x, _ in raw_points)
        max_x = max(x for x, _ in raw_points)
        min_y = min(y for _, y in raw_points)
        max_y = max(y for _, y in raw_points)
        object_width = abs(to_float(obj.get("width"), 0.0) * scale_x)
        object_height = abs(to_float(obj.get("height"), 0.0) * scale_y)
        normalized_like = (
            0 <= min_x <= 1
            and 0 <= max_x <= 1
            and 0 <= min_y <= 1
            and 0 <= max_y <= 1
            and object_width <= 1
            and object_height <= 1
        )
        # Fabric payloads differ by drawing mode/version: polygon points may be
        # object-local or already canvas-absolute. Avoid adding left/top twice.
        absolute_like = (
            0 <= min_x <= width
            and 0 <= max_x <= width
            and 0 <= min_y <= height
            and 0 <= max_y <= height
            and object_width > 0
            and object_height > 0
            and (max_x > object_width + 2 or max_y > object_height + 2)
        )

    result = []
    for point in points:
        point_x = to_float(point.get("x"))
        point_y = to_float(point.get("y"))
        if point.get("_absolute"):
            x = point_x / max(width, 1)
            y = point_y / max(height, 1)
        elif point.get("_path") and absolute_like:
            x = point_x / max(width, 1)
            y = point_y / max(height, 1)
        elif point.get("_path"):
            x = (left + point_x * scale_x) / max(width, 1)
            y = (top + point_y * scale_y) / max(height, 1)
        elif absolute_like:
            x = point_x / max(width, 1)
            y = point_y / max(height, 1)
        elif normalized_like:
            x = point_x
            y = point_y
        else:
            x = (left + (point_x - offset_x) * scale_x) / max(width, 1)
            y = (top + (point_y - offset_y) * scale_y) / max(height, 1)
        result.append(Point(x=clamp_unit(x), y=clamp_unit(y)))
    return result
