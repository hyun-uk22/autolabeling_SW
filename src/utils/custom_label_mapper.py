import json
import os
from typing import Any, Dict, Iterable, List, Optional, Tuple

from PIL import Image

from ..core.llm_client import extract_json, normalize_confidence
from ..core.models import BoundingBox, DetectionResult
from . import json_io


CUSTOM_MAPPING_FORMAT = "custom_mapping"
CUSTOM_MAPPING_SYSTEM_PROMPT = """
You infer a safe JSON mapping specification for a custom vision label file.
Return only JSON. Do not return code.

The returned JSON must use this schema:
{
  "format": "json",
  "image_name_path": "$.image.file_name",
  "image_width_path": "$.image.width",
  "image_height_path": "$.image.height",
  "objects_path": "$.annotations[*]",
  "label_path": "@.label",
  "bbox_path": "@.bbox",
  "bbox_format": "xywh",
  "bbox_unit": "pixel",
  "confidence_path": "$.score",
  "default_image_ext": ".jpg"
}

Supported bbox_format values: xywh, xyxy.
Supported bbox_unit values: pixel, normalized.
Use null for optional paths that are absent.
Use @.field for fields inside each object selected by objects_path.
"""


def _path_parts(path: Optional[str]) -> List[str]:
    if not path:
        return []
    text = str(path).strip()
    if text in {"$", "."}:
        return []
    if text.startswith("$."):
        text = text[2:]
    elif text.startswith("$"):
        text = text[1:].lstrip(".")
    return [part for part in text.split(".") if part]


def _children(value: Any, token: str) -> List[Any]:
    wildcard = token.endswith("[*]")
    key = token[:-3] if wildcard else token
    index = None
    if "[" in key and key.endswith("]"):
        base, raw_index = key[:-1].split("[", 1)
        key = base
        try:
            index = int(raw_index)
        except ValueError:
            index = None

    values = []
    if key:
        if isinstance(value, dict) and key in value:
            values = [value[key]]
        else:
            return []
    else:
        values = [value]

    if wildcard:
        output = []
        for item in values:
            if isinstance(item, list):
                output.extend(item)
        return output
    if index is not None:
        output = []
        for item in values:
            if isinstance(item, list) and 0 <= index < len(item):
                output.append(item[index])
        return output
    return values


def select_values(data: Any, path: Optional[str], context: Optional[Any] = None) -> List[Any]:
    if not path:
        return []
    if str(path).startswith("@"):
        value = context
        parts = _path_parts("$." + str(path)[1:].lstrip("."))
    else:
        value = data
        parts = _path_parts(path)
    values = [value]
    for part in parts:
        next_values = []
        for item in values:
            next_values.extend(_children(item, part))
        values = next_values
        if not values:
            break
    return values


def select_first(data: Any, path: Optional[str], context: Optional[Any] = None) -> Any:
    values = select_values(data, path, context=context)
    return values[0] if values else None


def select_first_with_context_fallback(data: Any, path: Optional[str], context: Optional[Any] = None) -> Any:
    value = select_first(data, path, context=context)
    if value is not None or context is None or not path:
        return value
    text = str(path).strip()
    if text.startswith("$."):
        return select_first(data, "@." + text[2:], context=context)
    return value


def _candidate_json_files(input_path: str) -> List[str]:
    if os.path.isfile(input_path):
        return [input_path] if input_path.lower().endswith(".json") else []
    candidates = []
    for root, _, names in os.walk(input_path):
        for name in sorted(names):
            if name.lower().endswith(".json"):
                candidates.append(os.path.join(root, name))
    return sorted(candidates)


def sample_custom_label_file(input_path: str) -> str:
    candidates = _candidate_json_files(input_path)
    if not candidates:
        raise ValueError("커스텀 매핑을 생성할 JSON 라벨 파일을 찾지 못했습니다.")
    return candidates[0]


def _first_key(item: Dict[str, Any], keys: Iterable[str]) -> Optional[str]:
    lowered = {str(key).lower(): key for key in item}
    for key in keys:
        found = lowered.get(key.lower())
        if found is not None:
            return str(found)
    return None


def _looks_like_two_point_bbox(value: Any) -> bool:
    return (
        isinstance(value, list)
        and len(value) >= 2
        and all(isinstance(point, list) and len(point) >= 2 for point in value[:2])
    )


def _looks_like_pixel_bbox(value: Any) -> bool:
    if _looks_like_two_point_bbox(value):
        coordinates = [float(value[0][0]), float(value[0][1]), float(value[1][0]), float(value[1][1])]
        return any(abs(coordinate) > 1.0 for coordinate in coordinates)
    if isinstance(value, list) and len(value) >= 4:
        return any(abs(float(item)) > 1.0 for item in value[:4])
    if isinstance(value, dict):
        numeric_values = []
        for key in ("xmin", "ymin", "xmax", "ymax", "x", "y", "width", "height", "w", "h"):
            if key in value:
                numeric_values.append(float(value[key]))
        return any(abs(item) > 1.0 for item in numeric_values)
    return False


def infer_custom_mapping_spec_heuristic(sample_path: str) -> Dict[str, Any]:
    data = json_io.load_file(sample_path)
    if not isinstance(data, dict):
        raise ValueError("커스텀 매핑은 현재 JSON object 라벨 파일만 지원합니다.")

    object_path = None
    object_sample = None
    for key in ("annotations", "objects", "labels", "items", "instances", "shapes"):
        value = data.get(key)
        if isinstance(value, list) and value and isinstance(value[0], dict):
            object_path = f"$.{key}[*]"
            object_sample = value[0]
            break
    if object_sample is None and any(key in data for key in ("bbox", "box", "points")):
        object_path = "$"
        object_sample = data
    if not isinstance(object_sample, dict):
        raise ValueError("샘플 JSON에서 객체 annotation 배열을 찾지 못했습니다.")

    label_key = _first_key(object_sample, ("label", "class", "class_name", "name", "category", "category_name"))
    bbox_key = _first_key(object_sample, ("bbox", "box", "bounding_box", "rect"))
    if not bbox_key:
        raise ValueError("샘플 JSON에서 bbox 필드를 찾지 못했습니다.")

    image_name_path = None
    for key in ("image", "image_name", "file_name", "filename", "path"):
        if key in data and isinstance(data[key], str):
            image_name_path = f"$.{key}"
            break
    if not image_name_path and isinstance(data.get("image"), dict):
        image_key = _first_key(data["image"], ("file_name", "filename", "name", "path"))
        if image_key:
            image_name_path = f"$.image.{image_key}"
    if not image_name_path and isinstance(data.get("images"), list) and data["images"]:
        first_image = data["images"][0]
        if isinstance(first_image, dict):
            image_key = _first_key(first_image, ("file_name", "filename", "name", "path"))
            if image_key:
                image_name_path = f"$.images[0].{image_key}"
    if not image_name_path:
        object_image_key = _first_key(object_sample, ("image", "image_name", "file_name", "filename", "path"))
        if object_image_key:
            image_name_path = f"@.{object_image_key}"

    width_path = "$.width" if "width" in data else None
    height_path = "$.height" if "height" in data else None
    if isinstance(data.get("image"), dict):
        if "width" in data["image"]:
            width_path = "$.image.width"
        if "height" in data["image"]:
            height_path = "$.image.height"
    if isinstance(data.get("images"), list) and data["images"]:
        first_image = data["images"][0]
        if isinstance(first_image, dict):
            if "width" in first_image:
                width_path = "$.images[0].width"
            if "height" in first_image:
                height_path = "$.images[0].height"

    bbox_value = object_sample.get(bbox_key)
    bbox_format = "xyxy" if _looks_like_two_point_bbox(bbox_value) else "xywh"
    bbox_unit = "pixel" if width_path and height_path else "normalized"
    if _looks_like_pixel_bbox(bbox_value):
        bbox_unit = "pixel"

    return {
        "format": "json",
        "image_name_path": image_name_path,
        "image_width_path": width_path,
        "image_height_path": height_path,
        "objects_path": object_path,
        "label_path": f"@.{label_key}" if label_key else None,
        "bbox_path": f"@.{bbox_key}",
        "bbox_format": bbox_format,
        "bbox_unit": bbox_unit,
        "confidence_path": "@.score" if "score" in object_sample else ("@.confidence" if "confidence" in object_sample else None),
        "default_image_ext": ".jpg",
    }


def infer_custom_mapping_spec_from_sample(sample_path: str, model_name: Optional[str] = None) -> Dict[str, Any]:
    if not model_name:
        return infer_custom_mapping_spec_heuristic(sample_path)
    with open(sample_path, "r", encoding="utf-8") as f:
        sample_text = f.read(8000)
    try:
        from ..workflow.conversation_router import _call_model

        response = _call_model(
            model_name,
            CUSTOM_MAPPING_SYSTEM_PROMPT,
            f"Sample file path: {sample_path}\nSample JSON:\n{sample_text}",
            json_output=True,
        )
        spec = extract_json(response)
        return normalize_custom_mapping_spec(spec)
    except Exception:
        return infer_custom_mapping_spec_heuristic(sample_path)


def normalize_custom_mapping_spec(spec: Any) -> Dict[str, Any]:
    if isinstance(spec, str):
        spec = json.loads(spec)
    if not isinstance(spec, dict):
        raise ValueError("커스텀 매핑 스펙은 JSON object여야 합니다.")
    normalized = dict(spec)
    normalized["format"] = "json"
    normalized["bbox_format"] = str(normalized.get("bbox_format") or "xywh").lower()
    normalized["bbox_unit"] = str(normalized.get("bbox_unit") or "normalized").lower()
    if normalized["bbox_format"] not in {"xywh", "xyxy"}:
        raise ValueError("bbox_format은 xywh 또는 xyxy만 지원합니다.")
    if normalized["bbox_unit"] not in {"pixel", "normalized"}:
        raise ValueError("bbox_unit은 pixel 또는 normalized만 지원합니다.")
    if not normalized.get("objects_path"):
        raise ValueError("objects_path가 필요합니다.")
    if not normalized.get("bbox_path"):
        raise ValueError("bbox_path가 필요합니다.")
    return normalized


def load_custom_mapping_spec(spec_or_path: Any) -> Dict[str, Any]:
    if isinstance(spec_or_path, dict):
        return normalize_custom_mapping_spec(spec_or_path)
    if not spec_or_path:
        raise ValueError("custom_mapping source_format에는 커스텀 매핑 스펙 JSON이 필요합니다.")
    text = str(spec_or_path).strip()
    if os.path.exists(text):
        return normalize_custom_mapping_spec(json_io.load_file(text))
    return normalize_custom_mapping_spec(json.loads(text))


def _bbox_from_value(value: Any) -> Optional[Tuple[float, float, float, float]]:
    if isinstance(value, dict):
        if all(key in value for key in ("xmin", "ymin", "xmax", "ymax")):
            return float(value["xmin"]), float(value["ymin"]), float(value["xmax"]), float(value["ymax"])
        if all(key in value for key in ("x", "y", "width", "height")):
            x = float(value["x"])
            y = float(value["y"])
            return x, y, x + float(value["width"]), y + float(value["height"])
        if all(key in value for key in ("x", "y", "w", "h")):
            x = float(value["x"])
            y = float(value["y"])
            return x, y, x + float(value["w"]), y + float(value["h"])
    if isinstance(value, list) and len(value) >= 4:
        return tuple(float(item) for item in value[:4])  # type: ignore[return-value]
    if (
        isinstance(value, list)
        and len(value) >= 2
        and all(isinstance(point, list) and len(point) >= 2 for point in value[:2])
    ):
        return (
            float(value[0][0]),
            float(value[0][1]),
            float(value[1][0]),
            float(value[1][1]),
        )
    return None


def _image_size(image_dir: str, image_name: str) -> Tuple[Optional[float], Optional[float]]:
    if not image_dir or not image_name:
        return None, None
    basename = os.path.basename(image_name)
    stem = os.path.splitext(basename)[0]
    candidates = [os.path.join(image_dir, image_name)]
    for ext in (".jpg", ".jpeg", ".png", ".webp", ".bmp"):
        candidates.append(os.path.join(image_dir, stem + ext))
    for path in candidates:
        if not os.path.exists(path):
            continue
        try:
            with Image.open(path) as image:
                return float(image.width), float(image.height)
        except OSError:
            return None, None
    return None, None


def _normalize_box(
    bbox: Tuple[float, float, float, float],
    bbox_format: str,
    bbox_unit: str,
    width: Optional[float],
    height: Optional[float],
) -> Tuple[float, float, float, float]:
    x1, y1, x2, y2 = bbox
    if bbox_format == "xywh":
        x2 = x1 + x2
        y2 = y1 + y2
    if bbox_unit == "pixel":
        if not width or not height:
            raise ValueError("pixel bbox를 정규화하려면 image_width_path/image_height_path 또는 원본 이미지가 필요합니다.")
        return x1 / width, y1 / height, x2 / width, y2 / height
    return x1, y1, x2, y2


def _default_image_name(path: str, spec: Dict[str, Any]) -> str:
    ext = str(spec.get("default_image_ext") or ".jpg")
    if not ext.startswith("."):
        ext = "." + ext
    return os.path.splitext(os.path.basename(path))[0] + ext


def import_custom_mapping(input_path: str, image_dir: str, spec_or_path: Any) -> List[Tuple[str, DetectionResult]]:
    spec = load_custom_mapping_spec(spec_or_path)
    records: List[Tuple[str, DetectionResult]] = []
    for path in _candidate_json_files(input_path):
        data = json_io.load_file(path)
        objects = select_values(data, spec.get("objects_path"))
        if not objects and isinstance(data, dict):
            objects = [data]
        image_name = select_first(data, spec.get("image_name_path")) or _default_image_name(path, spec)
        width = select_first(data, spec.get("image_width_path"))
        height = select_first(data, spec.get("image_height_path"))
        if width is None or height is None:
            found_width, found_height = _image_size(image_dir, str(image_name))
            width = width or found_width
            height = height or found_height
        result = DetectionResult(task_type="object_detection")
        for obj in objects:
            bbox_value = select_first_with_context_fallback(data, spec.get("bbox_path"), context=obj)
            bbox = _bbox_from_value(bbox_value)
            if bbox is None:
                continue
            label = select_first_with_context_fallback(data, spec.get("label_path"), context=obj) or "object"
            confidence = select_first_with_context_fallback(data, spec.get("confidence_path"), context=obj)
            xmin, ymin, xmax, ymax = _normalize_box(
                bbox,
                spec["bbox_format"],
                spec["bbox_unit"],
                float(width) if width is not None else None,
                float(height) if height is not None else None,
            )
            result.boxes.append(BoundingBox(
                label=str(label),
                xmin=xmin,
                ymin=ymin,
                xmax=xmax,
                ymax=ymax,
                confidence=normalize_confidence(confidence if confidence is not None else 1.0),
            ))
        if result.boxes:
            records.append((str(image_name), result))
    return records
