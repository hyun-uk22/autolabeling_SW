import importlib
import json
from typing import Any, Dict, Iterable, List, Optional, Type

from .base import VisionTaskPlugin


DEFAULT_GENERATION_PLUGIN_CONFIGS: List[Dict[str, Any]] = [
    {
        "name": "classification",
        "enabled": True,
        "tasks": ["classification"],
        "weight": 1.0,
        "config": {
            "model": "google/siglip-base-patch16-224",
            "device": "auto",
            "labels": ["person", "animal", "vehicle", "document", "indoor", "outdoor"],
            "top_k": 3,
            "prompt_template": "a photo of a {label}",
        },
    },
    {
        "name": "grounding_dino",
        "enabled": True,
        "tasks": ["object_detection"],
        "weight": 1.2,
        "config": {
            "model": "IDEA-Research/grounding-dino-base",
            "device": "auto",
            "box_threshold": 0.45,
            "text_threshold": 0.30,
            "merge_iou": 0.35,
            "nms_iou": 0.60,
            "min_confidence": 0.20,
        },
    },
    {
        "name": "grounded_sam2",
        "enabled": True,
        "tasks": ["segmentation"],
        "weight": 1.5,
        "config": {
            "grounding_model": "IDEA-Research/grounding-dino-base",
            "sam_backend": "ultralytics_sam2",
            "sam_model": "sam2_b.pt",
            "device": "auto",
            "box_threshold": 0.45,
            "text_threshold": 0.30,
            "merge_iou": 0.35,
            "nms_iou": 0.60,
            "min_confidence": 0.20,
        },
    },
    {
        "name": "sam",
        "enabled": False,
        "tasks": ["segmentation"],
        "weight": 1.0,
        "config": {
            "backend": "ultralytics_sam2",
            "model": "sam2_b.pt",
            "device": "auto",
        },
    },
    {
        "name": "pose",
        "enabled": True,
        "tasks": ["pose_estimation"],
        "weight": 1.0,
        "config": {
            "model": "yolo26l-pose.pt",
            "device": "auto",
            "keypoint_threshold": 0.25,
        },
    },
    {
        "name": "vitpose",
        "enabled": False,
        "tasks": ["pose_estimation"],
        "weight": 1.2,
        "config": {
            "pose_config": "configs/body_2d_keypoint/topdown_heatmap/coco/td-hm_ViTPose-large_8xb64-210e_coco-256x192.py",
            "pose_checkpoint": "vitpose-l.pth",
            "det_config": "demo/mmdetection_cfg/rtmdet_m_640-8xb32_coco-person.py",
            "det_checkpoint": "https://download.openmmlab.com/mmpose/v1/projects/rtmpose/rtmdet_m_8xb32-100e_coco-obj365-person-235e8209.pth",
            "device": "auto",
            "keypoint_threshold": 0.5,
            "det_threshold": 0.5,
            "label": "person",
        },
    },
    {
        "name": "ocr",
        "enabled": True,
        "tasks": ["ocr"],
        "weight": 1.0,
        "config": {
            "backend": "paddleocr",
            "ocr_version": "PP-OCRv5",
            "text_detection_model_name": "PP-OCRv5_mobile_det",
            "text_recognition_model_name": "korean_PP-OCRv5_mobile_rec",
            "lang": "korean",
            "languages": ["ko", "en"],
            "gpu": "auto",
        },
    },
    {
        "name": "tracking",
        "enabled": True,
        "tasks": ["tracking"],
        "weight": 1.0,
        "config": {
            "model": "yolo26n.pt",
            "tracker": "bytetrack.yaml",
            "device": "auto",
        },
    },
]


class PluginRegistry:
    def __init__(self):
        self._plugins: Dict[str, Type[VisionTaskPlugin]] = {}

    def register(self, name: str, plugin_class: Type[VisionTaskPlugin]) -> None:
        if not issubclass(plugin_class, VisionTaskPlugin):
            raise TypeError(f"Plugin {name} must inherit VisionTaskPlugin")
        self._plugins[name] = plugin_class

    def register_path(self, name: str, class_path: str) -> None:
        if ":" not in class_path:
            raise ValueError("Plugin class path must use module:ClassName format")
        module_name, class_name = class_path.split(":", 1)
        module = importlib.import_module(module_name)
        self.register(name, getattr(module, class_name))

    def create(self, name: str, config: Dict[str, Any] | None = None) -> VisionTaskPlugin:
        if name not in self._plugins:
            raise KeyError(f"Unknown plugin: {name}. Available: {', '.join(sorted(self._plugins))}")
        return self._plugins[name](config or {})

    def create_from_specs(self, specs: Iterable[Dict[str, Any]]) -> List[VisionTaskPlugin]:
        plugins = []
        for item in specs:
            if not item.get("enabled", True):
                continue
            name = item["name"]
            if item.get("class"):
                self.register_path(name, item["class"])
            config = dict(item.get("config") or {})
            if item.get("tasks"):
                config["tasks"] = item["tasks"]
            config["weight"] = float(item.get("weight", config.get("weight", 1.0)))
            plugins.append(self.create(name, config))
        return plugins

    def load_config(self, path: str) -> List[VisionTaskPlugin]:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return self.create_from_specs(data.get("plugins", []))

    def names(self) -> Iterable[str]:
        return self._plugins.keys()


def _merge_generation_plugin_specs(config_path: Optional[str]) -> List[Dict[str, Any]]:
    specs_by_name = {
        item["name"]: dict(item)
        for item in DEFAULT_GENERATION_PLUGIN_CONFIGS
    }
    order = [item["name"] for item in DEFAULT_GENERATION_PLUGIN_CONFIGS]
    if config_path:
        with open(config_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        for item in data.get("plugins", []):
            name = item["name"]
            if name in specs_by_name:
                merged = dict(specs_by_name[name])
                merged["config"] = {
                    **dict(merged.get("config") or {}),
                    **dict(item.get("config") or {}),
                }
                if "enabled" in item:
                    merged["enabled"] = item["enabled"]
                elif not merged.get("enabled", True):
                    merged["enabled"] = True
                if item.get("tasks"):
                    merged["tasks"] = item["tasks"]
                if "weight" in item:
                    merged["weight"] = item["weight"]
                if item.get("class"):
                    merged["class"] = item["class"]
                specs_by_name[name] = merged
            else:
                specs_by_name[name] = dict(item)
            if name not in order:
                order.append(name)
    return [specs_by_name[name] for name in order]


def load_generation_plugins(
    config_path: Optional[str] = None,
    candidate_labels: Optional[Iterable[str]] = None,
) -> List[VisionTaskPlugin]:
    registry = create_default_registry()
    specs = _merge_generation_plugin_specs(config_path)
    labels = [str(label).strip() for label in (candidate_labels or []) if str(label).strip()]
    if labels:
        for spec in specs:
            if spec.get("name") in {"grounding_dino", "sam", "grounded_sam2"}:
                config = dict(spec.get("config") or {})
                config["labels"] = list(dict.fromkeys(labels))
                spec["config"] = config
    return registry.create_from_specs(specs)


def create_default_registry() -> PluginRegistry:
    from .builtin import (
        GroundedSAM2Plugin,
        GroundingDINOPlugin,
        OCRPlugin,
        SAMPlugin,
        TransformersClassificationPlugin,
        UltralyticsPosePlugin,
        UltralyticsTrackingPlugin,
        ViTPosePlugin,
    )

    registry = PluginRegistry()
    registry.register("classification", TransformersClassificationPlugin)
    registry.register("grounding_dino", GroundingDINOPlugin)
    registry.register("grounded_sam2", GroundedSAM2Plugin)
    registry.register("sam", SAMPlugin)
    registry.register("pose", UltralyticsPosePlugin)
    registry.register("vitpose", ViTPosePlugin)
    registry.register("ocr", OCRPlugin)
    registry.register("tracking", UltralyticsTrackingPlugin)
    return registry
