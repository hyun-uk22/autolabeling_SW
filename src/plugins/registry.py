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
            "model": "openai/clip-vit-base-patch32",
            "device": "cpu",
            "labels": ["person", "animal", "vehicle", "document", "indoor", "outdoor"],
            "top_k": 3,
        },
    },
    {
        "name": "grounding_dino",
        "enabled": True,
        "tasks": ["object_detection", "segmentation", "tracking"],
        "weight": 1.2,
        "config": {
            "model": "IDEA-Research/grounding-dino-tiny",
            "device": "cpu",
            "box_threshold": 0.45,
            "text_threshold": 0.30,
            "merge_iou": 0.35,
            "nms_iou": 0.60,
            "min_confidence": 0.20,
        },
    },
    {
        "name": "sam",
        "enabled": True,
        "tasks": ["segmentation"],
        "weight": 1.0,
        "config": {
            "model": "sam2_b.pt",
            "device": "cpu",
        },
    },
    {
        "name": "pose",
        "enabled": True,
        "tasks": ["pose_estimation"],
        "weight": 1.0,
        "config": {
            "model": "yolo11n-pose.pt",
            "device": "cpu",
            "keypoint_threshold": 0.25,
        },
    },
    {
        "name": "ocr",
        "enabled": True,
        "tasks": ["ocr"],
        "weight": 1.0,
        "config": {
            "languages": ["ko", "en"],
            "gpu": False,
        },
    },
    {
        "name": "tracking",
        "enabled": True,
        "tasks": ["tracking"],
        "weight": 1.0,
        "config": {
            "model": "yolo11n.pt",
            "tracker": "bytetrack.yaml",
            "device": "cpu",
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
            if not item.get("enabled", True):
                continue
            name = item["name"]
            if name in specs_by_name:
                merged = dict(specs_by_name[name])
                merged["config"] = {
                    **dict(merged.get("config") or {}),
                    **dict(item.get("config") or {}),
                }
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


def load_generation_plugins(config_path: Optional[str] = None) -> List[VisionTaskPlugin]:
    registry = create_default_registry()
    return registry.create_from_specs(_merge_generation_plugin_specs(config_path))


def create_default_registry() -> PluginRegistry:
    from .builtin import (
        EasyOCRPlugin,
        GroundingDINOPlugin,
        SAMPlugin,
        TransformersClassificationPlugin,
        UltralyticsPosePlugin,
        UltralyticsTrackingPlugin,
    )

    registry = PluginRegistry()
    registry.register("classification", TransformersClassificationPlugin)
    registry.register("grounding_dino", GroundingDINOPlugin)
    registry.register("sam", SAMPlugin)
    registry.register("pose", UltralyticsPosePlugin)
    registry.register("ocr", EasyOCRPlugin)
    registry.register("tracking", UltralyticsTrackingPlugin)
    return registry
