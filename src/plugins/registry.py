import importlib
import json
from typing import Any, Dict, Iterable, List, Type

from .base import VisionTaskPlugin


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

    def load_config(self, path: str) -> List[VisionTaskPlugin]:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        plugins = []
        for item in data.get("plugins", []):
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

    def names(self) -> Iterable[str]:
        return self._plugins.keys()


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
