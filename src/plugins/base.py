from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Optional, Set

from ..core.models import DetectionResult


@dataclass
class PluginOutput:
    result: DetectionResult
    score: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class VisionTaskPlugin(ABC):
    plugin_name = "vision_plugin"
    supported_tasks: Set[str] = set()

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        configured_tasks = self.config.get("tasks")
        if configured_tasks:
            self.supported_tasks = set(configured_tasks)

    def supports(self, task_type: str) -> bool:
        return task_type == "all" or task_type in self.supported_tasks

    @abstractmethod
    def refine(
        self,
        image_path: str,
        prompt: str,
        seed_result: DetectionResult,
    ) -> PluginOutput:
        raise NotImplementedError


def configured_labels(config: Dict[str, Any], seed_result: DetectionResult) -> Iterable[str]:
    labels = list(config.get("labels") or [])
    labels.extend(item.label for item in seed_result.classifications)
    labels.extend(item.label for item in seed_result.boxes)
    labels.extend(item.label for item in seed_result.segments)
    labels.extend(item.label for item in seed_result.poses)
    return list(dict.fromkeys(label for label in labels if label))
