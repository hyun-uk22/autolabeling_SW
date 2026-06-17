from typing import List

from ..core.llm_client import VisionLLMClient
from ..core.models import DetectionResult


class LabelingAgent:
    """
    Generates draft labels with repeated low-cost vision model inference.
    """

    def __init__(self, model: VisionLLMClient, inference_count: int = 3, temperature: float = 0.7):
        self.model = model
        self.inference_count = inference_count
        self.temperature = temperature

    def label(self, image_path: str, prompt: str, task_type: str = "object_detection") -> List[DetectionResult]:
        results = []
        for _ in range(self.inference_count):
            result = self.model.predict(image_path, prompt, temperature=self.temperature, task_type=task_type)
            result.source_model = self.model.model_name
            results.append(result)
        return results
