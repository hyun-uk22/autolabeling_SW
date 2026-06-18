from typing import Tuple
from ..core.llm_client import VisionLLMClient
from ..core.models import DetectionResult
from ..utils.geometry import get_consistency_score
from ..utils.result_metrics import mean_result_confidence, uncertainty_score
from .labeling_agent import LabelingAgent

class HierarchicalVerificationAgent:
    def __init__(
        self,
        low_model: VisionLLMClient,
        high_model: VisionLLMClient,
        threshold: float = 0.7,
        inference_count: int = 3,
        draft_temperature: float = 0.7,
    ):
        self.low_model = low_model
        self.high_model = high_model
        self.threshold = threshold
        self.labeling_agent = LabelingAgent(
            low_model,
            inference_count=inference_count,
            temperature=draft_temperature,
        )

    def _mean_confidence(self, result: DetectionResult) -> float:
        return mean_result_confidence(result)

    def _uncertainty_score(self, consistency: float, confidence: float) -> float:
        # Proposal metric: combine repeated-inference IoU consistency and model confidence.
        return uncertainty_score(consistency, confidence)

    def process(self, image_path: str, prompt: str, task_type: str = "object_detection") -> Tuple[DetectionResult, str]:
        # 1. Labeling Agent: repeated low-cost inference for draft labels.
        results = self.labeling_agent.label(image_path, prompt, task_type=task_type)
        
        # 2. Check Consistency (Uncertainty measurement)
        consistency = get_consistency_score(results)
        draft_confidence = self._mean_confidence(results[0]) if results else 0.0
        uncertainty = self._uncertainty_score(consistency, draft_confidence)
        
        # 3. Decision Logic
        if consistency < self.threshold:
            print(f"[*] Low consistency ({consistency:.2f}). Escalating to High-Level Model...")
            final_res = self.high_model.predict(image_path, prompt, temperature=0.0, task_type=task_type)
            final_res.source_model = self.high_model.model_name
            final_res.consistency_score = consistency
            final_res.mean_confidence = self._mean_confidence(final_res)
            final_res.uncertainty_score = uncertainty
            return final_res, "Escalated"
        else:
            final_res = results[0] # Use the first result if consistent
            final_res.source_model = self.low_model.model_name
            final_res.consistency_score = consistency
            final_res.mean_confidence = draft_confidence
            final_res.uncertainty_score = uncertainty
            return final_res, "Consistent"
