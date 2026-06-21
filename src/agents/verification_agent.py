from typing import List, Optional, Tuple
from ..core.llm_client import VisionLLMClient
from ..core.models import DetectionResult
from ..plugins.orchestrator import merge_results
from ..utils.geometry import compute_result_consistency, get_consistency_score
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

    def generate_draft_labels(
        self,
        image_path: str,
        prompt: str,
        task_type: str = "object_detection",
    ) -> Tuple[List[DetectionResult], DetectionResult, float]:
        """Run repeated low-capacity inference and return the seed label result."""
        drafts = self.labeling_agent.label(image_path, prompt, task_type=task_type)
        consistency = float(get_consistency_score(drafts))
        seed = drafts[0] if drafts else DetectionResult(task_type=task_type)
        seed.consistency_score = consistency
        seed.mean_confidence = self._mean_confidence(seed)
        seed.uncertainty_score = self._uncertainty_score(consistency, seed.mean_confidence)
        return drafts, seed, consistency

    def needs_escalation(
        self,
        result: DetectionResult,
        threshold: Optional[float] = None,
        plugin_records: Optional[List[dict]] = None,
        issues: Optional[List[str]] = None,
    ) -> Tuple[bool, str]:
        """Decide whether a sample should be escalated to the high-capacity LMM."""
        threshold = self.threshold if threshold is None else threshold
        reasons = []
        if (result.consistency_score or 0.0) < threshold:
            reasons.append(f"consistency {result.consistency_score or 0.0:.3f} < {threshold:.3f}")

        agreements = [
            float(record["agreement"])
            for record in (plugin_records or [])
            if record.get("status") == "ok" and record.get("agreement") is not None
        ]
        if agreements and min(agreements) < threshold:
            reasons.append(f"specialist agreement {min(agreements):.3f} < {threshold:.3f}")
        if issues:
            reasons.append(f"validation issues: {', '.join(issues[:3])}")
        return bool(reasons), "; ".join(reasons)

    def high_verify(
        self,
        image_path: str,
        prompt: str,
        task_type: str,
        specialist_result: DetectionResult,
        merge_with_current: bool = True,
    ) -> Tuple[DetectionResult, float]:
        """Use the high-capacity LMM and merge it with the current specialist result."""
        high_result = self.high_model.predict(
            image_path,
            prompt,
            temperature=0.0,
            task_type=task_type,
        )
        high_result.source_model = self.high_model.model_name
        agreement = compute_result_consistency(high_result, specialist_result)
        if not merge_with_current:
            high_result.consistency_score = specialist_result.consistency_score
            high_result.mean_confidence = self._mean_confidence(high_result)
            high_result.uncertainty_score = specialist_result.uncertainty_score
            return high_result, agreement

        merged = merge_results(high_result, specialist_result)
        merged.plugin_scores.update(specialist_result.plugin_scores)
        merged.plugin_metadata.update(specialist_result.plugin_metadata)
        previous = specialist_result.consistency_score if specialist_result.consistency_score is not None else agreement
        merged.consistency_score = (previous + agreement) / 2
        merged.mean_confidence = self._mean_confidence(merged)
        merged.uncertainty_score = self._uncertainty_score(merged.consistency_score, merged.mean_confidence)
        if specialist_result.plugin_scores:
            merged.source_model = f"{self.high_model.model_name}+{'+'.join(specialist_result.plugin_scores)}"
        return merged, agreement

    def process(self, image_path: str, prompt: str, task_type: str = "object_detection") -> Tuple[DetectionResult, str]:
        # 1. Labeling Agent: repeated low-cost inference for draft labels.
        results, seed, consistency = self.generate_draft_labels(image_path, prompt, task_type=task_type)
        
        # 2. Decision Logic
        required, _ = self.needs_escalation(seed)
        if required:
            print(f"[*] Low consistency ({consistency:.2f}). Escalating to High-Level Model...")
            final_res, _ = self.high_verify(image_path, prompt, task_type, seed, merge_with_current=False)
            return final_res, "Escalated"
        else:
            final_res = results[0] if results else seed # Use the first result if consistent
            final_res.source_model = self.low_model.model_name
            final_res.consistency_score = consistency
            final_res.mean_confidence = seed.mean_confidence
            final_res.uncertainty_score = seed.uncertainty_score
            return final_res, "Consistent"
