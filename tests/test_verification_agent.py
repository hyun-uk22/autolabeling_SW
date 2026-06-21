import unittest

from src.agents.verification_agent import HierarchicalVerificationAgent
from src.core.models import BoundingBox, DetectionResult


class FakeVisionClient:
    def __init__(self, model_name, results):
        self.model_name = model_name
        self.results = list(results)
        self.api_attempts = 0

    def predict(self, image_path, prompt, temperature=0.0, task_type="object_detection"):
        self.api_attempts += 1
        if self.results:
            result = self.results.pop(0)
        else:
            result = DetectionResult(task_type=task_type)
        result.task_type = task_type
        result.source_model = self.model_name
        return result


def box(label="car", offset=0.0):
    return DetectionResult(
        task_type="object_detection",
        boxes=[
            BoundingBox(
                label=label,
                xmin=0.1 + offset,
                ymin=0.1,
                xmax=0.4 + offset,
                ymax=0.4,
                confidence=0.8,
            )
        ],
    )


class HierarchicalVerificationAgentTests(unittest.TestCase):
    def test_generates_drafts_and_seed_metrics_in_agent(self):
        low = FakeVisionClient("low", [box(), box()])
        high = FakeVisionClient("high", [])
        agent = HierarchicalVerificationAgent(low, high, threshold=0.75, inference_count=2)

        drafts, seed, consistency = agent.generate_draft_labels("image.jpg", "detect cars")

        self.assertEqual(len(drafts), 2)
        self.assertEqual(low.api_attempts, 2)
        self.assertEqual(consistency, 1.0)
        self.assertEqual(seed.source_model, "low")
        self.assertEqual(seed.consistency_score, 1.0)
        self.assertAlmostEqual(seed.mean_confidence, 0.8)
        self.assertAlmostEqual(seed.uncertainty_score, 0.1)

    def test_escalation_reasons_include_consistency_specialists_and_validation(self):
        low = FakeVisionClient("low", [])
        high = FakeVisionClient("high", [])
        agent = HierarchicalVerificationAgent(low, high, threshold=0.75)
        result = DetectionResult(task_type="object_detection", consistency_score=0.5)

        required, reason = agent.needs_escalation(
            result,
            plugin_records=[{"status": "ok", "agreement": 0.4}],
            issues=["box[0]:missing_label"],
        )

        self.assertTrue(required)
        self.assertIn("consistency 0.500 < 0.750", reason)
        self.assertIn("specialist agreement 0.400 < 0.750", reason)
        self.assertIn("validation issues: box[0]:missing_label", reason)

    def test_high_verification_can_preserve_cli_high_model_only_behavior(self):
        seed = box()
        seed.consistency_score = 0.2
        seed.uncertainty_score = 0.5
        low = FakeVisionClient("low", [])
        high = FakeVisionClient("high", [box("person")])
        agent = HierarchicalVerificationAgent(low, high)

        result, agreement = agent.high_verify(
            "image.jpg",
            "detect",
            "object_detection",
            seed,
            merge_with_current=False,
        )

        self.assertEqual(high.api_attempts, 1)
        self.assertEqual(result.source_model, "high")
        self.assertEqual(result.boxes[0].label, "person")
        self.assertEqual(result.consistency_score, 0.2)
        self.assertEqual(result.uncertainty_score, 0.5)
        self.assertEqual(agreement, 0.0)


if __name__ == "__main__":
    unittest.main()
