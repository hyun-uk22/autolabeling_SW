import unittest

from src.core.models import BoundingBox, DetectionResult
from src.plugins.base import PluginOutput, VisionTaskPlugin
from src.plugins.orchestrator import TaskPluginOrchestrator
from src.workflow.models import OperationPlan
from src.workflow.runtime import WorkflowRuntime


class ThresholdBoxPlugin(VisionTaskPlugin):
    plugin_name = "grounding_dino"
    supported_tasks = {"object_detection"}

    def refine(self, image_path, prompt, seed_result):
        threshold = float(self.config.get("box_threshold", 0.45))
        result = DetectionResult(task_type="object_detection")
        result.boxes.append(
            BoundingBox(
                label="person",
                xmin=0.1 + threshold / 10,
                ymin=0.1,
                xmax=0.4 + threshold / 10,
                ymax=0.4,
                confidence=threshold,
            )
        )
        return PluginOutput(result=result, score=threshold, metadata={"threshold": threshold})


class SpecialistConsistencyTests(unittest.TestCase):
    def test_plugin_config_overrides_are_temporary(self):
        plugin = ThresholdBoxPlugin({"box_threshold": 0.45, "labels": ["person"]})
        orchestrator = TaskPluginOrchestrator([plugin])

        base, _ = orchestrator.process(
            "image.jpg",
            "detect person",
            "object_detection",
            DetectionResult(task_type="object_detection"),
        )
        rerun, _ = orchestrator.process(
            "image.jpg",
            "detect person",
            "object_detection",
            DetectionResult(task_type="object_detection"),
            config_overrides={"grounding_dino": {"box_threshold": 0.30}},
        )
        after, _ = orchestrator.process(
            "image.jpg",
            "detect person",
            "object_detection",
            DetectionResult(task_type="object_detection"),
        )

        self.assertNotEqual(base.boxes[0].xmin, rerun.boxes[0].xmin)
        self.assertEqual(base.boxes[0].xmin, after.boxes[0].xmin)
        self.assertEqual(plugin.config["box_threshold"], 0.45)

    def test_operation_plan_has_report_only_specialist_consistency_defaults(self):
        operation = OperationPlan(action="generate")

        self.assertEqual(operation.specialist_consistency_runs, 0)
        self.assertEqual(operation.specialist_advisor_mode, "none")

    def test_advisor_prompt_includes_first_pass_report_without_allowing_label_edits(self):
        runtime = WorkflowRuntime()
        operation = OperationPlan(action="generate", task_type="object_detection")
        result = DetectionResult(task_type="object_detection")
        result.boxes.append(
            BoundingBox(
                label="person",
                xmin=0.1,
                ymin=0.1,
                xmax=0.4,
                ymax=0.4,
                confidence=0.42,
            )
        )
        report = {
            "total_labels": 1,
            "class_counts": {"person": 1},
            "confidence": {"mean": 0.42, "low_confidence_count": 1},
            "current_parameters": {"grounding_dino": {"box_threshold": 0.45}},
        }

        prompt = runtime._advisor_prompt(operation, result, ["person"], report)

        self.assertIn("first-pass specialist report", prompt)
        self.assertIn("\"low_confidence_count\": 1", prompt)
        self.assertIn("Do not create labels or boxes", prompt)
        self.assertIn("Do not change, add, remove, translate, or synonymize classes", prompt)


if __name__ == "__main__":
    unittest.main()
