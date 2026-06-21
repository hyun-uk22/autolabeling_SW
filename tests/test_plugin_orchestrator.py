import unittest

from src.core.models import BoundingBox, DetectionResult
from src.plugins.base import PluginOutput, VisionTaskPlugin
from src.plugins.orchestrator import TaskPluginOrchestrator, merge_results


class FakeDetectionPlugin(VisionTaskPlugin):
    plugin_name = "fake_detection"
    supported_tasks = {"object_detection"}

    def __init__(self, config=None):
        super().__init__(config)
        self.loaded = False

    def _load(self):
        self.loaded = True

    def refine(self, image_path, prompt, seed_result):
        return PluginOutput(result=DetectionResult(task_type="object_detection"), score=1.0)


class FailingPreparePlugin(VisionTaskPlugin):
    plugin_name = "failing"
    supported_tasks = {"object_detection"}

    def _load(self):
        raise RuntimeError("missing dependency")

    def refine(self, image_path, prompt, seed_result):
        return PluginOutput(result=DetectionResult(task_type="object_detection"), score=0.0)


class PluginOrchestratorTests(unittest.TestCase):
    def test_prepare_loads_supported_plugins_before_refine(self):
        plugin = FakeDetectionPlugin()
        orchestrator = TaskPluginOrchestrator([plugin])

        records = orchestrator.prepare("object_detection")

        self.assertTrue(plugin.loaded)
        self.assertEqual(records, [{"plugin": "fake_detection", "status": "ok"}])

    def test_prepare_records_errors_without_fail_fast(self):
        orchestrator = TaskPluginOrchestrator([FailingPreparePlugin()])

        records = orchestrator.prepare("object_detection")

        self.assertEqual(records[0]["plugin"], "failing")
        self.assertEqual(records[0]["status"], "error")
        self.assertIn("missing dependency", records[0]["error"])

    def test_prepare_skips_unsupported_task(self):
        plugin = FakeDetectionPlugin()
        orchestrator = TaskPluginOrchestrator([plugin])

        records = orchestrator.prepare("ocr")

        self.assertFalse(plugin.loaded)
        self.assertEqual(records[0]["status"], "skipped")

    def test_merge_results_deduplicates_overlapping_boxes(self):
        base = DetectionResult(
            task_type="object_detection",
            boxes=[BoundingBox(label="car", xmin=0.10, ymin=0.10, xmax=0.50, ymax=0.50, confidence=0.75)],
        )
        incoming = DetectionResult(
            task_type="object_detection",
            boxes=[BoundingBox(label="car", xmin=0.12, ymin=0.12, xmax=0.52, ymax=0.52, confidence=0.90)],
        )

        merged = merge_results(base, incoming, match_iou=0.35, nms_iou=0.60)

        self.assertEqual(len(merged.boxes), 1)
        self.assertEqual(merged.boxes[0].label, "car")
        self.assertGreaterEqual(merged.boxes[0].confidence, 0.8)

    def test_merge_results_filters_low_confidence_boxes(self):
        base = DetectionResult(task_type="object_detection")
        incoming = DetectionResult(
            task_type="object_detection",
            boxes=[BoundingBox(label="car", xmin=0.10, ymin=0.10, xmax=0.50, ymax=0.50, confidence=0.10)],
        )

        merged = merge_results(base, incoming, min_confidence=0.20)

        self.assertEqual(merged.boxes, [])


if __name__ == "__main__":
    unittest.main()
