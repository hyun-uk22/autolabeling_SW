import unittest

from src.core.models import DetectionResult
from src.plugins.base import PluginOutput, VisionTaskPlugin
from src.plugins.orchestrator import TaskPluginOrchestrator


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


if __name__ == "__main__":
    unittest.main()
