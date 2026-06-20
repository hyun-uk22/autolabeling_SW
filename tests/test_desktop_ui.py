import os
import tempfile
import unittest
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

from src.ui.main_window import ConvertPage, EvaluatePage, GeneratePage, MainWindow


class DesktopUiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.workspace = self.temp_dir.name

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_main_window_contains_all_workflow_pages(self):
        window = MainWindow(self.workspace)
        self.assertEqual(window.navigation.count(), 4)
        self.assertEqual(window.pages.count(), 4)
        window.close()

    def test_convert_page_builds_mixed_format_plan(self):
        page = ConvertPage(self.workspace)
        page.input_path.setText("D:/dataset/labels")
        page.image_dir.setText("D:/dataset/images")
        page.output_dir.setText("D:/dataset/converted")

        plan, expensive = page.build_plan()
        operation = plan["operations"][0]

        self.assertFalse(expensive)
        self.assertEqual(operation["action"], "convert")
        self.assertEqual(operation["source_format"], "auto")
        self.assertEqual(operation["duplicate_iou"], 0.85)

    def test_generate_and_evaluate_pages_build_plans(self):
        generation = GeneratePage(self.workspace)
        generation.image_dir.setText("D:/dataset/images")
        generation.output_dir.setText("D:/dataset/labels")
        generation.visual_dir.setText("D:/dataset/visualized")
        generation_plan, expensive = generation.build_plan()
        self.assertTrue(expensive)
        self.assertEqual(generation_plan["operations"][0]["action"], "generate")
        self.assertEqual(
            generation_plan["operations"][0]["plugin_config"],
            str((Path(self.workspace) / "configs" / "plugins.json").resolve()),
        )

        evaluation = EvaluatePage(self.workspace)
        evaluation.output_dir.setText("D:/dataset/reports")
        evaluation.runs.setPlainText("baseline=D:/runs/baseline")
        evaluation_plan, expensive = evaluation.build_plan()
        self.assertFalse(expensive)
        self.assertEqual(
            evaluation_plan["operations"][0]["runs"]["baseline"],
            str(Path("D:/runs/baseline").resolve()),
        )


if __name__ == "__main__":
    unittest.main()
