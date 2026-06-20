import importlib.util
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


@unittest.skipUnless(importlib.util.find_spec("streamlit"), "Streamlit is not installed")
class StreamlitAppTests(unittest.TestCase):
    def test_initial_screen_requires_workspace_selection(self):
        from streamlit.testing.v1 import AppTest

        with tempfile.TemporaryDirectory() as appdata, tempfile.TemporaryDirectory() as workspace, patch.dict(os.environ, {"APPDATA": appdata}):
            app = AppTest.from_file("web_app.py").run(timeout=30)

            self.assertEqual(len(app.exception), 0)
            self.assertEqual(len(app.tabs), 0)
            self.assertEqual(app.text_input[0].label, "Workspace 경로")
            self.assertEqual(app.button[0].label, "Workspace 적용")

            app.text_input[0].set_value(workspace)
            app.button[0].click()
            app.run(timeout=30)

            self.assertEqual(len(app.exception), 0)
            self.assertEqual(len(app.tabs), 4)
            self.assertEqual(len(app.chat_input), 1)
            self.assertEqual(app.text_input[0].value, "data/labeled")
            self.assertEqual(app.text_input[1].value, "data/raw")

    def test_app_renders_all_workflow_tabs_without_exceptions(self):
        from streamlit.testing.v1 import AppTest

        with tempfile.TemporaryDirectory() as workspace:
            app = AppTest.from_file("web_app.py")
            app.session_state["workspace"] = workspace
            app.run(timeout=30)

            self.assertEqual(len(app.exception), 0)
            self.assertEqual(
                [tab.label for tab in app.tabs],
                ["형식 변환", "라벨 생성", "평가", "설정"],
            )
            self.assertEqual(len(app.get("form")), 4)
            self.assertEqual(len(app.chat_input), 1)
            self.assertIn("어떤 작업을 진행할까요?", app.chat_message[0].markdown[0].value)

    def test_chat_request_discovers_dataset_and_shows_execution_plan(self):
        from streamlit.testing.v1 import AppTest

        with tempfile.TemporaryDirectory() as workspace:
            root = Path(workspace)
            (root / "data" / "raw").mkdir(parents=True)
            (root / "data" / "labeled").mkdir(parents=True)
            (root / "data" / "raw" / "sample.jpg").write_bytes(b"discovery-only")
            (root / "data" / "labeled" / "sample.txt").write_text(
                "0 0.5 0.5 0.2 0.2\n",
                encoding="utf-8",
            )
            app = AppTest.from_file("web_app.py")
            app.session_state["workspace"] = workspace
            app.run(timeout=30)

            app.chat_input[0].set_value(
                "현재 데이터셋의 라벨링 형식을 MS COCO 형식으로 바꿔줘"
            )
            app.run(timeout=30)

            self.assertEqual(len(app.exception), 0)
            self.assertEqual(app.button[0].label, "계획 실행")
            self.assertIn("data/labeled", app.chat_message[-1].markdown[0].value)
            self.assertIn("coco", app.chat_message[-1].markdown[0].value)


if __name__ == "__main__":
    unittest.main()
