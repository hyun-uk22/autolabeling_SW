import importlib.util
import os
import tempfile
import unittest
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


if __name__ == "__main__":
    unittest.main()
