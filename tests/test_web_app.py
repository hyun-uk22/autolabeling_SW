import importlib.util
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


@unittest.skipUnless(importlib.util.find_spec("streamlit"), "Streamlit is not installed")
class StreamlitAppTests(unittest.TestCase):
    def test_polygon_canvas_absolute_points_are_not_shifted_by_left_top(self):
        from src.ui.canvas_geometry import object_polygon_points

        obj = {
            "type": "polygon",
            "left": 100,
            "top": 50,
            "width": 100,
            "height": 100,
            "scaleX": 1,
            "scaleY": 1,
            "pathOffset": {"x": 150, "y": 100},
            "points": [
                {"x": 100, "y": 50},
                {"x": 200, "y": 50},
                {"x": 200, "y": 150},
                {"x": 100, "y": 150},
            ],
        }

        points = object_polygon_points(obj, 400, 300)

        self.assertEqual(
            [(round(point.x, 3), round(point.y, 3)) for point in points],
            [(0.25, 0.167), (0.5, 0.167), (0.5, 0.5), (0.25, 0.5)],
        )

    def test_polygon_canvas_local_points_still_use_left_top_and_path_offset(self):
        from src.ui.canvas_geometry import object_polygon_points

        obj = {
            "type": "polygon",
            "left": 100,
            "top": 50,
            "width": 100,
            "height": 100,
            "scaleX": 1,
            "scaleY": 1,
            "pathOffset": {"x": 50, "y": 50},
            "points": [
                {"x": 0, "y": 0},
                {"x": 100, "y": 0},
                {"x": 100, "y": 100},
                {"x": 0, "y": 100},
            ],
        }

        points = object_polygon_points(obj, 400, 300)

        self.assertEqual(
            [(round(point.x, 3), round(point.y, 3)) for point in points],
            [(0.125, 0.0), (0.375, 0.0), (0.375, 0.333), (0.125, 0.333)],
        )

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
            self.assertEqual(len(app.tabs), 6)
            self.assertEqual(len(app.chat_input), 1)
            label_path = next(item for item in app.selectbox if item.label == "라벨 폴더 경로")
            image_path = next(item for item in app.selectbox if item.label == "이미지 폴더 경로")
            self.assertEqual(label_path.value, "data/labeled")
            self.assertEqual(image_path.value, "data/raw")
            self.assertIn("data/labeled", label_path.options)
            self.assertIn("data/raw", image_path.options)

    def test_app_renders_all_workflow_tabs_without_exceptions(self):
        from streamlit.testing.v1 import AppTest

        with tempfile.TemporaryDirectory() as workspace:
            app = AppTest.from_file("web_app.py")
            app.session_state["workspace"] = workspace
            app.run(timeout=30)

            self.assertEqual(len(app.exception), 0)
            self.assertEqual(
                [tab.label for tab in app.tabs],
                ["대화형 작업", "형식 변환", "라벨 생성", "라벨 편집", "결과 리포트", "설정"],
            )
            self.assertEqual(len(app.get("form")), 3)
            self.assertEqual(len(app.chat_input), 1)
            self.assertIn("어떤 작업을 진행할까요?", app.chat_message[0].markdown[0].value)
            self.assertNotIn("`", app.chat_message[0].markdown[0].value)
            prompt = next(area for area in app.text_area if area.label == "프롬프트")
            self.assertEqual(prompt.value, "")
            self.assertEqual(
                prompt.placeholder,
                "이미지에서 눈에 띄는 모든 객체를 탐지하고 분류해 주세요.",
            )

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
            self.assertTrue(any("실행 전 확인 사항" in item.value for item in app.markdown))
            self.assertGreaterEqual(len(app.dataframe), 1)

    def test_korean_generation_prompt_is_forwarded_to_workflow(self):
        from streamlit.testing.v1 import AppTest

        result = {
            "status": "completed",
            "operation_outputs": [],
            "errors": [],
            "history_path": "",
        }
        with tempfile.TemporaryDirectory() as workspace, patch(
            "src.workflow.service.execute_workflow_plan",
            return_value=result,
        ) as execute_workflow:
            app = AppTest.from_file("web_app.py")
            app.session_state["workspace"] = workspace
            app.run(timeout=30)

            korean_prompt = "이미지에서 차량과 보행자를 찾아 객체 탐지 라벨을 생성해줘."
            next(area for area in app.text_area if area.label == "프롬프트").set_value(korean_prompt)
            next(box for box in app.checkbox if box.label == "고비용 모델 API 호출 승인").set_value(True)
            next(button for button in app.button if button.label == "라벨 생성 실행").click()
            app.run(timeout=30)

            self.assertEqual(len(app.exception), 0)
            execute_workflow.assert_called_once()
            plan = execute_workflow.call_args.args[0]
            self.assertEqual(plan["operations"][0]["prompt"], korean_prompt)
            self.assertEqual(plan["operations"][0]["insight_imbalance_ratio"], 3.0)

    def test_conversion_report_renders_metrics_actions_insight_and_download(self):
        from streamlit.testing.v1 import AppTest

        with tempfile.TemporaryDirectory() as workspace:
            report_path = Path(workspace) / "conversion_report.json"
            report_path.write_text('{"status": "partial_success"}', encoding="utf-8")
            output = {
                "action": "convert",
                "records_read": 2,
                "records_converted": 1,
                "validation": {"failed_records": 1},
                "preflight": {
                    "status": "needs_attention",
                    "notices": [{
                        "severity": "warning",
                        "code": "missing_yolo_class_mapping",
                        "message": "YOLO class mapping 파일을 찾지 못했습니다.",
                        "user_action": "data.yaml 또는 classes.txt를 지정하세요.",
                    }],
                },
                "export_validation": {"failed_records": 0, "artifact_issues": []},
                "user_action_report": {
                    "completion_rate": 50.0,
                    "summary": {
                        "clean": 1,
                        "needs_review": 1,
                        "artifact_issues": 0,
                    },
                    "recommended_actions": ["좌표 범위를 검토하세요."],
                    "detailed_records": [{
                        "image": "bad.jpg",
                        "status": "needs_attention",
                        "total_issues": 1,
                        "priority_actions": ["좌표를 수정하세요."],
                    }],
                },
                "dataset_insight": {
                    "distribution": {"car": {"count": 3, "percentage": 100.0}},
                    "suggestions": [],
                },
                "report_path": str(report_path),
                "artifacts": {},
            }
            result = {
                "status": "completed",
                "operation_outputs": [output],
                "errors": [],
                "history_path": "",
            }
            with patch("src.workflow.service.execute_workflow_plan", return_value=result):
                app = AppTest.from_file("web_app.py")
                app.session_state["workspace"] = workspace
                app.run(timeout=30)
                next(button for button in app.button if button.label == "변환 실행").click()
                app.run(timeout=30)

            self.assertEqual(len(app.exception), 0)
            metric_labels = [metric.label for metric in app.metric]
            self.assertIn("읽은 데이터", metric_labels)
            self.assertIn("완료율", metric_labels)
            self.assertGreaterEqual(len(app.dataframe), 2)
            self.assertFalse(any("변환 전 확인 사항" in item.value for item in app.markdown))
            self.assertGreaterEqual(len(app.get("download_button")), 1)

            app.run(timeout=30)
            self.assertEqual(len(app.exception), 0)
            self.assertIn("읽은 데이터", [metric.label for metric in app.metric])
            self.assertGreaterEqual(len(app.get("download_button")), 1)


if __name__ == "__main__":
    unittest.main()
