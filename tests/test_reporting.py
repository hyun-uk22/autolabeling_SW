import json
import tempfile
import unittest
from pathlib import Path

from src.core.models import BoundingBox, DetectionResult, TextRegion
from src.agents.insight_agent import DatasetInsightAgent
from src.reporting import (
    ArtifactAuditor,
    build_conversion_preflight,
    build_generation_performance,
    build_user_action_report,
)
from src.reporting.issue_reporter import categorize_issue
from src.utils.format_converter import LabelExportWriter, resolve_export_formats
from src.workflow.models import OperationPlan
from src.workflow.runtime import WorkflowRuntime


class ReportingTests(unittest.TestCase):
    def test_prefixed_validator_and_export_issues_are_categorized(self):
        missing_label = categorize_issue("box[0]:missing_label")
        empty_coco = categorize_issue("coco:no_annotations:D:/output.json")

        self.assertEqual(missing_label["code"], "missing_label")
        self.assertEqual(missing_label["severity"], "high")
        self.assertEqual(empty_coco["code"], "no_annotations")
        self.assertEqual(empty_coco["category"], "output_format")
        self.assertEqual(empty_coco["affected_path"], "D:/output.json")
        self.assertIn("D:/output.json", empty_coco["fix_instruction"])
        self.assertTrue(empty_coco["suggestions"])

    def test_user_action_report_prioritizes_specific_fix_instructions(self):
        report = build_user_action_report(
            [{"image": "missing.jpg", "issues": ["missing_image:D:/images/missing.jpg"]}],
            total_records=1,
        )

        record = report["detailed_records"][0]

        self.assertEqual(report["status"], "needs_review")
        self.assertEqual(record["status"], "blocked")
        self.assertIn("D:/images/missing.jpg", record["priority_actions"][0])
        self.assertEqual(
            record["detailed_issues"][0]["affected_path"],
            "D:/images/missing.jpg",
        )

    def test_completion_rate_uses_all_processed_records(self):
        report = build_user_action_report(
            [
                {"image": "clean.jpg", "issues": []},
                {"image": "bad.jpg", "issues": ["box[0]:missing_label"]},
            ],
            total_records=2,
        )

        self.assertEqual(report["status"], "partial_success")
        self.assertEqual(report["summary"]["clean"], 1)
        self.assertEqual(report["summary"]["needs_review"], 1)
        self.assertEqual(report["completion_rate"], 50.0)

    def test_artifact_auditor_checks_generated_label_contents(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            empty_yolo = root / "empty.txt"
            empty_yolo.write_text("", encoding="utf-8")
            coco = root / "coco.json"
            coco.write_text(json.dumps({
                "images": [{"id": 1}],
                "annotations": [],
                "categories": [],
            }), encoding="utf-8")
            auditor = ArtifactAuditor()

            self.assertIn("empty_output_file", auditor.audit_record({"yolo": str(empty_yolo)})[0])
            self.assertIn("no_annotations", auditor.audit_artifacts({"coco": str(coco)})[0])

    def test_dataset_insight_reports_distribution_and_imbalance(self):
        results = [
            DetectionResult(boxes=[BoundingBox(
                label="car", xmin=0.1, ymin=0.1, xmax=0.2, ymax=0.2,
            )])
            for _ in range(4)
        ]
        results.append(DetectionResult(boxes=[BoundingBox(
            label="person", xmin=0.1, ymin=0.1, xmax=0.2, ymax=0.2,
        )]))

        insight = DatasetInsightAgent().analyze(results)

        self.assertEqual(insight["total_labels"], 5)
        self.assertEqual(insight["distribution"]["car"]["count"], 4)
        self.assertTrue(insight["imbalance"]["detected"])
        self.assertTrue(insight["suggestions"])
        self.assertEqual(insight["agent"], "DatasetInsightAgent")

        relaxed = DatasetInsightAgent(imbalance_ratio_threshold=5.0).analyze(results)
        self.assertEqual(relaxed["status"], "balanced")
        self.assertFalse(relaxed["imbalance"]["detected"])

        operation = OperationPlan(action="convert", insight_imbalance_ratio=5.0)
        runtime_insight = WorkflowRuntime.analyze_dataset(operation, results)
        self.assertEqual(runtime_insight["status"], "balanced")

    def test_dataset_insight_agent_keeps_cli_accumulator_compatibility(self):
        agent = DatasetInsightAgent()
        agent.add_result(DetectionResult(boxes=[BoundingBox(
            label="car", xmin=0.1, ymin=0.1, xmax=0.2, ymax=0.2,
        )]))

        insight = agent.analyze()

        self.assertEqual(insight["distribution"]["car"]["count"], 1)
        self.assertIn("car", agent.get_report())
        agent.reset()
        self.assertEqual(agent.analyze()["status"], "empty")

    def test_performance_report_marks_estimated_values(self):
        performance = build_generation_performance(2, 10.0, 6, 1, 1)

        self.assertEqual(performance["avg_elapsed_sec"], 5.0)
        self.assertEqual(performance["estimated_manual_time_sec"], 90.0)
        self.assertEqual(performance["escalation_rate"], 0.5)
        self.assertIn("추정치", performance["estimation_notice"])

    def test_conversion_preflight_reports_missing_yolo_mapping_and_images(self):
        preflight = build_conversion_preflight(
            {
                "sources_discovered": 1,
                "records_after_merge": 1,
                "processed_files": [{
                    "path": "labels/sample.txt",
                    "format": "yolo",
                    "records": 1,
                    "class_mapping": {
                        "status": "missing",
                        "searched": ["labels/data.yaml", "labels/classes.txt"],
                    },
                }],
                "failed_files": [],
                "skipped_files": [],
                "merge": {"conflicts": []},
                "class_list": ["3"],
            },
            ["yolo"],
            [{"image": "sample.jpg", "issues": ["missing_image:D:/sample.jpg"]}],
        )

        self.assertEqual(preflight["status"], "needs_attention")
        codes = [notice["code"] for notice in preflight["notices"]]
        self.assertIn("missing_yolo_class_mapping", codes)
        self.assertIn("missing_images", codes)
        missing_images = next(notice for notice in preflight["notices"] if notice["code"] == "missing_images")
        self.assertEqual(missing_images["severity"], "warning")

    def test_conversion_preflight_reports_numeric_label_normalization(self):
        preflight = build_conversion_preflight(
            {
                "sources_discovered": 2,
                "records_after_merge": 1,
                "processed_files": [],
                "failed_files": [],
                "skipped_files": [],
                "merge": {
                    "conflicts": [],
                    "label_normalizations": [{
                        "numeric_label": "3",
                        "canonical_label": "Lion",
                    }],
                },
                "class_list": ["Lion"],
            },
            ["coco"],
            [],
        )

        codes = [notice["code"] for notice in preflight["notices"]]
        self.assertIn("numeric_label_normalized", codes)

    def test_conversion_preflight_summarizes_detailed_validation_issues(self):
        preflight = build_conversion_preflight(
            {
                "sources_discovered": 1,
                "records_after_merge": 1,
                "processed_files": [],
                "failed_files": [],
                "skipped_files": [],
                "merge": {"conflicts": []},
                "class_list": ["car"],
            },
            ["coco"],
            [{
                "image": "bad.jpg",
                "issues": [
                    "box[0]:invalid_box_order",
                    "box[1]:coordinate_out_of_range",
                    "classification[0]:confidence_out_of_range",
                    "segment[0]:too_few_points",
                    "pose[0]:empty_keypoints",
                    "pose[0].keypoint[0]:missing_name",
                    "text[0]:missing_text",
                    "track[0]:missing_track_id",
                    "empty_result",
                ],
            }],
        )

        notices = {notice["code"]: notice for notice in preflight["notices"]}
        for code in (
            "invalid_box_order",
            "coordinate_out_of_range",
            "confidence_out_of_range",
            "too_few_points",
            "empty_keypoints",
            "missing_name",
            "missing_text",
            "missing_track_id",
            "empty_result",
        ):
            self.assertIn(code, notices)
            self.assertEqual(notices[code]["severity"], "warning")
            self.assertEqual(notices[code]["count"], 1)

        self.assertEqual(preflight["status"], "needs_attention")

    def test_export_format_resolution_falls_back_to_vision_json_for_non_box_labels(self):
        result = DetectionResult(
            task_type="ocr",
            texts=[TextRegion(text="hello", xmin=0.1, ymin=0.1, xmax=0.4, ymax=0.2)],
        )

        formats = resolve_export_formats(result, ["yolo", "pascal_voc"], "ocr")

        self.assertEqual(formats, ["vision_json"])

    def test_writer_records_fallback_vision_json_artifact(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image_path = root / "image.jpg"
            from PIL import Image

            Image.new("RGB", (20, 20), color="white").save(image_path)
            result = DetectionResult(
                task_type="ocr",
                texts=[TextRegion(text="hello", xmin=0.1, ymin=0.1, xmax=0.4, ymax=0.2)],
            )
            writer = LabelExportWriter(str(root / "labels"), formats=["yolo"])

            resolved = resolve_export_formats(result, writer.formats, result.task_type)
            paths = writer.save(result, str(image_path), formats=resolved)
            artifacts = writer.finalize()

            self.assertNotIn("yolo", paths)
            self.assertIn("vision_json", paths)
            self.assertIn("vision_json", artifacts)
            self.assertTrue(Path(artifacts["vision_json"]).exists())


if __name__ == "__main__":
    unittest.main()
