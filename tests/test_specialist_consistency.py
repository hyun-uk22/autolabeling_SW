import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from src.core.models import BoundingBox, DetectionResult, Point, PolygonSegment
from src.plugins.base import PluginOutput, VisionTaskPlugin
from src.plugins.orchestrator import TaskPluginOrchestrator
from src.workflow.models import OperationPlan
from src.workflow.runtime import WorkflowRuntime, _llm_review_queue


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


class FakeLLMClient:
    def __init__(self, model_name, result):
        self.model_name = model_name
        self.result = result
        self.api_attempts = 0

    def predict(self, image_path, prompt, temperature=0.0, task_type="object_detection"):
        self.api_attempts += 1
        result = self.result.model_copy(deep=True)
        result.task_type = task_type
        return result


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

    def test_prepare_generation_discovers_nested_images(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image_root = root / "datasets" / "ocr"
            (image_root / "csv" / "data").mkdir(parents=True)
            (image_root / "csv" / "data" / "a.jpg").write_bytes(b"image")
            (image_root / "extra" / "b.png").parent.mkdir(parents=True)
            (image_root / "extra" / "b.png").write_bytes(b"image")
            operation = OperationPlan(
                action="generate",
                img_dir=str(image_root),
                out_dir=str(root / "out"),
                vis_dir=str(root / "vis"),
            )

            with patch("src.workflow.runtime.load_generation_plugins", return_value=[]):
                prepared = WorkflowRuntime().prepare_generation(operation)

            self.assertEqual(
                prepared["images"],
                [str(Path("csv") / "data" / "a.jpg"), str(Path("extra") / "b.png")],
            )

    def test_llm_consistency_compares_vision_result_to_selected_llm_result(self):
        runtime = WorkflowRuntime()
        operation = OperationPlan(
            action="generate",
            task_type="object_detection",
            llm_consistency_mode="low",
            low_model="low",
            high_model="high",
            threshold=0.75,
        )
        base = DetectionResult(task_type="object_detection")
        base.boxes.append(BoundingBox(label="person", xmin=0.1, ymin=0.1, xmax=0.4, ymax=0.4, confidence=0.9))
        llm = DetectionResult(task_type="object_detection")
        llm.boxes.append(BoundingBox(label="person", xmin=0.12, ymin=0.1, xmax=0.42, ymax=0.4, confidence=0.8))
        runtime.low_client = FakeLLMClient("low", llm)
        runtime.high_client = FakeLLMClient("high", DetectionResult(task_type="object_detection"))

        report = runtime._llm_consistency_rerun("image.jpg", operation, base)

        self.assertTrue(report["enabled"])
        self.assertEqual(report["mode"], "low")
        self.assertEqual(report["comparisons"][0]["level"], "low")
        self.assertGreater(report["comparisons"][0]["bbox_agreement"]["agreement"], 0.0)
        self.assertFalse(report["review_required"])

    def test_low_only_llm_consistency_does_not_validate_unused_high_model(self):
        runtime = WorkflowRuntime()
        model_name = "bedrock:anthropic.claude-sonnet-test"
        operation = OperationPlan(
            action="generate",
            task_type="object_detection",
            llm_consistency_mode="low",
            low_model=model_name,
            high_model=model_name,
            threshold=0.75,
        )
        base = DetectionResult(task_type="object_detection")
        base.boxes.append(BoundingBox(label="person", xmin=0.1, ymin=0.1, xmax=0.4, ymax=0.4, confidence=0.9))
        llm = DetectionResult(task_type="object_detection")
        llm.boxes.append(BoundingBox(label="person", xmin=0.12, ymin=0.1, xmax=0.42, ymax=0.4, confidence=0.8))
        runtime.low_client = FakeLLMClient(model_name, llm)

        report = runtime._llm_consistency_rerun("image.jpg", operation, base)

        self.assertTrue(report["enabled"])
        self.assertEqual([item["level"] for item in report["comparisons"]], ["low"])

    def test_llm_review_queue_keeps_only_threshold_failures(self):
        records = [
            {
                "image": "pass.jpg",
                "specialist_consistency": {
                    "llm_consistency": {
                        "enabled": True,
                        "review_required": False,
                        "mode": "low",
                        "threshold": 0.75,
                        "comparisons": [],
                    }
                },
            },
            {
                "image": "fail.jpg",
                "specialist_consistency": {
                    "llm_consistency": {
                        "enabled": True,
                        "review_required": True,
                        "mode": "high",
                        "threshold": 0.75,
                        "comparisons": [
                            {
                                "level": "high",
                                "model": "high-model",
                                "bbox_agreement": {
                                    "agreement": 0.4,
                                    "mean_matched_iou": 0.6,
                                    "pseudo_precision": 0.5,
                                    "pseudo_recall": 0.5,
                                    "pseudo_f1": 0.5,
                                },
                                "review_required": True,
                            }
                        ],
                    }
                },
            },
        ]

        queue = _llm_review_queue(records)

        self.assertEqual([item["image"] for item in queue], ["fail.jpg"])
        self.assertEqual(queue[0]["mode"], "high")
        self.assertEqual(queue[0]["mean_bbox_agreement"], 0.4)

    def test_export_generation_writes_low_and_high_llm_labels_separately(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            img_dir = root / "images"
            out_dir = root / "labeled"
            vis_dir = root / "visualized"
            img_dir.mkdir()
            Image.new("RGB", (100, 100), "white").save(img_dir / "sample.jpg")

            base = DetectionResult(task_type="object_detection", source_model="vision")
            base.boxes.append(BoundingBox(label="person", xmin=0.1, ymin=0.1, xmax=0.4, ymax=0.4, confidence=0.9))
            low = DetectionResult(task_type="object_detection", source_model="low")
            low.boxes.append(BoundingBox(label="person", xmin=0.12, ymin=0.1, xmax=0.42, ymax=0.4, confidence=0.8))
            high = DetectionResult(task_type="object_detection", source_model="high")
            high.boxes.append(BoundingBox(label="car", xmin=0.5, ymin=0.5, xmax=0.7, ymax=0.7, confidence=0.7))

            operation = OperationPlan(
                action="generate",
                task_type="object_detection",
                img_dir=str(img_dir),
                out_dir=str(out_dir),
                vis_dir=str(vis_dir),
                formats=["yolo"],
                llm_consistency_mode="both",
                threshold=0.75,
            )
            records = [
                {
                    "image": "sample.jpg",
                    "status": "Completed",
                    "result": base.model_dump(),
                    "issues": [],
                    "plugin_records": [],
                    "elapsed_sec": 0.1,
                    "low_api_attempts": 0,
                    "high_api_attempts": 0,
                    "specialist_consistency": {
                        "llm_consistency": {
                            "enabled": True,
                            "mode": "both",
                            "threshold": 0.75,
                            "review_required": True,
                            "comparisons": [
                                {
                                    "level": "low",
                                    "model": "low-model",
                                    "result": low.model_dump(),
                                    "bbox_agreement": {"agreement": 1.0},
                                    "review_required": False,
                                },
                                {
                                    "level": "high",
                                    "model": "high-model",
                                    "result": high.model_dump(),
                                    "bbox_agreement": {"agreement": 0.0},
                                    "review_required": True,
                                },
                            ],
                        }
                    },
                }
            ]

            summary = WorkflowRuntime().export_generation(operation, records)

            self.assertTrue((out_dir / "sample.txt").is_file())
            self.assertTrue((root / "labeled_llm" / "low" / "sample.txt").is_file())
            self.assertTrue((root / "labeled_llm" / "high" / "sample.txt").is_file())
            self.assertFalse((root / "labeled_review" / "low" / "sample.txt").exists())
            self.assertTrue((root / "labeled_review" / "high" / "sample.txt").is_file())
            self.assertTrue(summary["llm_exports"]["enabled"])
            self.assertEqual(summary["llm_exports"]["output_root"], str(root / "labeled_llm"))
            comparisons = records[0]["specialist_consistency"]["llm_consistency"]["comparisons"]
            self.assertIn("label_paths", comparisons[0])
            self.assertIn("review_label_paths", comparisons[1])

    def test_llm_consistency_can_reuse_existing_generation_records(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            img_dir = root / "images"
            out_dir = root / "labeled"
            vis_dir = root / "visualized"
            img_dir.mkdir()
            Image.new("RGB", (100, 100), "white").save(img_dir / "sample.jpg")

            base = DetectionResult(task_type="object_detection", source_model="vision")
            base.boxes.append(BoundingBox(label="person", xmin=0.1, ymin=0.1, xmax=0.4, ymax=0.4, confidence=0.9))
            llm = DetectionResult(task_type="object_detection", source_model="low")
            llm.boxes.append(BoundingBox(label="person", xmin=0.12, ymin=0.1, xmax=0.42, ymax=0.4, confidence=0.8))
            operation = OperationPlan(
                action="generate",
                task_type="object_detection",
                img_dir=str(img_dir),
                out_dir=str(out_dir),
                vis_dir=str(vis_dir),
                formats=["yolo"],
                llm_consistency_mode="none",
                low_model="low",
                high_model="high",
            )
            records = [{
                "image": "sample.jpg",
                "status": "Completed",
                "result": base.model_dump(),
                "issues": [],
                "plugin_records": [{"plugin": "grounding_dino"}],
                "elapsed_sec": 0.1,
                "low_api_attempts": 0,
                "high_api_attempts": 0,
                "first_pass_report": {"total_labels": 1},
                "specialist_consistency": {"enabled": False},
            }]
            runtime = WorkflowRuntime()
            runtime.low_client = FakeLLMClient("low", llm)

            summary = runtime.run_llm_consistency_on_records(operation, records, "low")

            self.assertEqual(summary["specialist_consistency"]["llm_enabled_images"], 1)
            self.assertTrue(summary["llm_exports"]["enabled"])
            self.assertTrue((root / "labeled_llm" / "low" / "sample.txt").is_file())
            self.assertEqual(records[0]["specialist_consistency"], {"enabled": False})

    def test_llm_consistency_summary_counts_llm_even_when_parent_specialist_disabled(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            out_dir = root / "out"
            img_dir = root / "images"
            out_dir.mkdir()
            img_dir.mkdir()
            Image.new("RGB", (100, 100), "white").save(img_dir / "sample.jpg")
            records = [{
                "image": "sample.jpg",
                "status": "Completed",
                "result": DetectionResult(task_type="object_detection").model_dump(),
                "issues": [],
                "plugin_records": [],
                "elapsed_sec": 0.1,
                "low_api_attempts": 1,
                "high_api_attempts": 0,
                "specialist_consistency": {
                    "enabled": False,
                    "llm_consistency": {
                        "enabled": True,
                        "mode": "low",
                        "threshold": 0.75,
                        "review_required": True,
                        "comparisons": [{
                            "level": "low",
                            "agreement": 0.25,
                            "result_consistency": 0.25,
                            "bbox_agreement": {"agreement": 0.0},
                            "review_required": True,
                        }],
                    },
                },
            }]
            operation = OperationPlan(action="generate", out_dir=str(out_dir), img_dir=str(img_dir))

            with patch("src.workflow.runtime.visualize_boxes", return_value="vis.jpg"):
                summary = WorkflowRuntime(insight_advisor=lambda _payload: {}).export_generation(operation, records)

            self.assertEqual(summary["specialist_consistency"]["llm_enabled_images"], 1)
            self.assertEqual(summary["specialist_consistency"]["llm_review_required_images"], 1)
            self.assertEqual(summary["specialist_consistency"]["llm_mean_prediction_consistency"], 0.25)

    def test_segmentation_llm_consistency_uses_polygon_consistency_when_no_boxes(self):
        runtime = WorkflowRuntime()
        operation = OperationPlan(
            action="generate",
            task_type="segmentation",
            llm_consistency_mode="low",
            low_model="low",
            high_model="high",
            threshold=0.75,
        )
        base = DetectionResult(task_type="segmentation")
        base.segments.append(PolygonSegment(
            label="person",
            polygon=[Point(x=0.1, y=0.1), Point(x=0.4, y=0.1), Point(x=0.4, y=0.4), Point(x=0.1, y=0.4)],
            confidence=0.9,
        ))
        llm = DetectionResult(task_type="segmentation")
        llm.segments.append(PolygonSegment(
            label="person",
            polygon=[Point(x=0.12, y=0.1), Point(x=0.42, y=0.1), Point(x=0.42, y=0.4), Point(x=0.12, y=0.4)],
            confidence=0.8,
        ))
        runtime.low_client = FakeLLMClient("low", llm)

        report = runtime._llm_consistency_rerun("image.jpg", operation, base)

        self.assertTrue(report["enabled"])
        self.assertGreater(report["mean_prediction_consistency"], 0.0)
        self.assertEqual(report["mean_bbox_agreement"], report["mean_prediction_consistency"])
        self.assertFalse(report["review_required"])


if __name__ == "__main__":
    unittest.main()
