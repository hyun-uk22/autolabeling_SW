import csv
import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from src.core.models import (
    BoundingBox,
    DetectionResult,
    Keypoint,
    Point,
    PolygonSegment,
    PoseInstance,
    TextRegion,
)
from src.utils.geometry import (
    character_similarity,
    compute_pairwise_consistency,
    compute_result_consistency,
    get_consistency_score,
    polygon_mask_iou,
    pose_oks_similarity,
)
from src.workflow.models import OperationPlan
from src.workflow.runtime import WorkflowRuntime


def segment(points):
    return PolygonSegment(label="object", polygon=[Point(x=x, y=y) for x, y in points])


def pose(points):
    return PoseInstance(
        label="person",
        keypoints=[Keypoint(name=name, x=x, y=y) for name, x, y in points],
    )


def text(value, box=(0.1, 0.1, 0.5, 0.3)):
    return TextRegion(
        text=value,
        xmin=box[0],
        ymin=box[1],
        xmax=box[2],
        ymax=box[3],
    )


class GeometryConsistencyTests(unittest.TestCase):
    def test_segmentation_uses_polygon_mask_instead_of_bounding_box(self):
        upper_left = segment([(0.1, 0.1), (0.9, 0.1), (0.1, 0.9)])
        lower_right = segment([(0.9, 0.9), (0.9, 0.1), (0.1, 0.9)])
        left_box = BoundingBox(label="object", xmin=0.1, ymin=0.1, xmax=0.9, ymax=0.9)
        right_box = BoundingBox(label="object", xmin=0.1, ymin=0.1, xmax=0.9, ymax=0.9)

        self.assertEqual(compute_pairwise_consistency([left_box], [right_box]), 1.0)
        self.assertLess(polygon_mask_iou(upper_left, lower_right), 0.01)
        score = compute_result_consistency(
            DetectionResult(task_type="segmentation", segments=[upper_left]),
            DetectionResult(task_type="segmentation", segments=[lower_right]),
        )
        self.assertEqual(score, 0.0)
        self.assertEqual(polygon_mask_iou(upper_left, upper_left), 1.0)

    def test_pose_oks_uses_keypoint_positions_and_penalizes_missing_points(self):
        reference = pose([
            ("nose", 0.5, 0.2),
            ("left_shoulder", 0.35, 0.5),
            ("right_shoulder", 0.65, 0.5),
        ])
        shifted = pose([
            ("nose", 0.7, 0.4),
            ("left_shoulder", 0.55, 0.7),
            ("right_shoulder", 0.85, 0.7),
        ])
        missing = pose([
            ("nose", 0.5, 0.2),
            ("left_shoulder", 0.35, 0.5),
        ])

        self.assertEqual(pose_oks_similarity(reference, reference), 1.0)
        self.assertLess(pose_oks_similarity(reference, shifted), 0.1)
        self.assertAlmostEqual(pose_oks_similarity(reference, missing), 2 / 3)

    def test_ocr_combines_character_similarity_with_region_iou(self):
        reference = DetectionResult(task_type="ocr", texts=[text("안녕하세요")])
        typo = DetectionResult(task_type="ocr", texts=[text("안녕하세오")])
        moved = DetectionResult(task_type="ocr", texts=[text("안녕하세요", (0.6, 0.6, 0.9, 0.9))])

        self.assertAlmostEqual(character_similarity("안녕하세요", "안녕하세오"), 0.8)
        self.assertEqual(character_similarity("가", "\u1100\u1161"), 1.0)
        self.assertAlmostEqual(compute_result_consistency(reference, typo), 0.88)
        self.assertEqual(compute_result_consistency(reference, moved), 0.0)

    def test_repeated_inference_score_averages_all_result_pairs(self):
        exact = DetectionResult(task_type="ocr", texts=[text("ABC")])
        typo = DetectionResult(task_type="ocr", texts=[text("ADC")])

        expected = (
            compute_result_consistency(exact, exact)
            + compute_result_consistency(exact, typo)
            + compute_result_consistency(exact, typo)
        ) / 3
        self.assertAlmostEqual(get_consistency_score([exact, exact, typo]), expected)

    def test_generation_report_records_task_specific_consistency_metric(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image_dir = root / "images"
            output_dir = root / "labels"
            visual_dir = root / "visualized"
            image_dir.mkdir()
            Image.new("RGB", (100, 100), "white").save(image_dir / "sample.jpg")
            result = DetectionResult(
                task_type="segmentation",
                segments=[segment([(0.1, 0.1), (0.8, 0.1), (0.1, 0.8)])],
                consistency_score=0.9,
                mean_confidence=1.0,
                uncertainty_score=0.05,
            )
            operation = OperationPlan(
                action="generate",
                task_type="segmentation",
                img_dir=str(image_dir),
                out_dir=str(output_dir),
                vis_dir=str(visual_dir),
                formats=["vision_json"],
            )

            summary = WorkflowRuntime().export_generation(operation, [{
                "image": "sample.jpg",
                "result": result.model_dump(),
                "status": "Consistent",
                "plugin_records": [],
                "issues": [],
                "low_api_attempts": 3,
                "high_api_attempts": 0,
                "elapsed_sec": 0.1,
            }])

            self.assertEqual(summary["consistency_metric"], "polygon_mask_iou")
            saved_summary = json.loads((output_dir / "run_summary.json").read_text(encoding="utf-8"))
            self.assertEqual(saved_summary["consistency_metric"], "polygon_mask_iou")
            with (output_dir / "run_metrics.csv").open(encoding="utf-8", newline="") as handle:
                row = next(csv.DictReader(handle))
            self.assertEqual(row["consistency_metric"], "polygon_mask_iou")


if __name__ == "__main__":
    unittest.main()
