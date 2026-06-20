import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from src.workflow.conversation import build_conversation_plan, describe_plan, discover_workspace
from src.workflow.service import execute_workflow_plan


class ConversationWorkflowTests(unittest.TestCase):
    def _workspace(self, root: Path) -> None:
        image_dir = root / "data" / "raw"
        label_dir = root / "data" / "labeled"
        image_dir.mkdir(parents=True)
        label_dir.mkdir(parents=True)
        (image_dir / "sample.jpg").write_bytes(b"not-needed-for-discovery")
        (label_dir / "sample.txt").write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")
        (label_dir / "classes.txt").write_text("car\n", encoding="utf-8")

    def test_discovers_images_and_yolo_labels(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._workspace(root)

            inventory = discover_workspace(root)

            self.assertEqual(inventory["image_directories"][0]["relative_path"], "data/raw")
            self.assertEqual(inventory["label_candidates"][0]["relative_path"], "data/labeled")
            self.assertEqual(inventory["label_candidates"][0]["format"], "yolo")
            self.assertEqual(inventory["label_candidates"][0]["file_count"], 1)

    def test_korean_conversion_request_builds_coco_plan_with_discovered_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._workspace(root)

            proposal = build_conversation_plan(
                "현재 데이터셋의 라벨링 형식을 MS COCO 형식으로 바꿔줘",
                root,
            )
            operation = proposal["plan"]["operations"][0]

            self.assertEqual(operation["action"], "convert")
            self.assertEqual(operation["source_format"], "auto")
            self.assertEqual(operation["formats"], ["coco"])
            self.assertEqual(Path(operation["input_path"]), root / "data" / "labeled")
            self.assertEqual(Path(operation["img_dir"]), root / "data" / "raw")
            self.assertEqual(Path(operation["out_dir"]), root / "data" / "converted")
            description = describe_plan(proposal, root)
            self.assertIn("data/labeled", description)
            self.assertIn("`coco`", description)

    def test_generated_outputs_and_config_json_are_not_selected_as_inputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._workspace(root)
            converted = root / "data" / "converted"
            converted.mkdir(parents=True)
            (converted / "coco_annotations.json").write_text(
                json.dumps({"images": [], "annotations": [], "categories": []}),
                encoding="utf-8",
            )
            configs = root / "configs"
            configs.mkdir()
            (configs / "plugins.json").write_text('{"plugins": []}', encoding="utf-8")

            inventory = discover_workspace(root)

            self.assertEqual(len(inventory["label_candidates"]), 1)
            self.assertEqual(inventory["label_candidates"][0]["relative_path"], "data/labeled")

    def test_requires_an_explicit_target_format(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._workspace(root)

            with self.assertRaisesRegex(ValueError, "출력 포맷"):
                build_conversation_plan("현재 라벨 형식을 바꿔줘", root)

    def test_conversation_plan_executes_yolo_to_coco_conversion(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._workspace(root)
            Image.new("RGB", (100, 80), "white").save(root / "data" / "raw" / "sample.jpg")
            proposal = build_conversation_plan(
                "현재 데이터셋의 라벨링 형식을 MS COCO 형식으로 바꿔줘",
                root,
            )

            result = execute_workflow_plan(proposal["plan"], thread_id="conversation-test")

            self.assertEqual(result["status"], "completed")
            output = result["operation_outputs"][0]
            self.assertEqual(output["resolved_source_format"], "yolo")
            self.assertEqual(output["records_converted"], 1)
            coco_path = root / "data" / "converted" / "coco_annotations.json"
            self.assertTrue(coco_path.is_file())
            coco = json.loads(coco_path.read_text(encoding="utf-8"))
            self.assertEqual(len(coco["images"]), 1)
            self.assertEqual(len(coco["annotations"]), 1)

    def test_requested_conversion_generation_and_evaluation_prompts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._workspace(root)
            Image.new("RGB", (100, 80), "white").save(root / "data" / "raw" / "sample.jpg")
            voc = root / "data" / "voc"
            voc.mkdir()
            (voc / "sample.xml").write_text(
                "<annotation><filename>sample.jpg</filename><size><width>100</width><height>80</height></size>"
                "<object><name>car</name><bndbox><xmin>10</xmin><ymin>10</ymin>"
                "<xmax>50</xmax><ymax>50</ymax></bndbox></object></annotation>",
                encoding="utf-8",
            )
            external = root / "data" / "external_labels"
            external.mkdir()
            (external / "sample.csv").write_text(
                "image,xmin,ymin,xmax,ymax,label\nsample.jpg,1,1,5,5,car\n",
                encoding="utf-8",
            )
            baseline = root / "runs" / "baseline"
            cascade = root / "runs" / "cascade"
            baseline.mkdir(parents=True)
            cascade.mkdir(parents=True)
            metrics = "image,objects,elapsed_sec\nsample.jpg,1,0.1\n"
            (baseline / "run_metrics.csv").write_text(metrics, encoding="utf-8")
            (cascade / "run_metrics.csv").write_text(metrics, encoding="utf-8")
            (baseline / "sample.txt").write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")
            (cascade / "sample.txt").write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")
            ground_truth = root / "data" / "ground_truth"
            ground_truth.mkdir()
            (ground_truth / "sample.txt").write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")

            cases = [
                ("기존 YOLO 라벨들을 전부 COCO 형식으로 변환해줄래?", "convert", ["coco"]),
                ("Pascal VOC 데이터셋을 YOLO 포맷으로 변경해줘.", "convert", ["yolo"]),
                ("여기 있는 라벨 데이터들을 MS COCO 포맷으로 싹 다 바꿔주세요.", "convert", ["coco"]),
                ("csv 라벨 파일들을 yolo 형식으로 변환하고 싶어.", "convert", ["yolo"]),
                ("현재 데이터셋 포맷이 뭔지 확인하고 Vision JSON 형식으로 통일시켜줘.", "convert", ["vision_json"]),
                ("data/external_labels 폴더에 있는 라벨들을 COCO 형식으로 바꿔서 내보내 줘. 중복 IoU 기준은 0.9로 설정해.", "convert", ["coco"]),
                ("라벨들을 YOLO로 변환할 건데, 검증 이슈(Validation issue)가 있는 레코드는 엄격하게(strict) 제외하고 진행해줘.", "convert", ["yolo"]),
                ("현재 데이터셋 라벨을 파스칼 VOC랑 MS COCO 두 가지 형식으로 동시에 변환해.", "convert", ["coco", "pascal_voc"]),
                ("이 이미지 폴더에 있는 사진들에 대해 YOLO 형식으로 객체 탐지 라벨을 새로 생성해줘.", "generate", ["yolo"]),
                ("이미지들에서 차량과 보행자를 찾아서 세그멘테이션(Segmentation) 라벨링을 진행해.", "generate", ["vision_json"]),
                ("아직 라벨이 없는 이미지들인데, MS COCO 포맷으로 자동 라벨링을 싹 돌려줘.", "generate", ["coco"]),
                ("신뢰도(Threshold) 0.8 이상인 객체들만 찾아서 Vision JSON 형식으로 라벨을 만들어.", "generate", ["vision_json"]),
                ("정답 라벨(Ground truth) 폴더랑 비교해서 현재 라벨들의 성능 평가를 진행해.", "evaluate", ["yolo"]),
                ("베이스라인 결과랑 이번에 Cascade 모델로 돌린 결과를 비교해서 평가 리포트를 뽑아줘.", "evaluate", ["yolo"]),
            ]
            for prompt, expected_action, expected_formats in cases:
                with self.subTest(prompt=prompt):
                    proposal = build_conversation_plan(prompt, root)
                    operation = proposal["plan"]["operations"][0]
                    self.assertEqual(operation["action"], expected_action)
                    self.assertEqual(operation["formats"], expected_formats)

            for prompt, _, expected_formats in cases[:8]:
                with self.subTest(execution=prompt):
                    proposal = build_conversation_plan(prompt, root)
                    result = execute_workflow_plan(proposal["plan"], thread_id="prompt-conversion-test")
                    self.assertEqual(result["status"], "completed")
                    self.assertEqual(result["operation_outputs"][0]["target_formats"], expected_formats)

            for prompt, _, _ in cases[12:14]:
                with self.subTest(execution=prompt):
                    proposal = build_conversation_plan(prompt, root)
                    result = execute_workflow_plan(proposal["plan"], thread_id="prompt-evaluation-test")
                    self.assertEqual(result["status"], "completed")
                    self.assertTrue(result["operation_outputs"][0]["rows"])

            voc_plan = build_conversation_plan(cases[1][0], root)["plan"]["operations"][0]
            csv_plan = build_conversation_plan(cases[3][0], root)["plan"]["operations"][0]
            path_plan = build_conversation_plan(cases[5][0], root)["plan"]["operations"][0]
            strict_plan = build_conversation_plan(cases[6][0], root)["plan"]["operations"][0]
            threshold_plan = build_conversation_plan(cases[11][0], root)["plan"]["operations"][0]
            self.assertEqual(Path(voc_plan["input_path"]), voc)
            self.assertEqual(Path(csv_plan["input_path"]), external)
            self.assertEqual(Path(path_plan["input_path"]), external)
            self.assertEqual(path_plan["duplicate_iou"], 0.9)
            self.assertTrue(strict_plan["strict"])
            self.assertEqual(threshold_plan["threshold"], 0.8)
            self.assertEqual(
                build_conversation_plan(cases[9][0], root)["plan"]["operations"][0]["task_type"],
                "segmentation",
            )

            unsupported = "변환된 라벨들이 원본 이미지랑 잘 맞는지 평가를 돌려볼래?"
            with self.assertRaisesRegex(ValueError, "공간 정합성 검증"):
                build_conversation_plan(unsupported, root)


if __name__ == "__main__":
    unittest.main()
