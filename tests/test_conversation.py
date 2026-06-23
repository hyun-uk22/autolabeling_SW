import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from src.workflow.conversation import build_conversation_plan, describe_plan, discover_workspace
from src.workflow import conversation_router as conversation_router_module
from src.workflow.conversation_router import ChatNode, IntentRouter, handle_conversation
from src.workflow.plan_patcher import revise_pending_proposal
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

    def test_conversion_discovers_common_labels_folder_when_labeled_is_absent(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image_dir = root / "images"
            label_dir = root / "labels"
            image_dir.mkdir()
            label_dir.mkdir()
            Image.new("RGB", (100, 80), "white").save(image_dir / "sample.jpg")
            (label_dir / "sample.txt").write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")
            (label_dir / "classes.txt").write_text("car\n", encoding="utf-8")

            proposal = build_conversation_plan(
                "현재 데이터셋의 라벨링 형식을 MS COCO 형식으로 바꿔줘",
                root,
            )
            operation = proposal["plan"]["operations"][0]

            self.assertEqual(Path(operation["input_path"]), label_dir)
            self.assertEqual(Path(operation["img_dir"]), image_dir)
            self.assertEqual(operation["formats"], ["coco"])

    def test_conversion_links_generated_labeled_folder_to_best_image_candidate(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            label_dir = root / "data" / "labeled"
            image_dir = root / "data" / "images"
            other_images = root / "misc" / "images"
            label_dir.mkdir(parents=True)
            image_dir.mkdir(parents=True)
            other_images.mkdir(parents=True)
            Image.new("RGB", (100, 80), "white").save(image_dir / "sample.jpg")
            Image.new("RGB", (100, 80), "white").save(other_images / "other.jpg")
            (label_dir / "sample.txt").write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")
            (label_dir / "classes.txt").write_text("car\n", encoding="utf-8")

            proposal = build_conversation_plan(
                "현재 데이터셋의 라벨링 형식을 MS COCO 형식으로 바꿔줘",
                root,
            )
            operation = proposal["plan"]["operations"][0]

            self.assertEqual(Path(operation["input_path"]), label_dir)
            self.assertEqual(Path(operation["img_dir"]), image_dir)
            self.assertTrue(any("이미지 후보" in warning for warning in proposal["warnings"]))

    def test_conversion_to_yolo_can_plan_without_images(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            label_dir = root / "data" / "labeled"
            label_dir.mkdir(parents=True)
            (label_dir / "sample.xml").write_text(
                """<?xml version="1.0" encoding="utf-8"?>
<annotation>
  <filename>sample.jpg</filename>
  <size><width>100</width><height>80</height><depth>3</depth></size>
  <object><name>car</name><bndbox><xmin>10</xmin><ymin>10</ymin><xmax>50</xmax><ymax>50</ymax></bndbox></object>
</annotation>
""",
                encoding="utf-8",
            )

            proposal = build_conversation_plan(
                "현재 데이터셋의 라벨링 형식을 yolo26에서 사용할 수 있게 바꿔줘",
                root,
            )
            operation = proposal["plan"]["operations"][0]

            self.assertEqual(operation["formats"], ["yolo"])
            self.assertEqual(Path(operation["input_path"]), label_dir)
            self.assertEqual(Path(operation["img_dir"]), label_dir)
            self.assertTrue(any("이미지 크기가 필요 없는 출력" in warning for warning in proposal["warnings"]))

    def test_conversion_to_coco_without_images_warns_instead_of_failing_plan(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            label_dir = root / "data" / "labeled"
            label_dir.mkdir(parents=True)
            (label_dir / "classes.txt").write_text("car\n", encoding="utf-8")
            (label_dir / "sample.txt").write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")

            proposal = build_conversation_plan(
                "현재 데이터셋의 라벨링 형식을 COCO형식으로 바꿔줘",
                root,
            )
            operation = proposal["plan"]["operations"][0]

            self.assertEqual(operation["formats"], ["coco"])
            self.assertEqual(Path(operation["img_dir"]), label_dir)
            self.assertTrue(any("COCO/Pascal VOC 출력은 이미지 크기가 필요" in warning for warning in proposal["warnings"]))

    def test_conversion_discovers_custom_json_with_bbox_key(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            label_dir = root / "data" / "labeled"
            label_dir.mkdir(parents=True)
            (label_dir / "custom.json").write_text(
                json.dumps({"items": [{"image": "sample.jpg", "label": "car", "bbox": [0.1, 0.1, 0.4, 0.4]}]}),
                encoding="utf-8",
            )

            proposal = build_conversation_plan(
                "현재 데이터셋의 라벨링 형식을 yolo 형식으로 바꿔줘. json 안에 bbox 가 있어",
                root,
            )
            operation = proposal["plan"]["operations"][0]

            self.assertEqual(Path(operation["input_path"]), label_dir / "custom.json")
            self.assertEqual(operation["formats"], ["yolo"])
            self.assertEqual(operation["source_format"], "auto")

    def test_rule_parser_runs_before_llm_intent_router(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._workspace(root)
            calls = []
            router = IntentRouter(caller=lambda request: calls.append(request))

            routed = handle_conversation(
                "현재 데이터셋의 라벨링 형식을 MS COCO 형식으로 바꿔줘",
                str(root),
                intent_router=router,
            )

            self.assertEqual(routed["kind"], "plan")
            self.assertEqual(routed["route_source"], "rules")
            self.assertEqual(calls, [])

    def test_llm_router_fills_unsupported_conversion_expression(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._workspace(root)
            router = IntentRouter(caller=lambda request: {
                "intent": "convert_labels",
                "confidence": 0.93,
                "parameters": {"target_formats": ["coco"], "source_format": "yolo"},
                "missing_parameters": [],
            })

            routed = handle_conversation(
                "보관된 annotation을 코코 규격에 맞게 재구성해줘",
                str(root),
                intent_router=router,
            )

            operation = routed["proposal"]["plan"]["operations"][0]
            self.assertEqual(routed["route_source"], "llm")
            self.assertEqual(operation["action"], "convert")
            self.assertEqual(operation["formats"], ["coco"])
            self.assertEqual(operation["source_format"], "auto")

    def test_llm_router_recovers_from_stale_conversation_plan_import(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._workspace(root)
            router = IntentRouter(caller=lambda request: {
                "intent": "convert_labels",
                "confidence": 0.93,
                "parameters": {"target_formats": ["coco"], "source_format": "yolo"},
                "missing_parameters": [],
            })
            current_builder = conversation_router_module.conversation_module.build_conversation_plan

            def stale_builder(request, workspace):
                return current_builder(request, workspace)

            conversation_router_module.conversation_module.build_conversation_plan = stale_builder
            try:
                routed = handle_conversation(
                    "보관된 annotation을 코코 규격에 맞게 재구성해줘",
                    str(root),
                    intent_router=router,
                )
            finally:
                conversation_router_module.conversation_module.build_conversation_plan = current_builder

            operation = routed["proposal"]["plan"]["operations"][0]
            self.assertEqual(routed["route_source"], "llm")
            self.assertEqual(operation["action"], "convert")
            self.assertEqual(operation["formats"], ["coco"])

    def test_general_chat_is_handled_without_an_execution_plan(self):
        with tempfile.TemporaryDirectory() as directory:
            router = IntentRouter(caller=lambda request: {
                "intent": "general_chat",
                "confidence": 0.98,
                "parameters": {},
                "missing_parameters": [],
            })
            chat = ChatNode(caller=lambda request: "COCO와 YOLO의 차이를 설명한 답변")

            routed = handle_conversation(
                "COCO와 YOLO는 어떤 차이가 있어?",
                directory,
                intent_router=router,
                chat_node=chat,
            )

            self.assertEqual(routed["kind"], "chat")
            self.assertNotIn("proposal", routed)
            self.assertIn("차이", routed["response"])

    def test_low_confidence_route_requests_clarification(self):
        with tempfile.TemporaryDirectory() as directory:
            router = IntentRouter(caller=lambda request: {
                "intent": "convert_labels",
                "confidence": 0.4,
                "parameters": {"target_formats": ["coco"]},
                "missing_parameters": [],
            })

            routed = handle_conversation("이거 적당히 처리해줘", directory, intent_router=router)

            self.assertEqual(routed["kind"], "clarification")
            self.assertIn("구체적으로", routed["response"])

    def test_missing_router_parameters_do_not_create_a_plan(self):
        with tempfile.TemporaryDirectory() as directory:
            router = IntentRouter(caller=lambda request: {
                "intent": "convert_labels",
                "confidence": 0.95,
                "parameters": {},
                "missing_parameters": ["target_formats"],
            })

            routed = handle_conversation("annotation 규격을 다시 맞춰줘", directory, intent_router=router)

            self.assertEqual(routed["kind"], "clarification")
            self.assertIn("target_formats", routed["response"])

    def test_router_provider_failure_is_reported_as_a_conversation_error(self):
        with tempfile.TemporaryDirectory() as directory:
            def fail(_request):
                raise RuntimeError("provider unavailable")

            with self.assertRaisesRegex(ValueError, "Intent Router 호출에 실패"):
                handle_conversation(
                    "annotation 규격을 다시 맞춰줘",
                    directory,
                    intent_router=IntentRouter(caller=fail),
                )

    def test_llm_router_cannot_select_a_path_outside_workspace(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._workspace(root)
            router = IntentRouter(caller=lambda request: {
                "intent": "convert_labels",
                "confidence": 0.99,
                "parameters": {
                    "target_formats": ["coco"],
                    "source_path": "../outside/labels",
                },
                "missing_parameters": [],
            })

            with self.assertRaisesRegex(ValueError, "Workspace 밖"):
                handle_conversation("annotation을 새 규격으로 맞춰줘", str(root), intent_router=router)

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
            self.assertEqual(output["report_version"], "2.0")
            self.assertEqual(output["user_action_report"]["status"], "success")
            self.assertEqual(output["dataset_insight"]["distribution"]["car"]["count"], 1)
            self.assertEqual(output["dataset_insight"]["agent"], "DatasetInsightAgent")
            self.assertTrue((root / "data" / "converted" / "user_action_report.json").is_file())
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

    def test_names_block_detection_prompt_uses_path_as_image_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._workspace(root)
            test_images = root / "test_images"
            test_images.mkdir()
            Image.new("RGB", (100, 80), "white").save(test_images / "sample.jpg")
            prompt = f"""names:
  0: person
  1: giraffe
  2: dining table
  3: cup
path: {test_images}

위 클래스 객체만 검출해줘"""

            proposal = build_conversation_plan(prompt, root)
            operation = proposal["plan"]["operations"][0]

            self.assertEqual(operation["action"], "generate")
            self.assertEqual(operation["task_type"], "object_detection")
            self.assertEqual(Path(operation["img_dir"]), test_images)
            self.assertEqual(operation["formats"], ["yolo"])
            self.assertIn("names:", operation["prompt"])

    def test_generate_prompt_can_use_explicit_image_path_outside_workspace(self):
        with tempfile.TemporaryDirectory() as workspace_dir, tempfile.TemporaryDirectory() as image_dir:
            root = Path(workspace_dir)
            images = Path(image_dir)
            Image.new("RGB", (100, 80), "white").save(images / "sample.jpg")
            prompt = f"names: 0: person 1: giraffe 2: dining table 3: cup path: {images}\n\n위 클래스 객체만 검출해줘"

            proposal = build_conversation_plan(prompt, root)
            operation = proposal["plan"]["operations"][0]

            self.assertEqual(operation["action"], "generate")
            self.assertEqual(Path(operation["img_dir"]), images)
            self.assertEqual(operation["formats"], ["yolo"])

    def test_generate_prompt_starts_with_first_pass_only(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._workspace(root)
            Image.new("RGB", (100, 80), "white").save(root / "data" / "raw" / "sample.jpg")

            proposal = build_conversation_plan(
                "현재 이미지들을 YOLO로 객체 탐지 라벨 생성해줘. 1차 추론 이후 specialist 재추론 1회 진행하고 low advisor를 사용해줘.",
                root,
            )
            operation = proposal["plan"]["operations"][0]
            description = describe_plan(proposal, root)

            self.assertEqual(operation["action"], "generate")
            self.assertEqual(operation["specialist_consistency_runs"], 0)
            self.assertEqual(operation["specialist_advisor_mode"], "none")
            self.assertIn("Specialist 재추론", description)
            self.assertIn("재추론 Advisor", description)

    def test_llm_router_generate_plan_starts_with_first_pass_only(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._workspace(root)
            Image.new("RGB", (100, 80), "white").save(root / "data" / "raw" / "sample.jpg")
            router = IntentRouter(caller=lambda request: {
                "intent": "generate_labels",
                "confidence": 0.94,
                "parameters": {
                    "target_formats": ["yolo"],
                    "task_type": "object_detection",
                },
                "missing_parameters": [],
            })

            routed = handle_conversation(
                "이 데이터셋을 후속 검증까지 포함해서 준비해줘",
                str(root),
                intent_router=router,
            )
            operation = routed["proposal"]["plan"]["operations"][0]

            self.assertEqual(routed["route_source"], "llm")
            self.assertEqual(operation["specialist_consistency_runs"], 0)
            self.assertEqual(operation["specialist_advisor_mode"], "none")

    def test_generation_prefers_task_matching_image_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            classification = root / "datasets" / "classification" / "csv" / "data"
            ocr = root / "datasets" / "ocr" / "csv" / "data"
            classification.mkdir(parents=True)
            ocr.mkdir(parents=True)
            for index in range(3):
                Image.new("RGB", (10, 10), "white").save(classification / f"class_{index}.jpg")
            Image.new("RGB", (10, 10), "white").save(ocr / "ocr_0.jpg")

            proposal = build_conversation_plan("OCR 데이터셋에 대해서만 라벨 생성해줘", root)
            operation = proposal["plan"]["operations"][0]

            self.assertEqual(operation["task_type"], "ocr")
            self.assertEqual(Path(operation["img_dir"]).resolve(), ocr.resolve())

    def test_explicit_generation_path_counts_nested_images(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image_root = root / "datasets" / "ocr"
            nested = image_root / "csv" / "data"
            nested.mkdir(parents=True)
            Image.new("RGB", (10, 10), "white").save(nested / "ocr_0.jpg")

            proposal = build_conversation_plan(
                "path: datasets/ocr\nOCR 데이터셋에 대해서만 라벨 생성해줘",
                root,
            )
            operation = proposal["plan"]["operations"][0]

            self.assertEqual(Path(operation["img_dir"]).resolve(), image_root.resolve())
            self.assertIn("이미지 1개", proposal["summary"])

    def test_llm_plan_patch_updates_allowed_fields_only(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._workspace(root)
            proposal = build_conversation_plan("현재 데이터셋을 COCO로 바꿔줘", root)

            revision = revise_pending_proposal(
                "출력 위치를 data/converted_test로 바꾸고 strict 모드로 해줘",
                proposal,
                root,
                caller=lambda _: {
                    "mode": "patch",
                    "reason": "사용자가 출력 위치와 strict 모드를 수정했습니다.",
                    "updates": {
                        "out_dir": "data/converted_test",
                        "strict": True,
                    },
                },
            )

            operation = revision["proposal"]["plan"]["operations"][0]
            self.assertEqual(revision["kind"], "patch")
            self.assertTrue(operation["strict"])
            self.assertEqual(Path(operation["out_dir"]), (root / "data" / "converted_test").resolve())
            self.assertTrue(any("out_dir" in change for change in revision["changes"]))

    def test_llm_plan_patch_rejects_action_change_and_external_path(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._workspace(root)
            proposal = build_conversation_plan("현재 데이터셋을 COCO로 바꿔줘", root)

            with self.assertRaises(ValueError):
                revise_pending_proposal(
                    "라벨 생성으로 바꿔줘",
                    proposal,
                    root,
                    caller=lambda _: {
                        "mode": "patch",
                        "updates": {"action": "generate"},
                    },
                )

            with self.assertRaises(ValueError):
                revise_pending_proposal(
                    "출력 위치를 외부 폴더로 바꿔줘",
                    proposal,
                    root,
                    caller=lambda _: {
                        "mode": "patch",
                        "updates": {"out_dir": "D:/outside"},
                    },
                )

    def test_model_dataset_request_asks_for_missing_information(self):
        with tempfile.TemporaryDirectory() as directory:
            routed = handle_conversation(
                "SegFormer 학습용 데이터셋 구조로 만들어줘",
                directory,
            )

            self.assertEqual(routed["kind"], "clarification")
            self.assertIn("추가 정보가 필요", routed["response"])
            self.assertIn("프레임워크", routed["response"])
            self.assertIn("현재 라벨 형식", routed["response"])
            self.assertEqual(routed["diagnosis"]["model_name"], "segformer")
            self.assertFalse(routed["diagnosis"]["can_create_plan"])

    def test_model_dataset_request_builds_plan_when_information_is_complete(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            proposal = build_conversation_plan(
                "SegFormer를 MMSegmentation에서 training 목적으로 사용할 거야. "
                "현재 라벨 형식은 COCO이고 train 8 val 2로 나눠줘. "
                "출력 위치는 datasets/segformer",
                root,
            )
            operation = proposal["plan"]["operations"][0]

            self.assertEqual(operation["action"], "prepare_model_dataset")
            self.assertEqual(operation["model_name"], "segformer")
            self.assertEqual(operation["framework"], "mmsegmentation")
            self.assertEqual(operation["dataset_purpose"], "training")
            self.assertEqual(operation["source_format"], "coco")
            self.assertEqual(operation["split_train"], 0.8)
            self.assertEqual(operation["split_val"], 0.2)
            self.assertEqual(Path(operation["out_dir"]), root / "datasets" / "segformer")

    def test_model_dataset_plan_executes_and_writes_layout_report(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            proposal = build_conversation_plan(
                "SegFormer를 Hugging Face에서 training 목적으로 사용할 거야. "
                "현재 라벨 형식은 mask image이고 train 8 val 2로 나눠줘. "
                "출력 위치는 datasets/segformer_hf",
                root,
            )

            result = execute_workflow_plan(proposal["plan"], thread_id="model-dataset-test")

            self.assertEqual(result["status"], "completed")
            output = result["operation_outputs"][0]
            self.assertEqual(output["action"], "prepare_model_dataset")
            self.assertEqual(output["layout"], "segformer_huggingface")
            self.assertTrue((root / "datasets" / "segformer_hf" / "images" / "train").is_dir())
            self.assertTrue((root / "datasets" / "segformer_hf" / "masks" / "validation").is_dir())
            report = root / "datasets" / "segformer_hf" / "dataset_layout.json"
            self.assertTrue(report.is_file())
            report_data = json.loads(report.read_text(encoding="utf-8"))
            self.assertEqual(report_data["source_format"], "mask_image")

    def test_official_repo_model_dataset_request_requires_repo_source(self):
        with tempfile.TemporaryDirectory() as directory:
            routed = handle_conversation(
                "MaskDINO 공식 repo git clone 방식으로 training 데이터셋 구조를 만들어줘. "
                "프레임워크는 Detectron2이고 현재 라벨 형식은 COCO야. "
                "train 8 val 2, 출력 위치는 datasets/maskdino",
                directory,
            )

            self.assertEqual(routed["kind"], "clarification")
            self.assertEqual(routed["diagnosis"]["usage_mode"], "official_repo")
            self.assertIn("clone URL", routed["response"])

    def test_official_repo_model_dataset_plan_records_repo_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            proposal = build_conversation_plan(
                "MaskDINO 공식 repo git clone 방식으로 training 데이터셋 구조를 만들어줘. "
                "프레임워크는 Detectron2이고 현재 라벨 형식은 COCO야. "
                "repo: https://github.com/IDEA-Research/MaskDINO "
                "train 8 val 2, 출력 위치는 datasets/maskdino",
                root,
            )
            operation = proposal["plan"]["operations"][0]

            self.assertEqual(operation["usage_mode"], "official_repo")
            self.assertEqual(operation["framework"], "detectron2")
            self.assertEqual(operation["repo_url"], "https://github.com/IDEA-Research/MaskDINO")

            result = execute_workflow_plan(proposal["plan"], thread_id="official-repo-dataset-test")

            output = result["operation_outputs"][0]
            self.assertEqual(output["layout"], "maskdino_official_repo_detectron2")
            self.assertEqual(output["repo_url"], "https://github.com/IDEA-Research/MaskDINO")
            self.assertTrue((root / "datasets" / "maskdino" / "datasets").is_dir())
            report = root / "datasets" / "maskdino" / "dataset_layout.json"
            report_data = json.loads(report.read_text(encoding="utf-8"))
            self.assertEqual(report_data["usage_mode"], "official_repo")


if __name__ == "__main__":
    unittest.main()
