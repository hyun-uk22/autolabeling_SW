from typing import List

from PIL import Image

from ..core.models import (
    BoundingBox,
    ClassificationLabel,
    DetectionResult,
    Keypoint,
    Point,
    PolygonSegment,
    PoseInstance,
    TextRegion,
    TrackInstance,
)
from .base import PluginOutput, VisionTaskPlugin, configured_labels


def _resolve_device(device: str | None = None) -> str:
    requested = str(device or "auto").strip().lower()
    try:
        import torch
    except ImportError:
        return "cpu"

    cuda_available = bool(getattr(torch, "cuda", None) and torch.cuda.is_available())
    mps_backend = getattr(getattr(torch, "backends", None), "mps", None)
    mps_available = bool(mps_backend and mps_backend.is_available())

    if requested in {"", "auto", "gpu"}:
        if cuda_available:
            return "cuda:0"
        if mps_available:
            return "mps"
        return "cpu"
    if requested.startswith("cuda") and not cuda_available:
        return "cpu"
    if requested == "mps" and not mps_available:
        return "cpu"
    return requested


def _device(config: dict) -> str:
    if "_resolved_device" not in config:
        config["_resolved_device"] = _resolve_device(config.get("device", "auto"))
    return config["_resolved_device"]


def _device_index(device: str) -> int:
    if device.startswith("cuda"):
        parts = device.split(":", 1)
        return int(parts[1]) if len(parts) == 2 else 0
    return -1


def _pipeline_device(device: str):
    return _device_index(device) if device != "mps" else "mps"


def _easyocr_gpu(config: dict) -> bool:
    configured = config.get("gpu", "auto")
    if isinstance(configured, bool):
        return configured
    return _device(config).startswith("cuda")


def _paddleocr_lang(config: dict) -> str:
    if config.get("lang"):
        return str(config["lang"])
    languages = [str(value).lower() for value in config.get("languages", [])]
    if any(value in {"ko", "kr", "korean"} for value in languages):
        return "korean"
    if any(value in {"en", "english"} for value in languages):
        return "en"
    return "korean"


def _mean(values) -> float:
    values = list(values)
    return sum(values) / len(values) if values else 0.0


def _value_at(values, index: int, default=None):
    if values is None:
        return default
    if hasattr(values, "detach"):
        values = values.detach().cpu()
    if hasattr(values, "tolist"):
        values = values.tolist()
    try:
        return values[index]
    except (IndexError, TypeError):
        return default


def _mask_to_polygons(mask, width: int, height: int) -> List[List[Point]]:
    if hasattr(mask, "detach"):
        mask = mask.detach().cpu().numpy()
    try:
        import cv2
        import numpy as np
    except ImportError:
        ys, xs = mask.nonzero()
        if len(xs) == 0 or len(ys) == 0:
            return []
        x1, x2 = float(xs.min()) / width, float(xs.max()) / width
        y1, y2 = float(ys.min()) / height, float(ys.max()) / height
        return [[Point(x=x1, y=y1), Point(x=x2, y=y1), Point(x=x2, y=y2), Point(x=x1, y=y2)]]

    mask_array = np.asarray(mask)
    if mask_array.ndim > 2:
        mask_array = np.squeeze(mask_array)
    mask_array = (mask_array > 0).astype("uint8") * 255
    contours, _ = cv2.findContours(mask_array, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polygons: List[List[Point]] = []
    for contour in contours:
        if len(contour) < 3:
            continue
        epsilon = 0.002 * cv2.arcLength(contour, True)
        approximated = cv2.approxPolyDP(contour, epsilon, True)
        points = [
            Point(
                x=max(0.0, min(1.0, float(point[0][0]) / width)),
                y=max(0.0, min(1.0, float(point[0][1]) / height)),
            )
            for point in approximated
        ]
        if len(points) >= 3:
            polygons.append(points)
    return polygons


class TransformersClassificationPlugin(VisionTaskPlugin):
    plugin_name = "classification"
    supported_tasks = {"classification"}

    def __init__(self, config=None):
        super().__init__(config)
        self._pipeline = None

    def _load(self):
        if self._pipeline is None:
            try:
                from transformers import pipeline
            except ImportError as exc:
                raise RuntimeError("Install requirements-specialists.txt to use the classification plugin") from exc
            self._pipeline = pipeline(
                "zero-shot-image-classification",
                model=self.config.get("model", "openai/clip-vit-base-patch32"),
                device=_pipeline_device(_device(self.config)),
            )

    def refine(self, image_path, prompt, seed_result):
        self._load()
        labels = list(configured_labels(self.config, seed_result))
        if not labels:
            raise ValueError("classification plugin requires config.labels or VLM candidate labels")
        predictions = self._pipeline(Image.open(image_path).convert("RGB"), candidate_labels=labels)
        limit = int(self.config.get("top_k", 5))
        result = DetectionResult(task_type="classification")
        for item in predictions[:limit]:
            result.classifications.append(
                ClassificationLabel(label=item["label"], confidence=float(item["score"]))
            )
        return PluginOutput(
            result=result,
            score=result.classifications[0].confidence if result.classifications else 0.0,
            metadata={"model": self.config.get("model", "openai/clip-vit-base-patch32")},
        )


class GroundingDINOPlugin(VisionTaskPlugin):
    plugin_name = "grounding_dino"
    supported_tasks = {"object_detection", "segmentation", "tracking"}

    def __init__(self, config=None):
        super().__init__(config)
        self._processor = None
        self._model = None

    def _load(self):
        if self._model is not None:
            return
        try:
            import torch
            from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
        except ImportError as exc:
            raise RuntimeError("Install requirements-specialists.txt to use Grounding DINO") from exc
        model_id = self.config.get("model", "IDEA-Research/grounding-dino-base")
        self._processor = AutoProcessor.from_pretrained(model_id)
        self._model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id)
        self._model.to(_device(self.config))
        self._model.eval()
        self._torch = torch

    def refine(self, image_path, prompt, seed_result):
        self._load()
        labels = list(configured_labels(self.config, seed_result))
        if not labels:
            raise ValueError("Grounding DINO requires config.labels or VLM candidate labels")
        image = Image.open(image_path).convert("RGB")
        text_prompt = ". ".join(labels) + "."
        inputs = self._processor(images=image, text=text_prompt, return_tensors="pt")
        inputs = {key: value.to(_device(self.config)) for key, value in inputs.items()}
        with self._torch.no_grad():
            outputs = self._model(**inputs)
        box_threshold = float(self.config.get("box_threshold", 0.35))
        text_threshold = float(self.config.get("text_threshold", 0.25))
        try:
            processed = self._processor.post_process_grounded_object_detection(
                outputs,
                inputs["input_ids"],
                box_threshold=box_threshold,
                text_threshold=text_threshold,
                target_sizes=[image.size[::-1]],
            )[0]
        except TypeError:
            processed = self._processor.post_process_grounded_object_detection(
                outputs,
                inputs["input_ids"],
                threshold=box_threshold,
                text_threshold=text_threshold,
                target_sizes=[image.size[::-1]],
            )[0]
        width, height = image.size
        result = DetectionResult(task_type="object_detection")
        detected_labels = processed.get("text_labels")
        if detected_labels is None:
            detected_labels = processed.get("labels", [])
        for box, score, label in zip(processed["boxes"], processed["scores"], detected_labels):
            x1, y1, x2, y2 = [float(value) for value in box.tolist()]
            result.boxes.append(
                BoundingBox(
                    label=str(label).rstrip("."),
                    xmin=max(0.0, min(1.0, x1 / width)),
                    ymin=max(0.0, min(1.0, y1 / height)),
                    xmax=max(0.0, min(1.0, x2 / width)),
                    ymax=max(0.0, min(1.0, y2 / height)),
                    confidence=float(score),
                )
            )
        return PluginOutput(
            result=result,
            score=_mean(box.confidence for box in result.boxes),
            metadata={"model": self.config.get("model", "IDEA-Research/grounding-dino-base"), "labels": labels, "device": _device(self.config)},
        )


class SAMPlugin(VisionTaskPlugin):
    plugin_name = "sam"
    supported_tasks = {"object_detection", "segmentation"}

    def __init__(self, config=None):
        super().__init__(config)
        self._model = None
        self._processor = None

    @property
    def backend(self) -> str:
        return self.config.get("backend", "ultralytics_sam2")

    def _load(self):
        if self._model is not None:
            return
        backend = self.backend
        if backend == "official_sam3":
            try:
                from transformers import Sam3Model, Sam3Processor
            except ImportError as exc:
                raise RuntimeError("Install requirements-specialists.txt to use the SAM3 plugin") from exc
            model_name = self.config.get("model", "facebook/sam3")
            try:
                self._processor = Sam3Processor.from_pretrained(model_name)
                self._model = Sam3Model.from_pretrained(model_name)
            except OSError as exc:
                raise RuntimeError(
                    f"Cannot load SAM3 model '{model_name}'. "
                    "Request access to https://huggingface.co/facebook/sam3, run `hf auth login`, "
                    "or set plugins.json sam.config.model to a local downloaded SAM3 directory."
                ) from exc
            self._model.to(_device(self.config))
            self._model.eval()
            try:
                import torch
            except ImportError as exc:
                raise RuntimeError("Install requirements-specialists.txt to use the SAM3 plugin") from exc
            self._torch = torch
            return
        if backend == "ultralytics_sam2":
            try:
                from ultralytics import SAM
            except ImportError as exc:
                raise RuntimeError("Install requirements-specialists.txt to use the SAM2 plugin") from exc
            self._model = SAM(self.config.get("model", "sam2_b.pt"))
            return
        raise ValueError(f"Unsupported SAM backend: {backend}")

    def _segment_label(self, image, label: str, width: int, height: int) -> DetectionResult:
        inputs = self._processor(images=image, text=label, return_tensors="pt")
        target_sizes = inputs.get("original_sizes")
        if hasattr(inputs, "to"):
            inputs = inputs.to(_device(self.config))
        else:
            inputs = {
                key: value.to(_device(self.config)) if hasattr(value, "to") else value
                for key, value in inputs.items()
            }
        if target_sizes is None:
            target_sizes = [[height, width]]
        elif hasattr(target_sizes, "tolist"):
            target_sizes = target_sizes.tolist()
        with self._torch.no_grad():
            outputs = self._model(**inputs)
        processed = self._processor.post_process_instance_segmentation(
            outputs,
            threshold=float(self.config.get("threshold", 0.5)),
            mask_threshold=float(self.config.get("mask_threshold", 0.5)),
            target_sizes=target_sizes,
        )[0]

        result = DetectionResult(task_type="segmentation")
        masks = processed.get("masks", [])
        boxes = processed.get("boxes", [])
        scores = processed.get("scores", [])
        max_instances = int(self.config.get("max_instances_per_label", 0))
        for index, mask in enumerate(masks):
            if max_instances and index >= max_instances:
                break
            box_values = _value_at(boxes, index)
            score = float(_value_at(scores, index, 1.0))
            if box_values is not None and len(box_values) == 4:
                x1, y1, x2, y2 = [float(value) for value in box_values]
                result.boxes.append(
                    BoundingBox(
                        label=label,
                        xmin=max(0.0, min(1.0, x1 / width)),
                        ymin=max(0.0, min(1.0, y1 / height)),
                        xmax=max(0.0, min(1.0, x2 / width)),
                        ymax=max(0.0, min(1.0, y2 / height)),
                        confidence=score,
                    )
                )
            for polygon in _mask_to_polygons(mask, width, height):
                result.segments.append(PolygonSegment(label=label, polygon=polygon, confidence=score))
        return result

    def _refine_sam3(self, image_path, prompt, seed_result):
        self._load()
        labels = list(configured_labels(self.config, seed_result))
        if not labels:
            raise ValueError("SAM3 requires config.labels, classes_path, prompt names block, or seed labels")
        labels = labels[: int(self.config.get("max_labels", 0)) or None]
        image = Image.open(image_path).convert("RGB")
        width, height = image.size
        result = DetectionResult(task_type="segmentation")
        for label in labels:
            label_result = self._segment_label(image, label, width, height)
            result.boxes.extend(label_result.boxes)
            result.segments.extend(label_result.segments)
        return PluginOutput(
            result=result,
            score=_mean(segment.confidence for segment in result.segments),
            metadata={
                "model": self.config.get("model", "facebook/sam3"),
                "labels": labels,
                "masks": len(result.segments),
                "loader": "transformers",
                "device": _device(self.config),
            },
        )

    def _refine_sam2(self, image_path, prompt, seed_result):
        self._load()
        width, height = Image.open(image_path).size
        boxes = seed_result.boxes
        if not boxes:
            raise ValueError("SAM2 requires seed bounding boxes; run Grounding DINO before SAM2")
        pixel_boxes = [[box.xmin * width, box.ymin * height, box.xmax * width, box.ymax * height] for box in boxes]
        predictions = self._model.predict(
            image_path,
            bboxes=pixel_boxes,
            device=_device(self.config),
            verbose=False,
        )
        result = DetectionResult(task_type="segmentation")
        polygons = []
        for prediction in predictions:
            if prediction.masks is not None:
                polygons.extend(prediction.masks.xy)
        for index, polygon in enumerate(polygons):
            if len(polygon) < 3:
                continue
            source_box = boxes[min(index, len(boxes) - 1)]
            points = [Point(x=float(x) / width, y=float(y) / height) for x, y in polygon]
            result.segments.append(
                PolygonSegment(label=source_box.label, polygon=points, confidence=source_box.confidence)
            )
        return PluginOutput(
            result=result,
            score=_mean(segment.confidence for segment in result.segments),
            metadata={
                "model": self.config.get("model", "sam2_b.pt"),
                "masks": len(result.segments),
                "backend": "ultralytics_sam2",
                "device": _device(self.config),
            },
        )

    def refine(self, image_path, prompt, seed_result):
        if self.backend == "official_sam3":
            return self._refine_sam3(image_path, prompt, seed_result)
        return self._refine_sam2(image_path, prompt, seed_result)


class GroundedSAM2Plugin(VisionTaskPlugin):
    plugin_name = "grounded_sam2"
    supported_tasks = {"segmentation"}

    def __init__(self, config=None):
        super().__init__(config)
        self._grounding = None
        self._sam = None

    def _grounding_config(self) -> dict:
        return {
            "model": self.config.get("grounding_model", self.config.get("model", "IDEA-Research/grounding-dino-base")),
            "device": self.config.get("device", "auto"),
            "labels": self.config.get("labels", []),
            "box_threshold": self.config.get("box_threshold", 0.45),
            "text_threshold": self.config.get("text_threshold", 0.30),
            "merge_iou": self.config.get("merge_iou", 0.35),
            "nms_iou": self.config.get("nms_iou", 0.60),
            "min_confidence": self.config.get("min_confidence", 0.20),
        }

    def _sam_config(self) -> dict:
        return {
            "backend": self.config.get("sam_backend", self.config.get("backend", "ultralytics_sam2")),
            "model": self.config.get("sam_model", "sam2_b.pt"),
            "device": self.config.get("device", "auto"),
            "labels": self.config.get("labels", []),
            "threshold": self.config.get("threshold", 0.5),
            "mask_threshold": self.config.get("mask_threshold", 0.5),
        }

    def _load(self):
        if self._grounding is None:
            self._grounding = GroundingDINOPlugin(self._grounding_config())
            self._grounding._load()
        if self._sam is None:
            self._sam = SAMPlugin(self._sam_config())
            self._sam._load()

    def refine(self, image_path, prompt, seed_result):
        self._load()
        self._grounding.config = self._grounding_config()
        self._sam.config = self._sam_config()
        grounding_output = self._grounding.refine(image_path, prompt, seed_result)
        sam_output = self._sam.refine(image_path, prompt, grounding_output.result)
        result = DetectionResult(task_type="segmentation")
        result.boxes.extend(grounding_output.result.boxes)
        result.segments.extend(sam_output.result.segments)
        result.boxes.extend(sam_output.result.boxes)
        return PluginOutput(
            result=result,
            score=sam_output.score if sam_output.score is not None else grounding_output.score,
            metadata={
                "grounding": grounding_output.metadata,
                "sam": sam_output.metadata,
                "boxes": len(grounding_output.result.boxes),
                "segments": len(sam_output.result.segments),
                "pipeline": "grounding_dino+sam2",
            },
        )


class UltralyticsPosePlugin(VisionTaskPlugin):
    plugin_name = "pose"
    supported_tasks = {"pose_estimation"}

    def __init__(self, config=None):
        super().__init__(config)
        self._model = None

    def _load(self):
        if self._model is None:
            try:
                from ultralytics import YOLO
            except ImportError as exc:
                raise RuntimeError("Install requirements-specialists.txt to use the pose plugin") from exc
            self._model = YOLO(self.config.get("model", "yolo11n-pose.pt"))

    def refine(self, image_path, prompt, seed_result):
        self._load()
        predictions = self._model.predict(image_path, device=_device(self.config), verbose=False)
        width, height = Image.open(image_path).size
        names: List[str] = self.config.get("keypoint_names") or []
        result = DetectionResult(task_type="pose_estimation")
        for prediction in predictions:
            if prediction.keypoints is None:
                continue
            xy = prediction.keypoints.xy.cpu().tolist()
            conf = prediction.keypoints.conf.cpu().tolist() if prediction.keypoints.conf is not None else []
            box_conf = prediction.boxes.conf.cpu().tolist() if prediction.boxes is not None else []
            for pose_index, points in enumerate(xy):
                keypoints = []
                for index, (x, y) in enumerate(points):
                    point_conf = float(conf[pose_index][index]) if conf else 1.0
                    keypoints.append(
                        Keypoint(
                            name=names[index] if index < len(names) else f"keypoint_{index}",
                            x=float(x) / width,
                            y=float(y) / height,
                            visible=point_conf > float(self.config.get("keypoint_threshold", 0.25)),
                            confidence=point_conf,
                        )
                    )
                result.poses.append(
                    PoseInstance(
                        label=self.config.get("label", "person"),
                        keypoints=keypoints,
                        confidence=float(box_conf[pose_index]) if pose_index < len(box_conf) else 1.0,
                    )
                )
        return PluginOutput(result=result, score=_mean(pose.confidence for pose in result.poses), metadata={"model": self.config.get("model", "yolo11n-pose.pt"), "device": _device(self.config)})


class OCRPlugin(VisionTaskPlugin):
    plugin_name = "ocr"
    supported_tasks = {"ocr"}

    def __init__(self, config=None):
        super().__init__(config)
        self._engine = None

    @property
    def backend(self) -> str:
        return str(self.config.get("backend", "paddleocr")).lower()

    def _load(self):
        if self._engine is not None:
            return
        if self.backend == "easyocr":
            try:
                import easyocr
            except ImportError as exc:
                raise RuntimeError("Install requirements-specialists.txt to use the EasyOCR backend") from exc
            self._engine = easyocr.Reader(self.config.get("languages", ["en"]), gpu=_easyocr_gpu(self.config))
            return

        try:
            from paddleocr import PaddleOCR
        except ImportError as exc:
            raise RuntimeError("Install requirements-specialists.txt to use the PaddleOCR backend") from exc

        use_gpu = _easyocr_gpu(self.config)
        lang = _paddleocr_lang(self.config)
        attempts = [
            {"use_angle_cls": True, "lang": lang, "use_gpu": use_gpu, "show_log": False},
            {"use_angle_cls": True, "lang": lang, "use_gpu": use_gpu},
            {"use_textline_orientation": True, "lang": lang},
            {"lang": lang},
        ]
        last_error = None
        for kwargs in attempts:
            try:
                self._engine = PaddleOCR(**kwargs)
                return
            except TypeError as exc:
                last_error = exc
        raise RuntimeError(f"Failed to initialize PaddleOCR backend: {last_error}") from last_error

    def _read_easyocr(self, image_path, width: int, height: int) -> DetectionResult:
        result = DetectionResult(task_type="ocr")
        for polygon, text, confidence in self._engine.readtext(image_path):
            self._append_text_region(result, polygon, text, confidence, width, height)
        return result

    def _read_paddleocr(self, image_path, width: int, height: int) -> DetectionResult:
        result = DetectionResult(task_type="ocr")
        if hasattr(self._engine, "ocr"):
            try:
                raw = self._engine.ocr(image_path, cls=True)
            except TypeError:
                raw = self._engine.ocr(image_path)
        elif hasattr(self._engine, "predict"):
            raw = self._engine.predict(image_path)
        else:
            raw = []
        for item in self._iter_paddleocr_items(raw):
            polygon, text, confidence = item
            self._append_text_region(result, polygon, text, confidence, width, height)
        return result

    def _iter_paddleocr_items(self, raw):
        if raw is None:
            return
        if isinstance(raw, dict):
            boxes = raw.get("dt_polys") or raw.get("rec_boxes") or raw.get("boxes") or []
            texts = raw.get("rec_texts") or raw.get("texts") or []
            scores = raw.get("rec_scores") or raw.get("scores") or []
            for index, polygon in enumerate(boxes):
                yield polygon, texts[index] if index < len(texts) else "", scores[index] if index < len(scores) else 0.0
            return
        if not isinstance(raw, (list, tuple)):
            return
        if len(raw) == 2 and isinstance(raw[0], (list, tuple)) and all(isinstance(value, str) for value in raw[0]):
            return
        for entry in raw:
            if isinstance(entry, dict):
                yield from self._iter_paddleocr_items(entry)
            elif isinstance(entry, (list, tuple)) and entry:
                first = entry[0]
                if self._looks_like_polygon(first):
                    text = ""
                    confidence = 0.0
                    if len(entry) > 1:
                        text_info = entry[1]
                        if isinstance(text_info, (list, tuple)) and text_info:
                            text = text_info[0]
                            confidence = text_info[1] if len(text_info) > 1 else 0.0
                        else:
                            text = text_info
                    yield first, text, confidence
                else:
                    yield from self._iter_paddleocr_items(entry)

    @staticmethod
    def _looks_like_polygon(value) -> bool:
        if not isinstance(value, (list, tuple)) or not value:
            return False
        point = value[0]
        if not isinstance(point, (list, tuple)) or len(point) < 2:
            return False
        return isinstance(point[0], (int, float)) and isinstance(point[1], (int, float))

    @staticmethod
    def _append_text_region(result: DetectionResult, polygon, text, confidence, width: int, height: int) -> None:
        points = [point for point in polygon if isinstance(point, (list, tuple)) and len(point) >= 2]
        if not points:
            return
        xs = [float(point[0]) for point in points]
        ys = [float(point[1]) for point in points]
        result.texts.append(
            TextRegion(
                text=str(text),
                xmin=max(0.0, min(1.0, min(xs) / width)),
                ymin=max(0.0, min(1.0, min(ys) / height)),
                xmax=max(0.0, min(1.0, max(xs) / width)),
                ymax=max(0.0, min(1.0, max(ys) / height)),
                confidence=float(confidence or 0.0),
            )
        )

    def refine(self, image_path, prompt, seed_result):
        self._load()
        width, height = Image.open(image_path).size
        if self.backend == "easyocr":
            result = self._read_easyocr(image_path, width, height)
        else:
            result = self._read_paddleocr(image_path, width, height)
        return PluginOutput(
            result=result,
            score=_mean(item.confidence for item in result.texts),
            metadata={
                "backend": self.backend,
                "languages": self.config.get("languages", ["ko", "en"]),
                "lang": _paddleocr_lang(self.config) if self.backend == "paddleocr" else None,
                "gpu": _easyocr_gpu(self.config),
            },
        )


EasyOCRPlugin = OCRPlugin


class ViTPosePlugin(VisionTaskPlugin):
    plugin_name = "vitpose"
    supported_tasks = {"pose_estimation"}

    COCO_KEYPOINTS = [
        "nose", "left_eye", "right_eye", "left_ear", "right_ear",
        "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
        "left_wrist", "right_wrist", "left_hip", "right_hip",
        "left_knee", "right_knee", "left_ankle", "right_ankle"
    ]

    MPII_KEYPOINTS = [
        "right_ankle", "right_knee", "right_hip", "left_hip",
        "left_knee", "left_ankle", "pelvis", "thorax",
        "upper_neck", "head_top", "right_wrist", "right_elbow",
        "right_shoulder", "left_shoulder", "left_elbow", "left_wrist"
    ]

    def __init__(self, config=None):
        super().__init__(config)
        self._model = None
        self._detector = None

    def _get_keypoint_names(self, prompt: str) -> list:
        prompt_lower = prompt.lower() if prompt else ""
        if "mpii" in prompt_lower:
            return self.MPII_KEYPOINTS
        return self.COCO_KEYPOINTS

    def _load(self):
        if self._model is not None:
            return
        try:
            from mmpose.apis import inference_topdown, init_model
            from mmdet.apis import inference_detector, init_detector
        except ImportError as exc:
            raise RuntimeError("Install requirements-specialists.txt to use the ViTPose plugin") from exc

        device = _device(self.config)

        pose_config = self.config.get("pose_config", "configs/body_2d_keypoint/topdown_heatmap/coco/td-hm_ViTPose-large_8xb64-210e_coco-256x192.py")
        pose_checkpoint = self.config.get("pose_checkpoint", "vitpose-l.pth")

        det_config = self.config.get("det_config", "demo/mmdetection_cfg/rtmdet_m_640-8xb32_coco-person.py")
        det_checkpoint = self.config.get("det_checkpoint", "https://download.openmmlab.com/mmpose/v1/projects/rtmpose/rtmdet_m_8xb32-100e_coco-obj365-person-235e8209.pth")

        self._detector = init_detector(det_config, det_checkpoint, device=device)
        self._model = init_model(pose_config, pose_checkpoint, device=device)

    def refine(self, image_path, prompt, seed_result):
        self._load()

        try:
            from mmpose.apis import inference_topdown
            from mmdet.apis import inference_detector
            import mmcv
        except ImportError as exc:
            raise RuntimeError("Install requirements-specialists.txt to use the ViTPose plugin") from exc

        image = mmcv.imread(image_path)
        height, width = image.shape[:2]

        det_results = inference_detector(self._detector, image)
        pred_instances = det_results.pred_instances

        bboxes = []
        if hasattr(pred_instances, 'bboxes') and hasattr(pred_instances, 'scores'):
            det_threshold = float(self.config.get("det_threshold", 0.5))
            for bbox, score in zip(pred_instances.bboxes, pred_instances.scores):
                if score > det_threshold:
                    bboxes.append({'bbox': bbox.cpu().numpy()})

        if not bboxes:
            return PluginOutput(
                result=DetectionResult(task_type="pose_estimation"),
                score=0.0,
                metadata={"model": "vitpose", "device": _device(self.config), "message": "No person detected"}
            )

        pose_results = inference_topdown(self._model, image, bboxes)

        keypoint_names = self._get_keypoint_names(prompt)
        keypoint_threshold = float(self.config.get("keypoint_threshold", 0.5))

        result = DetectionResult(task_type="pose_estimation")

        for pose_result in pose_results:
            if not hasattr(pose_result, 'pred_instances'):
                continue

            pred = pose_result.pred_instances
            keypoints_data = pred.keypoints[0] if len(pred.keypoints) > 0 else None
            keypoint_scores = pred.keypoint_scores[0] if len(pred.keypoint_scores) > 0 else None

            if keypoints_data is None:
                continue

            keypoints = []
            for idx, (x, y) in enumerate(keypoints_data):
                confidence = float(keypoint_scores[idx]) if keypoint_scores is not None else 1.0
                keypoints.append(
                    Keypoint(
                        name=keypoint_names[idx] if idx < len(keypoint_names) else f"keypoint_{idx}",
                        x=float(x) / width,
                        y=float(y) / height,
                        visible=confidence > keypoint_threshold,
                        confidence=confidence,
                    )
                )

            pose_confidence = float(keypoint_scores.mean()) if keypoint_scores is not None else 1.0
            result.poses.append(
                PoseInstance(
                    label=self.config.get("label", "person"),
                    keypoints=keypoints,
                    confidence=pose_confidence,
                )
            )

        return PluginOutput(
            result=result,
            score=_mean(pose.confidence for pose in result.poses),
            metadata={
                "model": self.config.get("pose_checkpoint", "vitpose-l.pth"),
                "device": _device(self.config),
                "keypoint_format": "MPII" if "mpii" in prompt.lower() else "COCO",
                "num_poses": len(result.poses),
            }
        )


class UltralyticsTrackingPlugin(VisionTaskPlugin):
    plugin_name = "tracking"
    supported_tasks = {"tracking"}

    def __init__(self, config=None):
        super().__init__(config)
        self._model = None
        self._frame_id = 0

    def _load(self):
        if self._model is None:
            try:
                from ultralytics import YOLO
            except ImportError as exc:
                raise RuntimeError("Install requirements-specialists.txt to use the tracking plugin") from exc
            self._model = YOLO(self.config.get("model", "yolo11n.pt"))

    def refine(self, image_path, prompt, seed_result):
        self._load()
        predictions = self._model.track(
            image_path,
            persist=True,
            tracker=self.config.get("tracker", "bytetrack.yaml"),
            device=_device(self.config),
            verbose=False,
        )
        width, height = Image.open(image_path).size
        result = DetectionResult(task_type="tracking")
        for prediction in predictions:
            if prediction.boxes is None:
                continue
            xyxy = prediction.boxes.xyxy.cpu().tolist()
            confidences = prediction.boxes.conf.cpu().tolist()
            classes = prediction.boxes.cls.cpu().tolist()
            ids = prediction.boxes.id.cpu().tolist() if prediction.boxes.id is not None else range(len(xyxy))
            for box, confidence, class_id, track_id in zip(xyxy, confidences, classes, ids):
                x1, y1, x2, y2 = box
                result.tracks.append(
                    TrackInstance(
                        track_id=str(int(track_id)),
                        frame_id=self._frame_id,
                        label=prediction.names.get(int(class_id), str(int(class_id))),
                        xmin=x1 / width,
                        ymin=y1 / height,
                        xmax=x2 / width,
                        ymax=y2 / height,
                        confidence=float(confidence),
                    )
                )
        self._frame_id += 1
        return PluginOutput(result=result, score=_mean(item.confidence for item in result.tracks), metadata={"model": self.config.get("model", "yolo11n.pt"), "frame_id": self._frame_id - 1, "device": _device(self.config)})
