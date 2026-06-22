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


def _device_index(device: str) -> int:
    if device.startswith("cuda"):
        parts = device.split(":", 1)
        return int(parts[1]) if len(parts) == 2 else 0
    return -1


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
                device=_device_index(self.config.get("device", "cpu")),
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
        model_id = self.config.get("model", "IDEA-Research/grounding-dino-tiny")
        self._processor = AutoProcessor.from_pretrained(model_id)
        self._model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id)
        self._model.to(self.config.get("device", "cpu"))
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
        inputs = {key: value.to(self.config.get("device", "cpu")) for key, value in inputs.items()}
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
            metadata={"model": self.config.get("model", "IDEA-Research/grounding-dino-tiny"), "labels": labels},
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
            self._model.to(self.config.get("device", "cpu"))
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
            inputs = inputs.to(self.config.get("device", "cpu"))
        else:
            inputs = {
                key: value.to(self.config.get("device", "cpu")) if hasattr(value, "to") else value
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
            device=self.config.get("device"),
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
            },
        )

    def refine(self, image_path, prompt, seed_result):
        if self.backend == "official_sam3":
            return self._refine_sam3(image_path, prompt, seed_result)
        return self._refine_sam2(image_path, prompt, seed_result)


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
        predictions = self._model.predict(image_path, device=self.config.get("device"), verbose=False)
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
        return PluginOutput(result=result, score=_mean(pose.confidence for pose in result.poses), metadata={"model": self.config.get("model", "yolo11n-pose.pt")})


class EasyOCRPlugin(VisionTaskPlugin):
    plugin_name = "ocr"
    supported_tasks = {"ocr"}

    def __init__(self, config=None):
        super().__init__(config)
        self._reader = None

    def _load(self):
        if self._reader is None:
            try:
                import easyocr
            except ImportError as exc:
                raise RuntimeError("Install requirements-specialists.txt to use the OCR plugin") from exc
            self._reader = easyocr.Reader(self.config.get("languages", ["en"]), gpu=self.config.get("gpu", False))

    def refine(self, image_path, prompt, seed_result):
        self._load()
        width, height = Image.open(image_path).size
        result = DetectionResult(task_type="ocr")
        for polygon, text, confidence in self._reader.readtext(image_path):
            xs = [point[0] for point in polygon]
            ys = [point[1] for point in polygon]
            result.texts.append(
                TextRegion(
                    text=text,
                    xmin=min(xs) / width,
                    ymin=min(ys) / height,
                    xmax=max(xs) / width,
                    ymax=max(ys) / height,
                    confidence=float(confidence),
                )
            )
        return PluginOutput(result=result, score=_mean(item.confidence for item in result.texts), metadata={"languages": self.config.get("languages", ["en"])})


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
            device=self.config.get("device"),
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
        return PluginOutput(result=result, score=_mean(item.confidence for item in result.tracks), metadata={"model": self.config.get("model", "yolo11n.pt"), "frame_id": self._frame_id - 1})
