import json
import os
import xml.etree.ElementTree as ET
from typing import Dict, Iterable, List, Optional

from PIL import Image

from ..core.models import BoundingBox, DetectionResult

SUPPORTED_LABEL_FORMATS = {"yolo", "pascal_voc", "coco", "custom", "vision_json"}


def normalize_label_formats(value: str | Iterable[str]) -> List[str]:
    if isinstance(value, str):
        raw_formats = [part.strip().lower() for part in value.split(",")]
    else:
        raw_formats = [str(part).strip().lower() for part in value]

    requested = [fmt for fmt in raw_formats if fmt]
    if not requested:
        return ["yolo"]
    if "all" in requested:
        return ["yolo", "pascal_voc", "coco", "vision_json"]

    unsupported = sorted(set(requested) - SUPPORTED_LABEL_FORMATS)
    if unsupported:
        raise ValueError(
            f"Unsupported label format(s): {', '.join(unsupported)}. "
            f"Supported formats: {', '.join(sorted(SUPPORTED_LABEL_FORMATS))}, all"
        )
    return list(dict.fromkeys(requested))


def get_image_size(image_path: str) -> tuple[int, int]:
    with Image.open(image_path) as img:
        return img.size


def normalized_to_pixel_box(box: BoundingBox, width: int, height: int) -> dict:
    xmin = max(0, min(width, round(box.xmin * width)))
    ymin = max(0, min(height, round(box.ymin * height)))
    xmax = max(0, min(width, round(box.xmax * width)))
    ymax = max(0, min(height, round(box.ymax * height)))
    return {
        "xmin": xmin,
        "ymin": ymin,
        "xmax": xmax,
        "ymax": ymax,
        "width": max(0, xmax - xmin),
        "height": max(0, ymax - ymin),
    }


def box_to_export_dict(box: BoundingBox, class_id: int, image_width: int, image_height: int) -> dict:
    pixel_box = normalized_to_pixel_box(box, image_width, image_height)
    x_center = (box.xmin + box.xmax) / 2
    y_center = (box.ymin + box.ymax) / 2
    width = box.xmax - box.xmin
    height = box.ymax - box.ymin
    return {
        "label": box.label,
        "class_id": class_id,
        "confidence": box.confidence,
        "normalized": {
            "xmin": box.xmin,
            "ymin": box.ymin,
            "xmax": box.xmax,
            "ymax": box.ymax,
        },
        "pixel": pixel_box,
        "yolo": {
            "class_id": class_id,
            "x_center": x_center,
            "y_center": y_center,
            "width": width,
            "height": height,
        },
        "coco": {
            "category_id": class_id + 1,
            "bbox": [
                pixel_box["xmin"],
                pixel_box["ymin"],
                pixel_box["width"],
                pixel_box["height"],
            ],
            "area": pixel_box["width"] * pixel_box["height"],
        },
    }

def point_to_pixel(point, width: int, height: int) -> dict:
    return {
        "x": round(point.x * width),
        "y": round(point.y * height),
    }

def result_to_export_dict(result: DetectionResult, image_path: str) -> dict:
    width, height = get_image_size(image_path)
    return {
        "image_name": os.path.basename(image_path),
        "image_path": image_path,
        "image_width": width,
        "image_height": height,
        "task_type": result.task_type,
        "source_model": result.source_model,
        "consistency_score": result.consistency_score,
        "mean_confidence": result.mean_confidence,
        "uncertainty_score": result.uncertainty_score,
        "plugin_scores": result.plugin_scores,
        "plugin_metadata": result.plugin_metadata,
        "classifications": [item.model_dump() for item in result.classifications],
        "boxes": [box_to_export_dict(box, 0, width, height) for box in result.boxes],
        "segments": [
            {
                "label": segment.label,
                "confidence": segment.confidence,
                "polygon": [point.model_dump() for point in segment.polygon],
                "polygon_pixels": [point_to_pixel(point, width, height) for point in segment.polygon],
            }
            for segment in result.segments
        ],
        "poses": [pose.model_dump() for pose in result.poses],
        "texts": [
            {
                "text": text.text,
                "confidence": text.confidence,
                "normalized": {
                    "xmin": text.xmin,
                    "ymin": text.ymin,
                    "xmax": text.xmax,
                    "ymax": text.ymax,
                },
                "pixel": normalized_to_pixel_box(text, width, height),
            }
            for text in result.texts
        ],
        "tracks": [
            {
                "track_id": track.track_id,
                "frame_id": track.frame_id,
                "label": track.label,
                "confidence": track.confidence,
                "normalized": {
                    "xmin": track.xmin,
                    "ymin": track.ymin,
                    "xmax": track.xmax,
                    "ymax": track.ymax,
                },
                "pixel": normalized_to_pixel_box(track, width, height),
            }
            for track in result.tracks
        ],
    }

def save_as_yolo(result: DetectionResult, image_name: str, output_dir: str, class_list: list):
    """
    Saves detection results in YOLO format (.txt).
    Format: <class_id> <x_center> <y_center> <width> <height> (normalized 0-1)
    """
    os.makedirs(output_dir, exist_ok=True)
    txt_name = os.path.splitext(image_name)[0] + ".txt"
    txt_path = os.path.join(output_dir, txt_name)
    
    with open(txt_path, "w") as f:
        for box in result.boxes:
            if box.label not in class_list:
                class_list.append(box.label)
            
            class_id = class_list.index(box.label)
            
            # Convert [xmin, ymin, xmax, ymax] to [x_center, y_center, width, height]
            x_center = (box.xmin + box.xmax) / 2
            y_center = (box.ymin + box.ymax) / 2
            width = box.xmax - box.xmin
            height = box.ymax - box.ymin
            
            f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
    
    return txt_path


def save_as_pascal_voc(result: DetectionResult, image_path: str, output_dir: str, class_list: list) -> str:
    os.makedirs(output_dir, exist_ok=True)
    width, height = get_image_size(image_path)
    image_name = os.path.basename(image_path)
    xml_path = os.path.join(output_dir, os.path.splitext(image_name)[0] + ".xml")

    annotation = ET.Element("annotation")
    ET.SubElement(annotation, "folder").text = os.path.basename(os.path.dirname(image_path))
    ET.SubElement(annotation, "filename").text = image_name
    ET.SubElement(annotation, "path").text = os.path.abspath(image_path)

    source = ET.SubElement(annotation, "source")
    ET.SubElement(source, "database").text = "Unknown"

    size = ET.SubElement(annotation, "size")
    ET.SubElement(size, "width").text = str(width)
    ET.SubElement(size, "height").text = str(height)
    ET.SubElement(size, "depth").text = "3"
    ET.SubElement(annotation, "segmented").text = "0"

    for box in result.boxes:
        if box.label not in class_list:
            class_list.append(box.label)
        pixel_box = normalized_to_pixel_box(box, width, height)

        obj = ET.SubElement(annotation, "object")
        ET.SubElement(obj, "name").text = box.label
        ET.SubElement(obj, "pose").text = "Unspecified"
        ET.SubElement(obj, "truncated").text = "0"
        ET.SubElement(obj, "difficult").text = "0"
        ET.SubElement(obj, "confidence").text = f"{box.confidence:.6f}"
        bndbox = ET.SubElement(obj, "bndbox")
        ET.SubElement(bndbox, "xmin").text = str(pixel_box["xmin"])
        ET.SubElement(bndbox, "ymin").text = str(pixel_box["ymin"])
        ET.SubElement(bndbox, "xmax").text = str(pixel_box["xmax"])
        ET.SubElement(bndbox, "ymax").text = str(pixel_box["ymax"])

    tree = ET.ElementTree(annotation)
    ET.indent(tree, space="  ")
    tree.write(xml_path, encoding="utf-8", xml_declaration=True)
    return xml_path


class LabelExportWriter:
    def __init__(
        self,
        output_dir: str,
        formats: str | Iterable[str] = "yolo",
        custom_template_path: Optional[str] = None,
        custom_extension: str = ".json",
    ):
        self.output_dir = output_dir
        self.formats = normalize_label_formats(formats)
        if custom_template_path and "custom" not in self.formats:
            self.formats.append("custom")
        if "custom" in self.formats and not custom_template_path:
            raise ValueError("--custom_label_template is required when using the custom label format")

        self.class_list: List[str] = []
        self.custom_template = None
        if custom_template_path:
            with open(custom_template_path, "r", encoding="utf-8") as f:
                self.custom_template = f.read()
        self.custom_extension = custom_extension if custom_extension.startswith(".") else f".{custom_extension}"
        self.coco_images = []
        self.coco_annotations = []
        self.vision_json_records = []
        self._next_image_id = 1
        self._next_annotation_id = 1
        os.makedirs(output_dir, exist_ok=True)

    def save(self, result: DetectionResult, image_path: str) -> Dict[str, str]:
        image_name = os.path.basename(image_path)
        paths = {}
        if "yolo" in self.formats:
            paths["yolo"] = save_as_yolo(result, image_name, self.output_dir, self.class_list)
        if "pascal_voc" in self.formats:
            paths["pascal_voc"] = save_as_pascal_voc(result, image_path, self.output_dir, self.class_list)
        if "coco" in self.formats:
            self._add_coco_image(result, image_path)
            paths["coco"] = os.path.join(self.output_dir, "coco_annotations.json")
        if "vision_json" in self.formats:
            self.vision_json_records.append(result_to_export_dict(result, image_path))
            paths["vision_json"] = os.path.join(self.output_dir, "vision_annotations.jsonl")
        if "custom" in self.formats:
            paths["custom"] = self._save_custom(result, image_path)
        return paths

    def finalize(self) -> Dict[str, str]:
        paths = {}
        if self.class_list:
            classes_path = os.path.join(self.output_dir, "classes.txt")
            with open(classes_path, "w", encoding="utf-8") as f:
                f.write("\n".join(self.class_list))
            paths["classes"] = classes_path
        if "coco" in self.formats:
            coco_path = os.path.join(self.output_dir, "coco_annotations.json")
            categories = [
                {"id": idx + 1, "name": label, "supercategory": "object"}
                for idx, label in enumerate(self.class_list)
            ]
            data = {
                "info": {"description": "Auto-generated labels"},
                "licenses": [],
                "images": self.coco_images,
                "annotations": self.coco_annotations,
                "categories": categories,
            }
            with open(coco_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            paths["coco"] = coco_path
        if "vision_json" in self.formats:
            jsonl_path = os.path.join(self.output_dir, "vision_annotations.jsonl")
            with open(jsonl_path, "w", encoding="utf-8") as f:
                for record in self.vision_json_records:
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
            paths["vision_json"] = jsonl_path
        return paths

    def _class_id(self, label: str) -> int:
        if label not in self.class_list:
            self.class_list.append(label)
        return self.class_list.index(label)

    def _export_objects(self, result: DetectionResult, image_width: int, image_height: int) -> List[dict]:
        objects = []
        for box in result.boxes:
            class_id = self._class_id(box.label)
            objects.append(box_to_export_dict(box, class_id, image_width, image_height))
        return objects

    def _add_coco_image(self, result: DetectionResult, image_path: str) -> None:
        width, height = get_image_size(image_path)
        image_id = self._next_image_id
        self._next_image_id += 1
        self.coco_images.append(
            {
                "id": image_id,
                "file_name": os.path.basename(image_path),
                "width": width,
                "height": height,
            }
        )

        for obj in self._export_objects(result, width, height):
            self.coco_annotations.append(
                {
                    "id": self._next_annotation_id,
                    "image_id": image_id,
                    "category_id": obj["coco"]["category_id"],
                    "bbox": obj["coco"]["bbox"],
                    "area": obj["coco"]["area"],
                    "iscrowd": 0,
                    "segmentation": [],
                    "score": obj["confidence"],
                }
            )
            self._next_annotation_id += 1

        for segment in result.segments:
            class_id = self._class_id(segment.label)
            polygon = []
            for point in segment.polygon:
                polygon.extend([round(point.x * width), round(point.y * height)])
            xs = polygon[0::2]
            ys = polygon[1::2]
            if not xs or not ys:
                continue
            bbox = [min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys)]
            self.coco_annotations.append(
                {
                    "id": self._next_annotation_id,
                    "image_id": image_id,
                    "category_id": class_id + 1,
                    "bbox": bbox,
                    "area": bbox[2] * bbox[3],
                    "iscrowd": 0,
                    "segmentation": [polygon],
                    "score": segment.confidence,
                }
            )
            self._next_annotation_id += 1

    def _save_custom(self, result: DetectionResult, image_path: str) -> str:
        width, height = get_image_size(image_path)
        image_name = os.path.basename(image_path)
        objects = self._export_objects(result, width, height)
        payload = {
            "image_name": image_name,
            "image_path": image_path,
            "image_width": width,
            "image_height": height,
            "source_model": result.source_model or "",
            "consistency_score": result.consistency_score if result.consistency_score is not None else "",
            "mean_confidence": result.mean_confidence if result.mean_confidence is not None else "",
            "uncertainty_score": result.uncertainty_score if result.uncertainty_score is not None else "",
            "object_count": len(objects),
            "objects_json": json.dumps(objects, ensure_ascii=False),
            "boxes_json": json.dumps([obj["normalized"] for obj in objects], ensure_ascii=False),
            "labels_json": json.dumps([obj["label"] for obj in objects], ensure_ascii=False),
            "result_json": json.dumps(result_to_export_dict(result, image_path), ensure_ascii=False),
        }
        content = self.custom_template.format(**payload)
        custom_path = os.path.join(
            self.output_dir,
            os.path.splitext(image_name)[0] + self.custom_extension,
        )
        with open(custom_path, "w", encoding="utf-8") as f:
            f.write(content)
        return custom_path
