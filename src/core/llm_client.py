import os
import base64
import json
import re
import time
from dotenv import load_dotenv
from .models import (
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

load_dotenv()

def extract_json(text: str) -> dict:
    match = re.search(r'```json\s*(.*?)\s*```', text, re.DOTALL)
    if match:
        text = match.group(1)
    # Additional cleanup for robust parsing
    text = text.strip()
    if text.startswith('```') and text.endswith('```'):
        text = text[3:-3].strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end > start:
            try:
                return json.loads(text[start:end + 1])
            except json.JSONDecodeError:
                pass
        print(f"[Warning] Failed to parse JSON. Raw output: {text[:100]}...")
        return {"boxes": []}

def normalize_coordinate(value) -> float:
    coord = float(value)
    if coord > 1.0:
        coord = coord / 1000.0
    return max(0.0, min(1.0, coord))

def normalize_confidence(value) -> float:
    confidence = float(value)
    if confidence > 1.0:
        confidence = confidence / 100.0
    return max(0.0, min(1.0, confidence))

def normalize_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value > 0
    return str(value).strip().lower() in {"true", "1", "yes", "visible"}

def get_task_schema(task_type: str) -> str:
    schemas = {
        "classification": """
        Return image classification results strictly as JSON:
        {"classifications": [{"label": "class_name", "confidence": float}]}
        """,
        "object_detection": """
        Return object detection results strictly as JSON:
        {"boxes": [{"label": "class_name", "xmin": float, "ymin": float, "xmax": float, "ymax": float, "confidence": float}]}
        """,
        "segmentation": """
        Return polygon segmentation results strictly as JSON:
        {"segments": [{"label": "class_name", "polygon": [{"x": float, "y": float}], "confidence": float}]}
        """,
        "pose_estimation": """
        Return pose estimation results strictly as JSON:
        {"poses": [{"label": "person", "confidence": float, "keypoints": [{"name": "nose", "x": float, "y": float, "visible": bool, "confidence": float}]}]}
        """,
        "ocr": """
        Return OCR/text-region results strictly as JSON:
        {"texts": [{"text": "recognized text", "xmin": float, "ymin": float, "xmax": float, "ymax": float, "confidence": float}]}
        """,
        "tracking": """
        Return tracking labels strictly as JSON:
        {"tracks": [{"track_id": "id", "frame_id": int, "label": "class_name", "xmin": float, "ymin": float, "xmax": float, "ymax": float, "confidence": float}]}
        """,
        "all": """
        Return any visible labels strictly as JSON using these keys as applicable:
        {"classifications": [], "boxes": [], "segments": [], "poses": [], "texts": [], "tracks": []}
        boxes/texts/tracks use xmin, ymin, xmax, ymax. segments use polygon points. poses use keypoints.
        """,
    }
    return schemas.get(task_type, schemas["object_detection"])

def normalize_box_like(item: dict) -> dict:
    item["xmin"] = normalize_coordinate(item.get("xmin", 0.0))
    item["ymin"] = normalize_coordinate(item.get("ymin", 0.0))
    item["xmax"] = normalize_coordinate(item.get("xmax", 1.0))
    item["ymax"] = normalize_coordinate(item.get("ymax", 1.0))
    item["confidence"] = normalize_confidence(item.get("confidence", 1.0))
    return item

def parse_labeling_result(data: dict, task_type: str) -> DetectionResult:
    result = DetectionResult(task_type=task_type)

    for item in data.get("classifications", []):
        try:
            item["confidence"] = normalize_confidence(item.get("confidence", 1.0))
            if item.get("label"):
                result.classifications.append(ClassificationLabel(**item))
        except (TypeError, ValueError, KeyError):
            continue

    for item in data.get("boxes", []):
        try:
            item = normalize_box_like(item)
            if item.get("label") and item["xmin"] < item["xmax"] and item["ymin"] < item["ymax"]:
                result.boxes.append(BoundingBox(**item))
        except (TypeError, ValueError, KeyError):
            continue

    for item in data.get("segments", []):
        try:
            points = []
            for point in item.get("polygon", []):
                points.append(
                    Point(
                        x=normalize_coordinate(point.get("x", 0.0)),
                        y=normalize_coordinate(point.get("y", 0.0)),
                    )
                )
            if item.get("label") and len(points) >= 3:
                result.segments.append(
                    PolygonSegment(
                        label=item["label"],
                        polygon=points,
                        confidence=normalize_confidence(item.get("confidence", 1.0)),
                    )
                )
        except (TypeError, ValueError, KeyError):
            continue

    for item in data.get("poses", []):
        try:
            keypoints = []
            for point in item.get("keypoints", []):
                if not point.get("name"):
                    continue
                keypoints.append(
                    Keypoint(
                        name=point["name"],
                        x=normalize_coordinate(point.get("x", 0.0)),
                        y=normalize_coordinate(point.get("y", 0.0)),
                        visible=normalize_bool(point.get("visible", True)),
                        confidence=normalize_confidence(point.get("confidence", 1.0)),
                    )
                )
            if keypoints:
                result.poses.append(
                    PoseInstance(
                        label=item.get("label", "person"),
                        keypoints=keypoints,
                        confidence=normalize_confidence(item.get("confidence", 1.0)),
                    )
                )
        except (TypeError, ValueError, KeyError):
            continue

    for item in data.get("texts", []):
        try:
            item = normalize_box_like(item)
            if item.get("text") and item["xmin"] < item["xmax"] and item["ymin"] < item["ymax"]:
                result.texts.append(TextRegion(**item))
        except (TypeError, ValueError, KeyError):
            continue

    for item in data.get("tracks", []):
        try:
            item = normalize_box_like(item)
            if item.get("track_id") and item.get("label") and item["xmin"] < item["xmax"] and item["ymin"] < item["ymax"]:
                result.tracks.append(
                    TrackInstance(
                        track_id=str(item["track_id"]),
                        frame_id=int(item.get("frame_id", 0)),
                        label=item["label"],
                        xmin=item["xmin"],
                        ymin=item["ymin"],
                        xmax=item["xmax"],
                        ymax=item["ymax"],
                        confidence=item["confidence"],
                    )
                )
        except (TypeError, ValueError, KeyError):
            continue

    return result

class VisionLLMClient:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.max_retries = 3
        self.api_attempts = 0
        self.successful_predictions = 0
        self.failed_predictions = 0
        self.bedrock_model_id = model_name.removeprefix("bedrock:")
        
        if model_name.startswith("bedrock:") or model_name.startswith("anthropic.claude"):
            import boto3

            self.client = boto3.client(
                "bedrock-runtime",
                region_name=os.getenv("AWS_REGION") or os.getenv("AWS_DEFAULT_REGION", "us-east-1"),
            )
            self.provider = "bedrock"
        elif "gpt" in model_name.lower():
            from openai import OpenAI

            self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            self.provider = "openai"
        elif "claude" in model_name.lower():
            from anthropic import Anthropic

            self.client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
            self.provider = "anthropic"
        else:
            raise ValueError(
                f"Unsupported model: {model_name}. Use gpt-4o, gpt-4o-mini, "
                "claude-3-5-sonnet-20240620, or bedrock:<bedrock-model-id>."
            )

    def _get_media_type(self, image_path: str):
        ext = image_path.split('.')[-1].lower()
        if ext in ['jpg', 'jpeg']: return "image/jpeg"
        if ext == 'png': return "image/png"
        if ext == 'webp': return "image/webp"
        return "image/jpeg"

    def _encode_image(self, image_path: str):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    def predict(self, image_path: str, prompt: str, temperature: float = 0.0, task_type: str = "object_detection") -> DetectionResult:
        base64_image = self._encode_image(image_path)
        media_type = self._get_media_type(image_path)
        
        system_msg = f"""
        You are an expert vision labeling AI.
        {get_task_schema(task_type)}
        - Coordinates must be normalized floats between 0.0 and 1.0.
        - Box-like labels must satisfy xmin < xmax and ymin < ymax.
        - Output ONLY valid JSON, without any markdown formatting, preamble, or explanation.
        """

        for attempt in range(self.max_retries):
            try:
                self.api_attempts += 1
                if self.provider == "openai":
                    response = self.client.chat.completions.create(
                        model=self.model_name,
                        messages=[
                            {"role": "system", "content": system_msg},
                            {"role": "user", "content": [
                                {"type": "text", "text": prompt},
                                {"type": "image_url", "image_url": {"url": f"data:{media_type};base64,{base64_image}"}}
                            ]}
                        ],
                        response_format={"type": "json_object"},
                        temperature=temperature
                    )
                    content = response.choices[0].message.content

                elif self.provider == "anthropic":
                    response = self.client.messages.create(
                        model=self.model_name,
                        max_tokens=1024,
                        temperature=temperature,
                        system=system_msg,
                        messages=[
                            {
                                "role": "user",
                                "content": [
                                    {
                                        "type": "image",
                                        "source": {
                                            "type": "base64",
                                            "media_type": media_type,
                                            "data": base64_image,
                                        },
                                    },
                                    {"type": "text", "text": prompt}
                                ],
                            }
                        ],
                    )
                    content = response.content[0].text

                elif self.provider == "bedrock":
                    body = {
                        "anthropic_version": "bedrock-2023-05-31",
                        "max_tokens": 1024,
                        "temperature": temperature,
                        "system": system_msg,
                        "messages": [
                            {
                                "role": "user",
                                "content": [
                                    {
                                        "type": "image",
                                        "source": {
                                            "type": "base64",
                                            "media_type": media_type,
                                            "data": base64_image,
                                        },
                                    },
                                    {"type": "text", "text": prompt},
                                ],
                            }
                        ],
                    }
                    response = self.client.invoke_model(
                        modelId=self.bedrock_model_id,
                        body=json.dumps(body),
                        contentType="application/json",
                        accept="application/json",
                    )
                    response_body = json.loads(response["body"].read())
                    content = response_body["content"][0]["text"]

                data = extract_json(content)
                result = parse_labeling_result(data, task_type)
                
                self.successful_predictions += 1
                return result

            except Exception as e:
                print(f"[Warning] API call failed on attempt {attempt+1}/{self.max_retries}: {e}")
                time.sleep(2 ** attempt) # Exponential backoff

        print(f"[Error] Failed to get prediction after {self.max_retries} attempts.")
        self.failed_predictions += 1
        return DetectionResult(task_type=task_type)
