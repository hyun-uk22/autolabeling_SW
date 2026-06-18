import json
import os
from typing import Optional

from ..core.llm_client import extract_json
from ..core.model_config import is_bedrock_model
from .models import OperationPlan, WorkflowPlan


PLANNER_SYSTEM_PROMPT = """
You convert a user's natural-language request into a deterministic vision-data workflow plan.
Return JSON only with this shape:
{
  "request_summary": "short summary",
  "operations": [
    {
      "action": "generate|convert|evaluate",
      "task_type": "classification|object_detection|segmentation|pose_estimation|ocr|tracking|all",
      "input_path": "optional label input",
      "img_dir": "image directory",
      "out_dir": "output directory",
      "vis_dir": "visualization directory",
      "formats": ["yolo|pascal_voc|coco|vision_json|custom|all"],
      "classes_path": "optional YOLO classes.txt",
      "custom_label_template": "required for custom output",
      "custom_label_extension": ".json",
      "prompt": "labeling instruction",
      "plugin_config": "optional plugin config path",
      "plugin_fail_fast": false,
      "gt_dir": "optional ground truth directory",
      "runs": {"method": "run directory"},
      "source_format": "auto|yolo|pascal_voc|coco|vision_json|csv|generic_json",
      "threshold": 0.75,
      "eval_iou": 0.5,
      "inference_count": 3,
      "draft_temperature": 0.7,
      "require_approval": true,
      "max_retries": 2
    }
  ]
}
Use separate operations in requested order when the request combines generation, conversion, and evaluation.
Do not invent absolute paths. Use project-relative defaults when paths are omitted.
"""


class WorkflowPlanner:
    def __init__(self, model_name: Optional[str] = None):
        self.model_name = model_name or os.getenv("PLANNER_MODEL") or os.getenv("LOW_MODEL")

    def plan(self, request: str, supplied_plan: Optional[dict] = None) -> WorkflowPlan:
        if supplied_plan:
            return WorkflowPlan.model_validate(supplied_plan)
        stripped = request.strip()
        if stripped.startswith("{"):
            return WorkflowPlan.model_validate(json.loads(stripped))
        if self.model_name:
            try:
                return WorkflowPlan.model_validate(self._call_model(request))
            except Exception as exc:
                print(f"[Warning] Planner model failed; using deterministic fallback: {exc}")
        return self._fallback_plan(request)

    def _call_model(self, request: str) -> dict:
        model_name = self.model_name
        if is_bedrock_model(model_name):
            import boto3

            client = boto3.client(
                "bedrock-runtime",
                region_name=os.getenv("AWS_REGION") or os.getenv("AWS_DEFAULT_REGION", "us-east-1"),
            )
            model_id = model_name.removeprefix("bedrock:")
            body = {
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": 2048,
                "temperature": 0.0,
                "system": PLANNER_SYSTEM_PROMPT,
                "messages": [{"role": "user", "content": [{"type": "text", "text": request}]}],
            }
            response = client.invoke_model(
                modelId=model_id,
                body=json.dumps(body),
                contentType="application/json",
                accept="application/json",
            )
            content = json.loads(response["body"].read())["content"][0]["text"]
            return extract_json(content)
        if "gpt" in model_name.lower():
            from openai import OpenAI

            response = OpenAI(api_key=os.getenv("OPENAI_API_KEY")).chat.completions.create(
                model=model_name,
                temperature=0.0,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": PLANNER_SYSTEM_PROMPT},
                    {"role": "user", "content": request},
                ],
            )
            return extract_json(response.choices[0].message.content)
        if "claude" in model_name.lower():
            from anthropic import Anthropic

            response = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY")).messages.create(
                model=model_name,
                max_tokens=2048,
                temperature=0.0,
                system=PLANNER_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": request}],
            )
            return extract_json(response.content[0].text)
        raise ValueError(f"Unsupported planner model: {model_name}")

    def _fallback_plan(self, request: str) -> WorkflowPlan:
        lowered = request.lower()
        operations = []
        if any(token in lowered for token in ["생성", "라벨링", "generate", "label image"]):
            task = "segmentation" if "segmentation" in lowered or "세그" in lowered else "object_detection"
            if "pose" in lowered or "포즈" in lowered:
                task = "pose_estimation"
            elif "ocr" in lowered or "문자" in lowered:
                task = "ocr"
            elif "tracking" in lowered or "추적" in lowered:
                task = "tracking"
            elif "classification" in lowered or "분류" in lowered:
                task = "classification"
            formats = ["vision_json"] if task != "object_detection" else ["yolo"]
            operations.append(OperationPlan(action="generate", task_type=task, formats=formats, prompt=request))
        if any(token in lowered for token in ["변환", "convert", "format"]):
            operations.append(OperationPlan(action="convert", input_path="data/external_labels", out_dir="data/converted"))
        if any(token in lowered for token in ["평가", "evaluate", "evaluation", "리포트"]):
            operations.append(OperationPlan(action="evaluate", out_dir="data/reports"))
        if not operations:
            operations.append(OperationPlan(action="generate", prompt=request))
        return WorkflowPlan(request_summary=request[:200], operations=operations)
