import json
import os
import inspect
from importlib import reload
from typing import Any, Callable, Dict, Literal, Optional

from pydantic import BaseModel, Field, field_validator

from ..core.llm_client import extract_json
from ..core.model_config import is_bedrock_model
from . import conversation as conversation_module


ROUTER_SYSTEM_PROMPT = """
You are an intent router for a vision dataset application. Classify the user's request; do not
answer it and do not perform any operation. Return JSON only with this shape:
{
  "intent": "convert_labels|generate_labels|evaluate_labels|inspect_dataset|explain_result|configure_workspace|general_chat|unknown",
  "confidence": 0.0,
    "parameters": {
    "target_formats": ["yolo|pascal_voc|coco|vision_json"],
    "source_format": "auto|yolo|pascal_voc|coco|vision_json|csv|generic_json",
    "source_path": "optional workspace-relative path",
    "task_type": "classification|object_detection|segmentation|pose_estimation|ocr|tracking|all",
    "threshold": 0.75,
    "duplicate_iou": 0.85,
    "strict": false
  },
  "missing_parameters": []
}
Only classify executable requests as convert_labels, generate_labels, or evaluate_labels.
Use general_chat for greetings and questions that do not request a dataset operation.
Never invent a path or parameter. Omit unknown parameters. Preserve workspace-relative paths.
"""

CHAT_SYSTEM_PROMPT = """
You are the conversational help node for AutoLabel, a vision dataset application.
Answer in the user's language. Explain supported label conversion, automatic labeling,
evaluation, reports, and workspace usage concisely. Do not claim that a file operation ran,
do not create an execution plan, and do not invent dataset results.
"""

ACTION_INTENTS = {
    "convert_labels": "convert",
    "generate_labels": "generate",
    "evaluate_labels": "evaluate",
}
ALLOWED_FORMATS = {"yolo", "pascal_voc", "coco", "vision_json"}
ALLOWED_SOURCE_FORMATS = ALLOWED_FORMATS | {"auto", "csv", "generic_json"}
ALLOWED_TASKS = {
    "classification", "object_detection", "segmentation", "pose_estimation", "ocr", "tracking", "all",
}


class IntentParameters(BaseModel):
    target_formats: list[str] = Field(default_factory=list)
    source_format: Optional[str] = None
    source_path: Optional[str] = None
    task_type: Optional[str] = None
    threshold: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    duplicate_iou: Optional[float] = Field(default=None, gt=0.0, le=1.0)
    strict: bool = False

    @field_validator("target_formats")
    @classmethod
    def validate_formats(cls, value: list[str]) -> list[str]:
        invalid = set(value) - ALLOWED_FORMATS
        if invalid:
            raise ValueError(f"Unsupported target format: {', '.join(sorted(invalid))}")
        return list(dict.fromkeys(value))

    @field_validator("source_format")
    @classmethod
    def validate_source_format(cls, value: Optional[str]) -> Optional[str]:
        if value is not None and value not in ALLOWED_SOURCE_FORMATS:
            raise ValueError(f"Unsupported source format: {value}")
        return value

    @field_validator("task_type")
    @classmethod
    def validate_task_type(cls, value: Optional[str]) -> Optional[str]:
        if value is not None and value not in ALLOWED_TASKS:
            raise ValueError(f"Unsupported task type: {value}")
        return value


class IntentRoute(BaseModel):
    intent: Literal[
        "convert_labels", "generate_labels", "evaluate_labels", "inspect_dataset",
        "explain_result", "configure_workspace", "general_chat", "unknown",
    ]
    confidence: float = Field(ge=0.0, le=1.0)
    parameters: IntentParameters = Field(default_factory=IntentParameters)
    missing_parameters: list[str] = Field(default_factory=list)


def _call_model(model_name: str, system_prompt: str, request: str, json_output: bool) -> str:
    if is_bedrock_model(model_name):
        import boto3

        client = boto3.client(
            "bedrock-runtime",
            region_name=os.getenv("AWS_REGION") or os.getenv("AWS_DEFAULT_REGION", "us-east-1"),
        )
        body = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 1024,
            "temperature": 0.0,
            "system": system_prompt,
            "messages": [{"role": "user", "content": [{"type": "text", "text": request}]}],
        }
        response = client.invoke_model(
            modelId=model_name.removeprefix("bedrock:"),
            body=json.dumps(body),
            contentType="application/json",
            accept="application/json",
        )
        return json.loads(response["body"].read())["content"][0]["text"]
    if "gpt" in model_name.lower():
        from openai import OpenAI

        options: Dict[str, Any] = {}
        if json_output:
            options["response_format"] = {"type": "json_object"}
        response = OpenAI(api_key=os.getenv("OPENAI_API_KEY")).chat.completions.create(
            model=model_name,
            temperature=0.0,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": request},
            ],
            **options,
        )
        return response.choices[0].message.content or ""
    if "claude" in model_name.lower():
        from anthropic import Anthropic

        response = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY")).messages.create(
            model=model_name,
            max_tokens=1024,
            temperature=0.0,
            system=system_prompt,
            messages=[{"role": "user", "content": request}],
        )
        return response.content[0].text
    raise ValueError(f"지원하지 않는 대화 라우터 모델입니다: {model_name}")


class IntentRouter:
    def __init__(
        self,
        model_name: Optional[str] = None,
        caller: Optional[Callable[[str], Dict[str, Any]]] = None,
    ):
        self.model_name = (
            model_name
            or os.getenv("INTENT_ROUTER_MODEL")
            or os.getenv("PLANNER_MODEL")
            or os.getenv("LOW_MODEL")
        )
        self.caller = caller

    @property
    def enabled(self) -> bool:
        return bool(self.model_name or self.caller)

    def route(self, request: str) -> IntentRoute:
        if self.caller:
            payload = self.caller(request)
        elif self.model_name:
            payload = extract_json(_call_model(self.model_name, ROUTER_SYSTEM_PROMPT, request, True))
        else:
            raise RuntimeError("Intent Router 모델이 설정되지 않았습니다.")
        return IntentRoute.model_validate(payload)


class ChatNode:
    def __init__(
        self,
        model_name: Optional[str] = None,
        caller: Optional[Callable[[str], str]] = None,
    ):
        self.model_name = (
            model_name
            or os.getenv("CHAT_MODEL")
            or os.getenv("INTENT_ROUTER_MODEL")
            or os.getenv("PLANNER_MODEL")
            or os.getenv("LOW_MODEL")
        )
        self.caller = caller

    def respond(self, request: str) -> str:
        if self.caller:
            return self.caller(request)
        if not self.model_name:
            return (
                "현재는 라벨 형식 변환, 라벨 생성, 평가 작업을 지원합니다. "
                "실행하려는 작업과 원하는 출력 형식을 함께 입력해 주세요."
            )
        return _call_model(self.model_name, CHAT_SYSTEM_PROMPT, request, False).strip()


def _build_conversation_plan_with_overrides(
    request: str,
    workspace: str,
    overrides: Dict[str, Any],
) -> Dict[str, Any]:
    builder = conversation_module.build_conversation_plan
    if "intent_overrides" not in inspect.signature(builder).parameters:
        # Streamlit can keep an older imported conversation module alive across reruns.
        builder = reload(conversation_module).build_conversation_plan
    return builder(request, workspace, intent_overrides=overrides)


def handle_conversation(
    request: str,
    workspace: str,
    intent_router: Optional[IntentRouter] = None,
    chat_node: Optional[ChatNode] = None,
    minimum_confidence: float = 0.65,
) -> Dict[str, Any]:
    try:
        proposal = conversation_module.build_conversation_plan(request, workspace)
        return {"kind": "plan", "proposal": proposal, "route_source": "rules"}
    except ValueError as rule_error:
        message = str(rule_error)
        parse_failure = "작업 종류를 확인하지 못했습니다" in message or "출력 포맷을 확인하지 못했습니다" in message
        router = intent_router or IntentRouter()
        if not parse_failure or not router.enabled:
            raise

        try:
            route = router.route(request)
        except Exception as exc:
            raise ValueError(f"LLM Intent Router 호출에 실패했습니다: {exc}") from exc
        if route.confidence < minimum_confidence or route.intent == "unknown":
            return {
                "kind": "clarification",
                "response": "요청 의도를 확실히 판단하지 못했습니다. 변환, 라벨 생성, 평가 중 원하는 작업을 구체적으로 알려주세요.",
                "route": route.model_dump(),
            }
        if route.missing_parameters:
            missing = ", ".join(route.missing_parameters)
            return {
                "kind": "clarification",
                "response": f"작업 계획에 필요한 정보가 부족합니다: {missing}",
                "route": route.model_dump(),
            }
        if route.intent not in ACTION_INTENTS:
            try:
                response = (chat_node or ChatNode(model_name=router.model_name)).respond(request)
            except Exception as exc:
                raise ValueError(f"Chat Node 호출에 실패했습니다: {exc}") from exc
            return {"kind": "chat", "response": response, "route": route.model_dump()}

        overrides = route.parameters.model_dump(exclude_none=True)
        overrides["action"] = ACTION_INTENTS[route.intent]
        proposal = _build_conversation_plan_with_overrides(request, workspace, overrides)
        proposal["warnings"].append("규칙 기반 해석이 불충분하여 LLM Intent Router로 의도를 보완했습니다.")
        return {
            "kind": "plan",
            "proposal": proposal,
            "route": route.model_dump(),
            "route_source": "llm",
        }
