from typing import Any, Dict, List, Literal, Optional, TypedDict

from pydantic import BaseModel, Field, field_validator


ActionType = Literal["generate", "convert", "evaluate"]


class OperationPlan(BaseModel):
    action: ActionType
    task_type: str = "object_detection"
    input_path: Optional[str] = None
    img_dir: str = "data/raw"
    out_dir: str = "data/labeled"
    vis_dir: str = "data/visualized"
    formats: List[str] = Field(default_factory=lambda: ["yolo"])
    classes_path: Optional[str] = None
    custom_label_template: Optional[str] = None
    custom_label_extension: str = ".json"
    prompt: str = "Detect and classify all prominent objects in this image. Output strictly as JSON."
    generation_mode: str = "vlm_plugin"
    plugin_config: Optional[str] = None
    plugin_fail_fast: bool = False
    gt_dir: Optional[str] = None
    runs: Dict[str, str] = Field(default_factory=dict)
    source_format: str = "auto"
    duplicate_iou: float = Field(default=0.85, gt=0.0, le=1.0)
    threshold: float = 0.75
    eval_iou: float = 0.5
    insight_imbalance_ratio: float = Field(default=3.0, gt=1.0)
    inference_count: int = 3
    draft_temperature: float = 0.7
    low_model: Optional[str] = None
    high_model: Optional[str] = None
    strict: bool = False
    require_approval: bool = True
    max_retries: int = 2

    @field_validator("formats", mode="before")
    @classmethod
    def normalize_formats(cls, value):
        if isinstance(value, str):
            return [part.strip() for part in value.split(",") if part.strip()]
        return value or ["yolo"]

    @field_validator("task_type")
    @classmethod
    def validate_task_type(cls, value):
        allowed = {
            "classification", "object_detection", "segmentation",
            "pose_estimation", "ocr", "tracking", "all",
        }
        if value not in allowed:
            raise ValueError(f"Unsupported task_type: {value}")
        return value

    @field_validator("generation_mode")
    @classmethod
    def validate_generation_mode(cls, value):
        allowed = {"vlm_plugin", "specialist_only"}
        if value not in allowed:
            raise ValueError(f"Unsupported generation_mode: {value}")
        return value


class WorkflowPlan(BaseModel):
    request_summary: str = ""
    operations: List[OperationPlan]

    @field_validator("operations")
    @classmethod
    def require_operations(cls, value):
        if not value:
            raise ValueError("At least one operation is required")
        return value


class WorkflowState(TypedDict, total=False):
    request: str
    supplied_plan: Dict[str, Any]
    plan: Dict[str, Any]
    plan_errors: List[str]
    operation_index: int
    active_operation: Dict[str, Any]
    operation_status: str
    images: List[str]
    image_index: int
    current_image: str
    draft_results: List[Dict[str, Any]]
    current_result: Dict[str, Any]
    current_status: str
    current_plugin_records: List[Dict[str, Any]]
    current_issues: List[str]
    current_low_attempts: int
    current_high_attempts: int
    current_elapsed_sec: float
    high_required: bool
    high_approved: bool
    approval_denied: bool
    approval_reason: str
    retries: int
    schema_candidates: List[str]
    schema_index: int
    resolved_source_format: str
    last_error: str
    conversion_records: List[Dict[str, Any]]
    conversion_input_summary: Dict[str, Any]
    conversion_issues: List[Dict[str, Any]]
    run_records: List[Dict[str, Any]]
    operation_outputs: List[Dict[str, Any]]
    history: List[Dict[str, Any]]
    errors: List[str]
    status: str
    history_path: str
    auto_approve: bool
    thread_id: str
