import os
from datetime import datetime, timezone
from typing import Any, Dict

from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt

from ..core.models import DetectionResult
from ..utils.label_validator import validate_result
from .models import OperationPlan, WorkflowState
from .runtime import WorkflowRuntime
from .schema_repair import repair_result


def _event(state: WorkflowState, event_type: str, **payload) -> list[dict]:
    return list(state.get("history", [])) + [{
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "event": event_type,
        "operation_index": state.get("operation_index", 0),
        "image": state.get("current_image"),
        **payload,
    }]


def _operation(state: WorkflowState) -> OperationPlan:
    return OperationPlan.model_validate(state["active_operation"])


def build_workflow_graph(runtime: WorkflowRuntime, checkpointer=None):
    builder = StateGraph(WorkflowState)

    def parse_request(state: WorkflowState) -> Dict[str, Any]:
        try:
            plan = runtime.plan(state.get("request", ""), state.get("supplied_plan"))
            return {
                "plan": plan.model_dump(),
                "plan_errors": [],
                "operation_index": 0,
                "operation_outputs": [],
                "history": _event(state, "plan_created", operations=len(plan.operations)),
                "status": "planned",
            }
        except Exception as exc:
            return {
                "plan_errors": [str(exc)],
                "errors": list(state.get("errors", [])) + [str(exc)],
                "status": "failed",
                "history": _event(state, "plan_failed", error=str(exc)),
            }

    def validate_plan(state: WorkflowState) -> Dict[str, Any]:
        if state.get("plan_errors"):
            return {}
        errors = []
        for index, raw in enumerate(state["plan"]["operations"]):
            operation = OperationPlan.model_validate(raw)
            if operation.action == "convert" and not operation.input_path:
                errors.append(f"operation[{index}] convert requires input_path")
            if operation.action == "evaluate" and not operation.runs:
                errors.append(f"operation[{index}] evaluate requires runs")
            if "custom" in operation.formats and not operation.custom_label_template:
                errors.append(f"operation[{index}] custom format requires custom_label_template")
        return {
            "plan_errors": errors,
            "errors": list(state.get("errors", [])) + errors,
            "status": "validated" if not errors else "failed",
            "history": _event(state, "plan_validated", errors=errors),
        }

    def begin_operation(state: WorkflowState) -> Dict[str, Any]:
        index = state.get("operation_index", 0)
        raw = state["plan"]["operations"][index]
        return {
            "active_operation": raw,
            "operation_status": "started",
            "images": [],
            "image_index": 0,
            "current_image": "",
            "draft_results": [],
            "current_result": {},
            "current_plugin_records": [],
            "current_issues": [],
            "high_required": False,
            "high_approved": False,
            "approval_denied": False,
            "approval_reason": "",
            "retries": 0,
            "schema_candidates": [],
            "schema_index": 0,
            "conversion_records": [],
            "conversion_issues": [],
            "run_records": [],
            "history": _event(state, "operation_started", action=raw["action"]),
            "status": "running",
        }

    def prepare_generate(state: WorkflowState) -> Dict[str, Any]:
        operation = _operation(state)
        try:
            prepared = runtime.prepare_generation(operation)
            return {
                "images": prepared["images"],
                "image_index": 0,
                "operation_status": "generating",
                "history": _event(
                    state,
                    "generation_prepared",
                    images=len(prepared["images"]),
                    low_model=prepared["low_model"],
                    high_model=prepared["high_model"],
                    warnings=prepared["warnings"],
                ),
            }
        except Exception as exc:
            return {
                "operation_status": "failed",
                "errors": list(state.get("errors", [])) + [str(exc)],
                "history": _event(state, "generation_prepare_failed", error=str(exc)),
            }

    def select_image(state: WorkflowState) -> Dict[str, Any]:
        index = state.get("image_index", 0)
        if index >= len(state.get("images", [])):
            return {"operation_status": "generation_complete"}
        return {
            "current_image": state["images"][index],
            "draft_results": [],
            "current_result": {},
            "current_plugin_records": [],
            "current_issues": [],
            "current_status": "Drafting",
            "high_required": False,
            "high_approved": False,
            "approval_denied": False,
            "approval_reason": "",
            "retries": 0,
            "current_low_attempts": 0,
            "current_high_attempts": 0,
            "current_elapsed_sec": 0.0,
            "history": _event(state, "image_selected", image=state["images"][index]),
        }

    def generate_draft(state: WorkflowState) -> Dict[str, Any]:
        operation = _operation(state)
        image_path = f"{operation.img_dir}/{state['current_image']}"
        try:
            output = runtime.generate_drafts(image_path, operation)
            return {
                "draft_results": output["drafts"],
                "current_result": output["result"],
                "current_low_attempts": output["low_attempts"],
                "current_elapsed_sec": output["elapsed_sec"],
                "history": _event(state, "draft_generated", consistency=output["consistency"]),
            }
        except Exception as exc:
            empty = DetectionResult(task_type=operation.task_type)
            return {
                "current_result": empty.model_dump(),
                "current_issues": [f"draft_failed:{exc}"],
                "errors": list(state.get("errors", [])) + [str(exc)],
                "history": _event(state, "draft_failed", error=str(exc)),
            }

    def run_specialists(state: WorkflowState) -> Dict[str, Any]:
        operation = _operation(state)
        result = DetectionResult.model_validate(state["current_result"])
        image_path = f"{operation.img_dir}/{state['current_image']}"
        output = runtime.run_specialists(image_path, operation, result)
        return {
            "current_result": output["result"].model_dump(),
            "current_plugin_records": output["records"],
            "current_elapsed_sec": state.get("current_elapsed_sec", 0.0) + output["elapsed_sec"],
            "history": _event(state, "specialists_completed", records=output["records"]),
        }

    def decide_high(state: WorkflowState) -> Dict[str, Any]:
        operation = _operation(state)
        result = DetectionResult.model_validate(state["current_result"])
        image_path = f"{operation.img_dir}/{state['current_image']}"
        issues = validate_result(result, image_path)
        required, reason = runtime.needs_high_verification(
            operation,
            result,
            state.get("current_plugin_records", []),
            issues,
        )
        return {
            "current_issues": issues,
            "high_required": required,
            "approval_reason": reason,
            "current_status": "HighApprovalRequired" if required else "Consistent",
            "history": _event(state, "high_decision", required=required, reason=reason),
        }

    def approval_gate(state: WorkflowState) -> Dict[str, Any]:
        operation = _operation(state)
        if state.get("auto_approve") or not operation.require_approval:
            approved = True
        else:
            decision = interrupt({
                "type": "expensive_model_approval",
                "message": "High-capacity VLM invocation requires approval.",
                "image": state.get("current_image"),
                "reason": state.get("approval_reason"),
                "operation_index": state.get("operation_index", 0),
            })
            if isinstance(decision, dict):
                approved = bool(decision.get("approved"))
            elif isinstance(decision, str):
                approved = decision.lower() in {"approve", "approved", "yes", "true", "1"}
            else:
                approved = bool(decision)
        return {
            "high_approved": approved,
            "approval_denied": not approved,
            "current_status": "HighApproved" if approved else "ApprovalDenied",
            "history": _event(state, "high_approval", approved=approved),
        }

    def high_verify(state: WorkflowState) -> Dict[str, Any]:
        operation = _operation(state)
        image_path = f"{operation.img_dir}/{state['current_image']}"
        result = DetectionResult.model_validate(state["current_result"])
        output = runtime.run_high_verification(image_path, operation, result)
        return {
            "current_result": output["result"].model_dump(),
            "current_high_attempts": output["high_attempts"],
            "current_elapsed_sec": state.get("current_elapsed_sec", 0.0) + output["elapsed_sec"],
            "current_status": "Escalated",
            "history": _event(state, "high_verified", agreement=output["agreement"]),
        }

    def validate_generated(state: WorkflowState) -> Dict[str, Any]:
        operation = _operation(state)
        result = DetectionResult.model_validate(state["current_result"])
        image_path = f"{operation.img_dir}/{state['current_image']}"
        issues = validate_result(result, image_path)
        can_repair = bool(issues) and state.get("retries", 0) < operation.max_retries
        return {
            "current_issues": issues,
            "operation_status": "repair_generated" if can_repair else "save_generated",
            "history": _event(state, "generated_validated", issues=issues, repair=can_repair),
        }

    def repair_generated(state: WorkflowState) -> Dict[str, Any]:
        repaired = repair_result(DetectionResult.model_validate(state["current_result"]))
        return {
            "current_result": repaired.model_dump(),
            "retries": state.get("retries", 0) + 1,
            "history": _event(state, "generated_repaired", retry=state.get("retries", 0) + 1),
        }

    def save_generated(state: WorkflowState) -> Dict[str, Any]:
        record = {
            "image": state["current_image"],
            "result": state["current_result"],
            "status": state.get("current_status", "Consistent"),
            "plugin_records": state.get("current_plugin_records", []),
            "issues": state.get("current_issues", []),
            "low_api_attempts": state.get("current_low_attempts", 0),
            "high_api_attempts": state.get("current_high_attempts", 0),
            "elapsed_sec": state.get("current_elapsed_sec", 0.0),
        }
        return {
            "run_records": list(state.get("run_records", [])) + [record],
            "image_index": state.get("image_index", 0) + 1,
            "history": _event(
                state,
                "image_completed",
                status=record["status"],
                issues=record["issues"],
            ),
        }

    def finalize_generate(state: WorkflowState) -> Dict[str, Any]:
        operation = _operation(state)
        try:
            output = runtime.export_generation(operation, state.get("run_records", []))
            return {
                "operation_outputs": list(state.get("operation_outputs", [])) + [output],
                "operation_status": "completed",
                "history": _event(state, "generation_exported", output=output),
            }
        except Exception as exc:
            return {
                "operation_status": "failed",
                "errors": list(state.get("errors", [])) + [str(exc)],
                "history": _event(state, "generation_export_failed", error=str(exc)),
            }

    def prepare_convert(state: WorkflowState) -> Dict[str, Any]:
        operation = _operation(state)
        candidates = runtime.conversion_schema_candidates(operation)
        return {
            "schema_candidates": candidates,
            "schema_index": 0,
            "retries": 0,
            "operation_status": "loading_conversion",
            "history": _event(state, "conversion_prepared", candidates=candidates),
        }

    def load_conversion(state: WorkflowState) -> Dict[str, Any]:
        operation = _operation(state)
        candidates = state.get("schema_candidates", [])
        index = state.get("schema_index", 0)
        source_format = candidates[index] if index < len(candidates) else operation.source_format
        try:
            records = runtime.load_conversion(operation, source_format)
            if not records:
                raise ValueError(f"No records parsed with source format {source_format}")
            return {
                "conversion_records": records,
                "resolved_source_format": source_format,
                "operation_status": "conversion_loaded",
                "history": _event(state, "conversion_loaded", source_format=source_format, records=len(records)),
            }
        except Exception as exc:
            return {
                "operation_status": "schema_error",
                "last_error": str(exc),
                "history": _event(state, "conversion_load_failed", source_format=source_format, error=str(exc)),
            }

    def reanalyze_schema(state: WorkflowState) -> Dict[str, Any]:
        next_index = state.get("schema_index", 0) + 1
        retry = state.get("retries", 0) + 1
        candidates = state.get("schema_candidates", [])
        can_retry = next_index < len(candidates) and retry <= _operation(state).max_retries
        error = state.get("last_error", "schema analysis failed")
        return {
            "schema_index": next_index,
            "retries": retry,
            "operation_status": "loading_conversion" if can_retry else "failed",
            "errors": list(state.get("errors", [])) + ([] if can_retry else [error]),
            "history": _event(state, "schema_reanalyzed", retry=retry, can_retry=can_retry),
        }

    def validate_conversion(state: WorkflowState) -> Dict[str, Any]:
        operation = _operation(state)
        validations = runtime.validate_conversion(operation, state.get("conversion_records", []))
        has_issues = any(item["issues"] for item in validations)
        can_repair = has_issues and state.get("retries", 0) < operation.max_retries
        return {
            "conversion_issues": validations,
            "operation_status": "repair_conversion" if can_repair else "export_conversion",
            "history": _event(state, "conversion_validated", issues=sum(bool(item["issues"]) for item in validations), repair=can_repair),
        }

    def repair_conversion(state: WorkflowState) -> Dict[str, Any]:
        records = runtime.repair_conversion(state.get("conversion_records", []))
        return {
            "conversion_records": records,
            "retries": state.get("retries", 0) + 1,
            "history": _event(state, "conversion_repaired", retry=state.get("retries", 0) + 1),
        }

    def export_conversion(state: WorkflowState) -> Dict[str, Any]:
        operation = _operation(state)
        try:
            output = runtime.export_conversion(
                operation,
                state.get("conversion_records", []),
                state.get("conversion_issues", []),
                state.get("resolved_source_format", operation.source_format),
            )
            return {
                "operation_outputs": list(state.get("operation_outputs", [])) + [output],
                "operation_status": "completed",
                "history": _event(state, "conversion_exported", output=output),
            }
        except Exception as exc:
            return {
                "operation_status": "failed",
                "errors": list(state.get("errors", [])) + [str(exc)],
                "history": _event(state, "conversion_export_failed", error=str(exc)),
            }

    def execute_evaluate(state: WorkflowState) -> Dict[str, Any]:
        try:
            output = runtime.evaluate(_operation(state))
            return {
                "operation_outputs": list(state.get("operation_outputs", [])) + [output],
                "operation_status": "completed",
                "history": _event(state, "evaluation_completed", output=output),
            }
        except Exception as exc:
            return {
                "operation_status": "failed",
                "errors": list(state.get("errors", [])) + [str(exc)],
                "history": _event(state, "evaluation_failed", error=str(exc)),
            }

    def advance_operation(state: WorkflowState) -> Dict[str, Any]:
        index = state.get("operation_index", 0) + 1
        return {
            "operation_index": index,
            "history": _event(state, "operation_finished", status=state.get("operation_status")),
        }

    def finalize_workflow(state: WorkflowState) -> Dict[str, Any]:
        output_dir = os.path.join("data", "workflow", state.get("thread_id", "default"))
        final_history = _event(state, "workflow_completed")
        history_path = runtime.save_history(output_dir, final_history)
        return {
            "status": "completed" if not state.get("errors") else "completed_with_errors",
            "history_path": history_path,
            "history": final_history,
        }

    def fail_workflow(state: WorkflowState) -> Dict[str, Any]:
        return {"status": "failed", "history": _event(state, "workflow_failed", errors=state.get("plan_errors", []))}

    def route_plan(state):
        return "fail" if state.get("plan_errors") else "begin"

    def route_action(state):
        return state["active_operation"]["action"]

    def route_generate_ready(state):
        return "advance" if state.get("operation_status") == "failed" else "select"

    def route_image(state):
        return "finalize" if state.get("image_index", 0) >= len(state.get("images", [])) else "draft"

    def route_high(state):
        return "approval" if state.get("high_required") else "validate"

    def route_approval(state):
        return "high" if state.get("high_approved") else "validate"

    def route_generated_validation(state):
        return "repair" if state.get("operation_status") == "repair_generated" else "save"

    def route_conversion_load(state):
        return "reanalyze" if state.get("operation_status") == "schema_error" else "validate"

    def route_schema_retry(state):
        return "load" if state.get("operation_status") == "loading_conversion" else "advance"

    def route_conversion_validation(state):
        return "repair" if state.get("operation_status") == "repair_conversion" else "export"

    def route_advance(state):
        return "begin" if state.get("operation_index", 0) < len(state["plan"]["operations"]) else "finalize"

    builder.add_node("parse_request", parse_request)
    builder.add_node("validate_plan", validate_plan)
    builder.add_node("begin_operation", begin_operation)
    builder.add_node("prepare_generate", prepare_generate)
    builder.add_node("select_image", select_image)
    builder.add_node("generate_draft", generate_draft)
    builder.add_node("run_specialists", run_specialists)
    builder.add_node("decide_high", decide_high)
    builder.add_node("approval_gate", approval_gate)
    builder.add_node("high_verify", high_verify)
    builder.add_node("validate_generated", validate_generated)
    builder.add_node("repair_generated", repair_generated)
    builder.add_node("save_generated", save_generated)
    builder.add_node("finalize_generate", finalize_generate)
    builder.add_node("prepare_convert", prepare_convert)
    builder.add_node("load_conversion", load_conversion)
    builder.add_node("reanalyze_schema", reanalyze_schema)
    builder.add_node("validate_conversion", validate_conversion)
    builder.add_node("repair_conversion", repair_conversion)
    builder.add_node("export_conversion", export_conversion)
    builder.add_node("execute_evaluate", execute_evaluate)
    builder.add_node("advance_operation", advance_operation)
    builder.add_node("finalize_workflow", finalize_workflow)
    builder.add_node("fail_workflow", fail_workflow)

    builder.add_edge(START, "parse_request")
    builder.add_edge("parse_request", "validate_plan")
    builder.add_conditional_edges("validate_plan", route_plan, {"fail": "fail_workflow", "begin": "begin_operation"})
    builder.add_conditional_edges("begin_operation", route_action, {
        "generate": "prepare_generate",
        "convert": "prepare_convert",
        "evaluate": "execute_evaluate",
    })
    builder.add_conditional_edges("prepare_generate", route_generate_ready, {"advance": "advance_operation", "select": "select_image"})
    builder.add_conditional_edges("select_image", route_image, {"draft": "generate_draft", "finalize": "finalize_generate"})
    builder.add_edge("generate_draft", "run_specialists")
    builder.add_edge("run_specialists", "decide_high")
    builder.add_conditional_edges("decide_high", route_high, {"approval": "approval_gate", "validate": "validate_generated"})
    builder.add_conditional_edges("approval_gate", route_approval, {"high": "high_verify", "validate": "validate_generated"})
    builder.add_edge("high_verify", "validate_generated")
    builder.add_conditional_edges("validate_generated", route_generated_validation, {"repair": "repair_generated", "save": "save_generated"})
    builder.add_edge("repair_generated", "validate_generated")
    builder.add_edge("save_generated", "select_image")
    builder.add_edge("finalize_generate", "advance_operation")
    builder.add_edge("prepare_convert", "load_conversion")
    builder.add_conditional_edges("load_conversion", route_conversion_load, {"reanalyze": "reanalyze_schema", "validate": "validate_conversion"})
    builder.add_conditional_edges("reanalyze_schema", route_schema_retry, {"load": "load_conversion", "advance": "advance_operation"})
    builder.add_conditional_edges("validate_conversion", route_conversion_validation, {"repair": "repair_conversion", "export": "export_conversion"})
    builder.add_edge("repair_conversion", "validate_conversion")
    builder.add_edge("export_conversion", "advance_operation")
    builder.add_edge("execute_evaluate", "advance_operation")
    builder.add_conditional_edges("advance_operation", route_advance, {"begin": "begin_operation", "finalize": "finalize_workflow"})
    builder.add_edge("finalize_workflow", END)
    builder.add_edge("fail_workflow", END)

    return builder.compile(checkpointer=checkpointer)
