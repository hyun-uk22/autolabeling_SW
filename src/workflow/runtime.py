import csv
import json
import os
import time
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv

from ..agents.insight_agent import DatasetInsightAgent
from ..agents.verification_agent import HierarchicalVerificationAgent
from ..core.llm_client import VisionLLMClient
from ..core.model_config import required_api_keys, resolve_model_names, validate_cascade_setup
from ..core.models import DetectionResult
from ..plugins.orchestrator import TaskPluginOrchestrator
from ..plugins.registry import load_generation_plugins
from ..reporting import (
    ArtifactAuditor,
    build_conversion_preflight,
    build_generation_performance,
    build_user_action_report,
)
from ..utils.evaluation import build_experiment_report, evaluate_yolo_dirs, save_experiment_report
from ..utils.format_converter import LabelExportWriter
from ..utils.geometry import consistency_metric_name
from ..utils.label_importer import find_image_path, import_labels_with_report
from ..utils.label_validator import summarize_validation, validate_result
from ..utils.result_metrics import count_result_labels
from ..utils.visualize import visualize_boxes
from .models import OperationPlan, WorkflowPlan
from .planner import WorkflowPlanner
from .schema_repair import repair_result, schema_candidates

load_dotenv()


class WorkflowRuntime:
    def __init__(self, planner_model: Optional[str] = None, allow_same_model: bool = False):
        self.planner = WorkflowPlanner(planner_model)
        self.allow_same_model = allow_same_model
        self._generation_key = None
        self.low_client = None
        self.high_client = None
        self.verification_agent = None
        self.plugin_orchestrator = None
        self.plugin_prepare_records = []

    def plan(self, request: str, supplied_plan: Optional[dict] = None) -> WorkflowPlan:
        return self.planner.plan(request, supplied_plan)

    @staticmethod
    def analyze_dataset(operation: OperationPlan, results: List[DetectionResult]) -> Dict[str, Any]:
        agent = DatasetInsightAgent(operation.insight_imbalance_ratio)
        return agent.analyze(results)

    def prepare_generation(self, operation: OperationPlan) -> Dict[str, Any]:
        os.makedirs(operation.img_dir, exist_ok=True)
        os.makedirs(operation.out_dir, exist_ok=True)
        os.makedirs(operation.vis_dir, exist_ok=True)
        images = sorted(
            name for name in os.listdir(operation.img_dir)
            if name.lower().endswith((".png", ".jpg", ".jpeg"))
        )
        if operation.generation_mode == "specialist_only":
            self._ensure_generation_clients(operation, None, None)
            return {
                "images": images,
                "low_model": None,
                "high_model": None,
                "warnings": ["specialist_only mode: VLM draft and high verification are disabled"],
                "plugin_prepare_records": self.plugin_prepare_records,
            }
        low_model, high_model = resolve_model_names(operation.low_model, operation.high_model)
        valid, messages = validate_cascade_setup(low_model, high_model, self.allow_same_model)
        if not valid:
            raise ValueError("; ".join(messages))
        missing_keys = [key for key in required_api_keys(low_model, high_model) if not os.getenv(key)]
        if missing_keys:
            raise ValueError(f"Missing API key(s): {', '.join(missing_keys)}")
        self._ensure_generation_clients(operation, low_model, high_model)
        return {
            "images": images,
            "low_model": low_model,
            "high_model": high_model,
            "warnings": messages,
            "plugin_prepare_records": self.plugin_prepare_records,
        }

    def _ensure_generation_clients(self, operation: OperationPlan, low_model: Optional[str], high_model: Optional[str]) -> None:
        key = (
            operation.generation_mode,
            low_model,
            high_model,
            operation.task_type,
            operation.inference_count,
            operation.draft_temperature,
            operation.plugin_config,
            operation.plugin_fail_fast,
        )
        if key == self._generation_key:
            return
        if operation.generation_mode == "specialist_only":
            self.low_client = None
            self.high_client = None
            self.verification_agent = None
        else:
            self.low_client = VisionLLMClient(low_model)
            self.high_client = VisionLLMClient(high_model)
            self.verification_agent = HierarchicalVerificationAgent(
                self.low_client,
                self.high_client,
                threshold=operation.threshold,
                inference_count=operation.inference_count,
                draft_temperature=operation.draft_temperature,
            )
        plugins = load_generation_plugins(operation.plugin_config, operation.generation_mode)
        self.plugin_orchestrator = TaskPluginOrchestrator(plugins, fail_fast=operation.plugin_fail_fast)
        self.plugin_prepare_records = self.plugin_orchestrator.prepare(operation.task_type)
        self._generation_key = key

    def _ensure_generation_for_operation(self, operation: OperationPlan) -> None:
        if operation.generation_mode == "specialist_only":
            if self._generation_key != (
                operation.generation_mode,
                None,
                None,
                operation.task_type,
                operation.inference_count,
                operation.draft_temperature,
                operation.plugin_config,
                operation.plugin_fail_fast,
            ):
                self._ensure_generation_clients(operation, None, None)
            return
        low_model, high_model = resolve_model_names(operation.low_model, operation.high_model)
        if self._generation_key != (
            operation.generation_mode,
            low_model,
            high_model,
            operation.task_type,
            operation.inference_count,
            operation.draft_temperature,
            operation.plugin_config,
            operation.plugin_fail_fast,
        ):
            self._ensure_generation_clients(operation, low_model, high_model)

    def generate_drafts(self, image_path: str, operation: OperationPlan) -> Dict[str, Any]:
        self._ensure_generation_for_operation(operation)
        if operation.generation_mode == "specialist_only":
            seed = DetectionResult(task_type=operation.task_type)
            return {
                "drafts": [],
                "result": seed.model_dump(),
                "consistency": 0.0,
                "low_attempts": 0,
                "elapsed_sec": 0.0,
            }
        started = time.perf_counter()
        before = self.low_client.api_attempts
        drafts, seed, consistency = self.verification_agent.generate_draft_labels(
            image_path,
            operation.prompt,
            operation.task_type,
        )
        return {
            "drafts": [item.model_dump() for item in drafts],
            "result": seed.model_dump(),
            "consistency": consistency,
            "low_attempts": self.low_client.api_attempts - before,
            "elapsed_sec": time.perf_counter() - started,
        }

    def run_specialists(self, image_path: str, operation: OperationPlan, result: DetectionResult) -> Dict[str, Any]:
        self._ensure_generation_for_operation(operation)
        if not self.plugin_orchestrator:
            return {"result": result, "records": [], "elapsed_sec": 0.0}
        started = time.perf_counter()
        merged, records = self.plugin_orchestrator.process(
            image_path,
            operation.prompt,
            operation.task_type,
            result,
        )
        return {"result": merged, "records": records, "elapsed_sec": time.perf_counter() - started}

    def needs_high_verification(
        self,
        operation: OperationPlan,
        result: DetectionResult,
        plugin_records: List[dict],
        issues: List[str],
    ) -> Tuple[bool, str]:
        self._ensure_generation_for_operation(operation)
        if operation.generation_mode == "specialist_only":
            return False, "specialist_only mode disables high VLM verification"
        return self.verification_agent.needs_escalation(
            result,
            threshold=operation.threshold,
            plugin_records=plugin_records,
            issues=issues,
        )

    def run_high_verification(
        self,
        image_path: str,
        operation: OperationPlan,
        specialist_result: DetectionResult,
    ) -> Dict[str, Any]:
        self._ensure_generation_for_operation(operation)
        if operation.generation_mode == "specialist_only":
            return {
                "result": specialist_result,
                "agreement": 0.0,
                "high_attempts": 0,
                "elapsed_sec": 0.0,
            }
        started = time.perf_counter()
        before = self.high_client.api_attempts
        merged, agreement = self.verification_agent.high_verify(
            image_path,
            operation.prompt,
            operation.task_type,
            specialist_result,
        )
        return {
            "result": merged,
            "agreement": agreement,
            "high_attempts": self.high_client.api_attempts - before,
            "elapsed_sec": time.perf_counter() - started,
        }

    def export_generation(
        self,
        operation: OperationPlan,
        records: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        formats = operation.formats
        if formats == ["yolo"] and operation.task_type != "object_detection":
            formats = ["vision_json"]
        writer = LabelExportWriter(
            operation.out_dir,
            formats=formats,
            custom_template_path=operation.custom_label_template,
            custom_extension=operation.custom_label_extension,
        )
        auditor = ArtifactAuditor()
        metric_rows = []
        export_records = []
        exported_results = []
        for record in records:
            result = DetectionResult.model_validate(record["result"])
            exported_results.append(result)
            image_path = os.path.join(operation.img_dir, record["image"])
            label_paths = writer.save(result, image_path)
            export_records.append({
                "image": record["image"],
                "paths": label_paths,
                "issues": auditor.audit_record(label_paths),
            })
            vis_path = visualize_boxes(image_path, result, operation.vis_dir)
            metric_rows.append({
                "image": record["image"],
                "status": record["status"],
                "source_model": result.source_model,
                "task_type": result.task_type,
                "consistency_metric": consistency_metric_name(result.task_type),
                "objects": count_result_labels(result),
                "boxes": len(result.boxes),
                "segments": len(result.segments),
                "poses": len(result.poses),
                "texts": len(result.texts),
                "tracks": len(result.tracks),
                "classifications": len(result.classifications),
                "consistency_score": result.consistency_score,
                "mean_confidence": result.mean_confidence,
                "uncertainty_score": result.uncertainty_score,
                "plugin_scores": json.dumps(result.plugin_scores, ensure_ascii=False),
                "plugin_records": json.dumps(record.get("plugin_records", []), ensure_ascii=False),
                "validation_issues": json.dumps(record.get("issues", []), ensure_ascii=False),
                "low_api_attempts": record.get("low_api_attempts", 0),
                "high_api_attempts": record.get("high_api_attempts", 0),
                "elapsed_sec": record.get("elapsed_sec", 0.0),
                "label_path": label_paths.get("yolo") or next(iter(label_paths.values()), ""),
                "label_paths": json.dumps(label_paths, ensure_ascii=False),
                "visualization_path": vis_path,
            })
        artifacts = writer.finalize()
        artifact_issues = auditor.audit_artifacts(artifacts)
        metrics_csv = os.path.join(operation.out_dir, "run_metrics.csv")
        metrics_jsonl = os.path.join(operation.out_dir, "run_metrics.jsonl")
        if metric_rows:
            with open(metrics_csv, "w", newline="", encoding="utf-8") as f:
                writer_csv = csv.DictWriter(f, fieldnames=list(metric_rows[0]))
                writer_csv.writeheader()
                writer_csv.writerows(metric_rows)
            with open(metrics_jsonl, "w", encoding="utf-8") as f:
                for row in metric_rows:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
        evaluation = None
        if operation.gt_dir and os.path.isdir(operation.gt_dir) and "yolo" in formats:
            evaluation = evaluate_yolo_dirs(operation.out_dir, operation.gt_dir, operation.eval_iou)
        validation_records = [
            {"image": record["image"], "issues": record.get("issues", [])}
            for record in records
        ]
        user_action_report = build_user_action_report(
            validation_records,
            export_records,
            artifact_issues,
            total_records=len(records),
        )
        total_elapsed = sum(row["elapsed_sec"] for row in metric_rows)
        low_attempts = sum(row["low_api_attempts"] for row in metric_rows)
        high_attempts = sum(row["high_api_attempts"] for row in metric_rows)
        escalation_count = sum(1 for row in metric_rows if row["status"] == "Escalated")
        plugins = sorted({
            plugin_record.get("plugin", "")
            for record in records
            for plugin_record in record.get("plugin_records", [])
            if plugin_record.get("plugin")
        })
        summary = {
            "report_version": "2.0",
            "action": "generate",
            "images": len(records),
            "task_type": operation.task_type,
            "consistency_metric": consistency_metric_name(operation.task_type),
            "formats": formats,
            "total_labels": sum(row["objects"] for row in metric_rows),
            "total_elapsed_sec": total_elapsed,
            "low_api_attempts": low_attempts,
            "high_api_attempts": high_attempts,
            "escalation_count": escalation_count,
            "plugins": plugins,
            "plugin_prepare_records": self.plugin_prepare_records,
            "evaluation": evaluation,
            "performance": build_generation_performance(
                len(records), total_elapsed, low_attempts, high_attempts, escalation_count,
            ),
            "dataset_insight": self.analyze_dataset(operation, exported_results),
            "export_validation": {
                "failed_records": sum(bool(record["issues"]) for record in export_records),
                "artifact_issues": artifact_issues,
                "records": export_records,
            },
            "user_action_report": user_action_report,
            "artifacts": artifacts,
        }
        summary_path = os.path.join(operation.out_dir, "run_summary.json")
        user_action_path = os.path.join(operation.out_dir, "user_action_report.json")
        summary["summary_path"] = summary_path
        summary["user_action_report_path"] = user_action_path
        with open(user_action_path, "w", encoding="utf-8") as f:
            json.dump(user_action_report, f, ensure_ascii=False, indent=2)
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        return summary

    def load_conversion(self, operation: OperationPlan, source_format: str) -> Dict[str, Any]:
        batch = import_labels_with_report(
            operation.input_path,
            operation.img_dir,
            source_format=source_format,
            classes_path=operation.classes_path,
            duplicate_iou=operation.duplicate_iou,
        )
        detected_formats = list(batch.report.get("formats", {}))
        resolved_source_format = source_format
        if source_format == "auto":
            resolved_source_format = detected_formats[0] if len(detected_formats) == 1 else "mixed"
        return {
            "records": [
                {"image": image, "result": result.model_dump()}
                for image, result in batch.records
            ],
            "input_summary": batch.report,
            "resolved_source_format": resolved_source_format,
        }

    def validate_conversion(self, operation: OperationPlan, records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        validations = []
        for record in records:
            image_path = find_image_path(operation.img_dir, record["image"])
            result = DetectionResult.model_validate(record["result"])
            validations.append({"image": record["image"], "issues": validate_result(result, image_path)})
        return validations

    def repair_conversion(self, records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return [
            {"image": record["image"], "result": repair_result(DetectionResult.model_validate(record["result"])).model_dump()}
            for record in records
        ]

    def export_conversion(
        self,
        operation: OperationPlan,
        records: List[Dict[str, Any]],
        validations: List[Dict[str, Any]],
        source_format: str,
        input_summary: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        validation_map = {item["image"]: item["issues"] for item in validations}
        writer = LabelExportWriter(
            operation.out_dir,
            formats=operation.formats,
            custom_template_path=operation.custom_label_template,
            custom_extension=operation.custom_label_extension,
            initial_class_list=(input_summary or {}).get("class_list"),
        )
        auditor = ArtifactAuditor()
        converted = 0
        export_records = []
        exported_results = []
        for record in records:
            issues = validation_map.get(record["image"], [])
            blocking = any(
                issue.startswith("missing_image:")
                or issue.startswith("image_open_failed:")
                or issue == "invalid_image_size"
                for issue in issues
            )
            if blocking or (operation.strict and issues):
                continue
            image_path = find_image_path(operation.img_dir, record["image"])
            result = DetectionResult.model_validate(record["result"])
            paths = writer.save(result, image_path)
            export_issues = auditor.audit_record(paths)
            export_records.append({
                "image": record["image"],
                "paths": paths,
                "issues": export_issues,
            })
            exported_results.append(result)
            if not export_issues:
                converted += 1
        artifacts = writer.finalize()
        artifact_issues = auditor.audit_artifacts(artifacts)
        preflight = build_conversion_preflight(input_summary or {}, writer.formats, validations)
        user_action_report = build_user_action_report(
            validations,
            export_records,
            artifact_issues,
            total_records=len(records),
        )
        preflight_actions = [
            notice["user_action"]
            for notice in preflight.get("notices", [])
            if notice.get("severity") in {"critical", "warning"}
        ]
        if preflight_actions:
            user_action_report["recommended_actions"] = list(dict.fromkeys(
                preflight_actions + user_action_report.get("recommended_actions", [])
            ))
        report = {
            "report_version": "2.0",
            "action": "convert",
            "input": operation.input_path,
            "resolved_source_format": source_format,
            "input_summary": input_summary or {},
            "target_formats": writer.formats,
            "records_read": len(records),
            "records_converted": converted,
            "preflight": preflight,
            "validation": summarize_validation(validations),
            "export_validation": {
                "failed_records": sum(bool(record["issues"]) for record in export_records),
                "artifact_issues": artifact_issues,
            },
            "user_action_report": user_action_report,
            "dataset_insight": self.analyze_dataset(operation, exported_results),
            "records": validations,
            "exports": export_records,
            "artifacts": artifacts,
        }
        report_path = os.path.join(operation.out_dir, "conversion_report.json")
        user_action_path = os.path.join(operation.out_dir, "user_action_report.json")
        report["report_path"] = report_path
        report["user_action_report_path"] = user_action_path
        with open(user_action_path, "w", encoding="utf-8") as f:
            json.dump(user_action_report, f, ensure_ascii=False, indent=2)
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        return report

    def conversion_schema_candidates(self, operation: OperationPlan) -> List[str]:
        return schema_candidates(operation.input_path, operation.source_format)

    def evaluate(self, operation: OperationPlan) -> Dict[str, Any]:
        report = build_experiment_report(operation.runs, gt_dir=operation.gt_dir)
        paths = save_experiment_report(report, operation.out_dir)
        return {"action": "evaluate", "rows": report, "artifacts": paths}

    def save_history(self, output_dir: str, history: List[Dict[str, Any]]) -> str:
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, "workflow_history.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(history, f, ensure_ascii=False, indent=2)
        return path
