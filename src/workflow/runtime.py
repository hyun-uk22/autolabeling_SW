import csv
import json
import os
import time
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv

from ..agents.labeling_agent import LabelingAgent
from ..agents.insight_agent import DatasetInsightAgent
from ..core.llm_client import VisionLLMClient
from ..core.model_config import required_api_keys, resolve_model_names, validate_cascade_setup
from ..core.models import DetectionResult
from ..plugins.orchestrator import TaskPluginOrchestrator, merge_results
from ..plugins.registry import create_default_registry
from ..reporting import (
    ArtifactAuditor,
    build_generation_performance,
    build_user_action_report,
)
from ..utils.evaluation import build_experiment_report, evaluate_yolo_dirs, save_experiment_report
from ..utils.format_converter import LabelExportWriter
from ..utils.geometry import compute_result_consistency, consistency_metric_name, get_consistency_score
from ..utils.label_importer import find_image_path, import_labels_with_report
from ..utils.label_validator import summarize_validation, validate_result
from ..utils.result_metrics import count_result_labels, mean_result_confidence, uncertainty_score
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
        self.labeling_agent = None
        self.plugin_orchestrator = None

    def plan(self, request: str, supplied_plan: Optional[dict] = None) -> WorkflowPlan:
        return self.planner.plan(request, supplied_plan)

    @staticmethod
    def analyze_dataset(operation: OperationPlan, results: List[DetectionResult]) -> Dict[str, Any]:
        labels = []
        for result in results:
            labels.extend(item.label for item in result.classifications)
            labels.extend(item.label for item in result.boxes)
            labels.extend(item.label for item in result.segments)
            labels.extend(item.label for item in result.poses)
            labels.extend("text" for item in result.texts)
            labels.extend(item.label for item in result.tracks)
        counts = Counter(labels)
        total = sum(counts.values())
        distribution = {
            label: {
                "count": count,
                "percentage": (count / total * 100.0) if total else 0.0,
            }
            for label, count in sorted(counts.items())
        }
        suggestions = []
        if len(counts) > 1:
            max_label, max_count = counts.most_common(1)[0]
            min_label, min_count = min(counts.items(), key=lambda item: item[1])
            ratio = max_count / min_count if min_count else float("inf")
            if ratio >= operation.insight_imbalance_ratio:
                suggestions.append(
                    f"'{min_label}' 클래스가 '{max_label}' 대비 부족합니다. "
                    "희소 클래스 중심 증강 또는 추가 수집을 권장합니다."
                )
        return {
            "total_labels": total,
            "distribution": distribution,
            "suggestions": suggestions,
        }

    def prepare_generation(self, operation: OperationPlan) -> Dict[str, Any]:
        os.makedirs(operation.img_dir, exist_ok=True)
        os.makedirs(operation.out_dir, exist_ok=True)
        os.makedirs(operation.vis_dir, exist_ok=True)
        images = sorted(
            name for name in os.listdir(operation.img_dir)
            if name.lower().endswith((".png", ".jpg", ".jpeg"))
        )
        low_model, high_model = resolve_model_names(operation.low_model, operation.high_model)
        valid, messages = validate_cascade_setup(low_model, high_model, self.allow_same_model)
        if not valid:
            raise ValueError("; ".join(messages))
        missing_keys = [key for key in required_api_keys(low_model, high_model) if not os.getenv(key)]
        if missing_keys:
            raise ValueError(f"Missing API key(s): {', '.join(missing_keys)}")
        self._ensure_generation_clients(operation, low_model, high_model)
        return {"images": images, "low_model": low_model, "high_model": high_model, "warnings": messages}

    def _ensure_generation_clients(self, operation: OperationPlan, low_model: str, high_model: str) -> None:
        key = (
            low_model,
            high_model,
            operation.inference_count,
            operation.draft_temperature,
            operation.plugin_config,
            operation.plugin_fail_fast,
        )
        if key == self._generation_key:
            return
        self.low_client = VisionLLMClient(low_model)
        self.high_client = VisionLLMClient(high_model)
        self.labeling_agent = LabelingAgent(
            self.low_client,
            inference_count=operation.inference_count,
            temperature=operation.draft_temperature,
        )
        self.plugin_orchestrator = None
        if operation.plugin_config:
            plugins = create_default_registry().load_config(operation.plugin_config)
            self.plugin_orchestrator = TaskPluginOrchestrator(plugins, fail_fast=operation.plugin_fail_fast)
        self._generation_key = key

    def _ensure_generation_for_operation(self, operation: OperationPlan) -> None:
        low_model, high_model = resolve_model_names(operation.low_model, operation.high_model)
        if self._generation_key != (
            low_model,
            high_model,
            operation.inference_count,
            operation.draft_temperature,
            operation.plugin_config,
            operation.plugin_fail_fast,
        ):
            self._ensure_generation_clients(operation, low_model, high_model)

    def generate_drafts(self, image_path: str, operation: OperationPlan) -> Dict[str, Any]:
        self._ensure_generation_for_operation(operation)
        started = time.perf_counter()
        before = self.low_client.api_attempts
        drafts = self.labeling_agent.label(image_path, operation.prompt, operation.task_type)
        consistency = float(get_consistency_score(drafts))
        seed = drafts[0] if drafts else DetectionResult(task_type=operation.task_type)
        seed.consistency_score = consistency
        seed.mean_confidence = mean_result_confidence(seed)
        seed.uncertainty_score = uncertainty_score(consistency, seed.mean_confidence)
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
        reasons = []
        if (result.consistency_score or 0.0) < operation.threshold:
            reasons.append(f"consistency {result.consistency_score or 0.0:.3f} < {operation.threshold:.3f}")
        agreements = [
            float(record["agreement"])
            for record in plugin_records
            if record.get("status") == "ok" and record.get("agreement") is not None
        ]
        if agreements and min(agreements) < operation.threshold:
            reasons.append(f"specialist agreement {min(agreements):.3f} < {operation.threshold:.3f}")
        if issues:
            reasons.append(f"validation issues: {', '.join(issues[:3])}")
        return bool(reasons), "; ".join(reasons)

    def run_high_verification(
        self,
        image_path: str,
        operation: OperationPlan,
        specialist_result: DetectionResult,
    ) -> Dict[str, Any]:
        self._ensure_generation_for_operation(operation)
        started = time.perf_counter()
        before = self.high_client.api_attempts
        high_result = self.high_client.predict(
            image_path,
            operation.prompt,
            temperature=0.0,
            task_type=operation.task_type,
        )
        high_result.source_model = self.high_client.model_name
        agreement = compute_result_consistency(high_result, specialist_result)
        merged = merge_results(high_result, specialist_result)
        merged.plugin_scores.update(specialist_result.plugin_scores)
        merged.plugin_metadata.update(specialist_result.plugin_metadata)
        previous = specialist_result.consistency_score if specialist_result.consistency_score is not None else agreement
        merged.consistency_score = (previous + agreement) / 2
        merged.mean_confidence = mean_result_confidence(merged)
        merged.uncertainty_score = uncertainty_score(merged.consistency_score, merged.mean_confidence)
        if specialist_result.plugin_scores:
            merged.source_model = f"{self.high_client.model_name}+{'+'.join(specialist_result.plugin_scores)}"
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
            sources = result.plugin_metadata.get("conversion_sources", {}) if result.plugin_metadata else {}
            validations.append({
                "image": record["image"],
                "image_path": image_path,
                "input_paths": sources.get("paths", []),
                "issues": validate_result(result, image_path),
            })
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
        )
        auditor = ArtifactAuditor()
        converted = 0
        export_records = []
        exported_results = []
        skipped_validations = self._skipped_file_validations(input_summary or {})
        all_validations = [*validations, *skipped_validations]
        for record in records:
            issues = validation_map.get(record["image"], [])
            blocking = any(
                issue.startswith("missing_image:")
                or issue.startswith("image_open_failed:")
                or issue == "invalid_image_size"
                or issue == "empty_result"
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
        user_action_report = build_user_action_report(
            all_validations,
            export_records,
            artifact_issues,
            total_records=len(records) + len(skipped_validations),
        )
        report = {
            "report_version": "2.0",
            "action": "convert",
            "input": operation.input_path,
            "resolved_source_format": source_format,
            "input_summary": input_summary or {},
            "target_formats": writer.formats,
            "records_read": len(records),
            "records_converted": converted,
            "validation": summarize_validation(all_validations),
            "export_validation": {
                "failed_records": sum(bool(record["issues"]) for record in export_records),
                "artifact_issues": artifact_issues,
            },
            "user_action_report": user_action_report,
            "dataset_insight": self.analyze_dataset(operation, exported_results),
            "records": all_validations,
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

    @staticmethod
    def _skipped_file_validations(input_summary: Dict[str, Any]) -> List[Dict[str, Any]]:
        records = []
        for item in input_summary.get("skipped_files", []):
            path = item.get("path", "")
            reason = item.get("reason", "skipped_input_file")
            records.append({
                "image": os.path.basename(path) if path else "skipped_file",
                "image_path": "",
                "input_paths": [path] if path else [],
                "issues": [reason],
            })
        for item in input_summary.get("failed_files", []):
            path = item.get("path", "")
            error = item.get("error", "schema_read_failed")
            records.append({
                "image": os.path.basename(path) if path else "failed_file",
                "image_path": "",
                "input_paths": [path] if path else [],
                "issues": [f"schema_read_failed:{error}"],
            })
        return records

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
