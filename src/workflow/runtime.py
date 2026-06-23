import csv
import json
import os
import tempfile
import time
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv
from PIL import Image, ImageEnhance

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
from ..utils.format_converter import LabelExportWriter, resolve_export_formats
from ..utils.geometry import calculate_iou, compute_result_consistency, consistency_metric_name
from ..utils import json_io
from ..utils.label_importer import (
    extract_class_names_from_text,
    find_image_path,
    import_labels_with_report,
    load_classes,
)
from ..utils.label_validator import summarize_validation, validate_result
from ..utils.result_metrics import count_result_labels
from ..utils.visualize import visualize_boxes
from .models import OperationPlan, WorkflowPlan
from .planner import WorkflowPlanner
from .schema_repair import repair_result, schema_candidates

load_dotenv()


ALLOWED_ADVISOR_FIELDS = {
    "box_threshold_delta",
    "text_threshold_delta",
    "nms_iou_delta",
    "min_confidence_delta",
    "prompt_prefix",
    "prompt_suffix",
    "augmentation",
}


def _clamp(value: Any, low: float, high: float, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return max(low, min(high, number))


def _result_labels(result: DetectionResult) -> List[str]:
    labels = []
    labels.extend(item.label for item in result.classifications)
    labels.extend(item.label for item in result.boxes)
    labels.extend(item.label for item in result.segments)
    labels.extend(item.label for item in result.poses)
    labels.extend(item.label for item in result.tracks)
    return list(dict.fromkeys(label for label in labels if label))


def _bbox_agreement(base: DetectionResult, rerun: DetectionResult, iou_threshold: float = 0.5) -> Dict[str, Any]:
    matches = []
    used = set()
    for base_index, base_box in enumerate(base.boxes):
        best_index = None
        best_iou = 0.0
        for rerun_index, rerun_box in enumerate(rerun.boxes):
            if rerun_index in used or rerun_box.label != base_box.label:
                continue
            iou = calculate_iou(
                [base_box.xmin, base_box.ymin, base_box.xmax, base_box.ymax],
                [rerun_box.xmin, rerun_box.ymin, rerun_box.xmax, rerun_box.ymax],
            )
            if iou > best_iou:
                best_iou = iou
                best_index = rerun_index
        if best_index is not None and best_iou >= iou_threshold:
            used.add(best_index)
            matches.append({"base_index": base_index, "rerun_index": best_index, "label": base_box.label, "iou": best_iou})
    base_count = len(base.boxes)
    rerun_count = len(rerun.boxes)
    match_count = len(matches)
    denominator = max(base_count, rerun_count, 1)
    pseudo_precision = match_count / rerun_count if rerun_count else 0.0
    pseudo_recall = match_count / base_count if base_count else 0.0
    pseudo_f1 = (
        2 * pseudo_precision * pseudo_recall / (pseudo_precision + pseudo_recall)
        if pseudo_precision + pseudo_recall else 0.0
    )
    return {
        "bbox_iou_threshold": iou_threshold,
        "base_boxes": base_count,
        "rerun_boxes": rerun_count,
        "matched_boxes": match_count,
        "missing_boxes": max(0, base_count - match_count),
        "new_boxes": max(0, rerun_count - match_count),
        "mean_matched_iou": sum(item["iou"] for item in matches) / match_count if match_count else 0.0,
        "agreement": match_count / denominator,
        "pseudo_precision": pseudo_precision,
        "pseudo_recall": pseudo_recall,
        "pseudo_f1": pseudo_f1,
        "matches": matches[:20],
    }


def _llm_review_queue(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    review_rows = []
    for record in records:
        llm_consistency = (record.get("specialist_consistency") or {}).get("llm_consistency") or {}
        if not llm_consistency.get("enabled") or not llm_consistency.get("review_required"):
            continue
        comparisons = llm_consistency.get("comparisons") or []
        agreements = [
            (comparison.get("bbox_agreement") or {}).get("agreement")
            for comparison in comparisons
            if (comparison.get("bbox_agreement") or {}).get("agreement") is not None
        ]
        review_rows.append({
            "image": record.get("image", ""),
            "mode": llm_consistency.get("mode"),
            "threshold": llm_consistency.get("threshold"),
            "mean_bbox_agreement": (
                sum(agreements) / len(agreements)
                if agreements else llm_consistency.get("mean_bbox_agreement")
            ),
            "comparisons": [
                {
                    "level": comparison.get("level"),
                    "model": comparison.get("model"),
                    "bbox_agreement": (comparison.get("bbox_agreement") or {}).get("agreement"),
                    "mean_matched_iou": (comparison.get("bbox_agreement") or {}).get("mean_matched_iou"),
                    "pseudo_precision": (comparison.get("bbox_agreement") or {}).get("pseudo_precision"),
                    "pseudo_recall": (comparison.get("bbox_agreement") or {}).get("pseudo_recall"),
                    "pseudo_f1": (comparison.get("bbox_agreement") or {}).get("pseudo_f1"),
                    "review_required": comparison.get("review_required"),
                }
                for comparison in comparisons
            ],
        })
    return review_rows


def _confidence_summary(values: List[float]) -> Dict[str, Any]:
    if not values:
        return {"count": 0, "mean": None, "min": None, "max": None, "low_confidence_count": 0}
    return {
        "count": len(values),
        "mean": sum(values) / len(values),
        "min": min(values),
        "max": max(values),
        "low_confidence_count": sum(value < 0.5 for value in values),
    }


def _class_counts(result: DetectionResult) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for label in _result_labels(result):
        counts[label] = (
            sum(item.label == label for item in result.boxes)
            + sum(item.label == label for item in result.segments)
            + sum(item.label == label for item in result.classifications)
            + sum(item.label == label for item in result.poses)
            + sum(item.label == label for item in result.tracks)
        )
    return counts


class WorkflowRuntime:
    def __init__(self, planner_model: Optional[str] = None, allow_same_model: bool = False):
        self.planner = WorkflowPlanner(planner_model)
        self.allow_same_model = allow_same_model
        self._generation_key = None
        self._plugin_key = None
        self._vlm_key = None
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
        image_exts = (".png", ".jpg", ".jpeg", ".webp", ".bmp")
        images = []
        for root, _, names in os.walk(operation.img_dir):
            for name in names:
                if not name.lower().endswith(image_exts):
                    continue
                path = os.path.join(root, name)
                images.append(os.path.relpath(path, operation.img_dir))
        images = sorted(images)
        low_model, high_model = resolve_model_names(operation.low_model, operation.high_model)
        messages = []
        self._ensure_plugins(operation)
        if operation.generation_strategy == "vlm_first":
            messages = self._ensure_vlm_clients(operation, low_model, high_model)
        else:
            messages.append("specialist_first: specialist model results are generated before VLM fallback")
        return {
            "images": images,
            "low_model": low_model,
            "high_model": high_model,
            "warnings": messages,
            "plugin_prepare_records": self.plugin_prepare_records,
        }

    def _candidate_labels_for_generation(self, operation: OperationPlan) -> List[str]:
        labels = load_classes(operation.classes_path)
        if not labels:
            labels = extract_class_names_from_text(operation.prompt)
        return list(dict.fromkeys(label for label in labels if label))

    def _ensure_plugins(self, operation: OperationPlan) -> None:
        candidate_labels = self._candidate_labels_for_generation(operation)
        key = (
            operation.task_type,
            operation.plugin_config,
            operation.plugin_fail_fast,
            tuple(candidate_labels),
        )
        if key == self._plugin_key:
            return
        plugins = load_generation_plugins(operation.plugin_config, candidate_labels=candidate_labels)
        self.plugin_orchestrator = TaskPluginOrchestrator(plugins, fail_fast=operation.plugin_fail_fast)
        self.plugin_prepare_records = self.plugin_orchestrator.prepare(operation.task_type)
        self._plugin_key = key

    def _ensure_vlm_clients(self, operation: OperationPlan, low_model: str, high_model: str) -> List[str]:
        key = (
            low_model,
            high_model,
            operation.task_type,
            operation.inference_count,
            operation.draft_temperature,
            operation.plugin_config,
            operation.plugin_fail_fast,
        )
        if key == self._vlm_key:
            return []
        valid, messages = validate_cascade_setup(low_model, high_model, self.allow_same_model)
        if not valid:
            raise ValueError("; ".join(messages))
        missing_keys = [key for key in required_api_keys(low_model, high_model) if not os.getenv(key)]
        if missing_keys:
            raise ValueError(f"Missing API key(s): {', '.join(missing_keys)}")
        self.low_client = VisionLLMClient(low_model)
        self.high_client = VisionLLMClient(high_model)
        self.verification_agent = HierarchicalVerificationAgent(
            self.low_client,
            self.high_client,
            threshold=operation.threshold,
            inference_count=operation.inference_count,
            draft_temperature=operation.draft_temperature,
        )
        self._vlm_key = key
        return messages

    def _ensure_generation_for_operation(self, operation: OperationPlan) -> None:
        low_model, high_model = resolve_model_names(operation.low_model, operation.high_model)
        self._ensure_plugins(operation)
        self._ensure_vlm_clients(operation, low_model, high_model)

    def _current_specialist_parameters(self) -> Dict[str, Any]:
        parameters: Dict[str, Any] = {}
        if not self.plugin_orchestrator:
            return parameters
        for plugin in self.plugin_orchestrator.plugins:
            if plugin.plugin_name in {"grounding_dino", "grounded_sam2"}:
                parameters[plugin.plugin_name] = {
                    "box_threshold": float(plugin.config.get("box_threshold", 0.45)),
                    "text_threshold": float(plugin.config.get("text_threshold", 0.30)),
                    "nms_iou": float(plugin.config.get("nms_iou", 0.60)),
                    "min_confidence": float(plugin.config.get("min_confidence", 0.20)),
                }
        return parameters

    def _first_pass_report(
        self,
        result: DetectionResult,
        plugin_records: List[dict],
        operation: OperationPlan,
    ) -> Dict[str, Any]:
        confidences = []
        confidences.extend(item.confidence for item in result.boxes)
        confidences.extend(item.confidence for item in result.segments)
        confidences.extend(item.confidence for item in result.classifications)
        confidences.extend(item.confidence for item in result.poses)
        confidences.extend(item.confidence for item in result.tracks)
        confidences.extend(item.confidence for item in result.texts)
        return {
            "task_type": operation.task_type,
            "generation_strategy": operation.generation_strategy,
            "total_labels": count_result_labels(result),
            "boxes": len(result.boxes),
            "segments": len(result.segments),
            "classifications": len(result.classifications),
            "poses": len(result.poses),
            "texts": len(result.texts),
            "tracks": len(result.tracks),
            "class_counts": _class_counts(result),
            "confidence": _confidence_summary(confidences),
            "source_model": result.source_model,
            "consistency_score": result.consistency_score,
            "mean_confidence": result.mean_confidence,
            "uncertainty_score": result.uncertainty_score,
            "plugin_records": plugin_records,
            "current_parameters": self._current_specialist_parameters(),
        }

    def _default_specialist_patch(self) -> Dict[str, Any]:
        return {
            "box_threshold_delta": -0.05,
            "text_threshold_delta": 0.03,
            "nms_iou_delta": -0.05,
            "min_confidence_delta": 0.0,
            "prompt_suffix": " Prefer clearly visible objects only.",
            "augmentation": {"enabled": False, "brightness": 1.0, "contrast": 1.0},
        }

    def _sanitize_advisor_patch(self, patch: Dict[str, Any], immutable_labels: List[str]) -> Dict[str, Any]:
        if not isinstance(patch, dict):
            return self._default_specialist_patch()
        blocked = [
            key for key in patch
            if key not in ALLOWED_ADVISOR_FIELDS or "label" in key.lower() or "class" in key.lower()
        ]
        sanitized = self._default_specialist_patch()
        sanitized.update({
            "box_threshold_delta": _clamp(patch.get("box_threshold_delta"), -0.20, 0.20, sanitized["box_threshold_delta"]),
            "text_threshold_delta": _clamp(patch.get("text_threshold_delta"), -0.20, 0.20, sanitized["text_threshold_delta"]),
            "nms_iou_delta": _clamp(patch.get("nms_iou_delta"), -0.20, 0.20, sanitized["nms_iou_delta"]),
            "min_confidence_delta": _clamp(patch.get("min_confidence_delta"), -0.20, 0.20, sanitized["min_confidence_delta"]),
            "prompt_prefix": str(patch.get("prompt_prefix", ""))[:200],
            "prompt_suffix": str(patch.get("prompt_suffix", sanitized["prompt_suffix"]))[:200],
        })
        augmentation = patch.get("augmentation") if isinstance(patch.get("augmentation"), dict) else {}
        sanitized["augmentation"] = {
            "enabled": bool(augmentation.get("enabled", False)),
            "brightness": _clamp(augmentation.get("brightness"), 0.80, 1.20, 1.0),
            "contrast": _clamp(augmentation.get("contrast"), 0.80, 1.20, 1.0),
        }
        if blocked:
            sanitized["blocked_fields"] = blocked
        sanitized["immutable_labels"] = immutable_labels
        return sanitized

    def _advisor_prompt(
        self,
        operation: OperationPlan,
        base_result: DetectionResult,
        immutable_labels: List[str],
        first_pass_report: Dict[str, Any],
    ) -> str:
        return (
            "Suggest a conservative JSON patch for one specialist vision-model rerun. "
            "Do not create labels or boxes. Do not change, add, remove, translate, or synonymize classes. "
            f"Immutable classes: {immutable_labels}. "
            "Allowed keys only: box_threshold_delta, text_threshold_delta, nms_iou_delta, "
            "min_confidence_delta, prompt_prefix, prompt_suffix, augmentation. "
            "Deltas must be between -0.20 and 0.20. augmentation may include enabled, brightness, contrast. "
            f"Current task: {operation.task_type}. Current label count: {count_result_labels(base_result)}. "
            "Use this first-pass specialist report as reference only; do not edit boxes or labels directly: "
            f"{json.dumps(first_pass_report, ensure_ascii=False)}. "
            "Return only JSON."
        )

    def _advisor_patch(
        self,
        image_path: str,
        operation: OperationPlan,
        base_result: DetectionResult,
        immutable_labels: List[str],
        first_pass_report: Dict[str, Any],
    ) -> Dict[str, Any]:
        if operation.specialist_advisor_mode == "none":
            return self._sanitize_advisor_patch({}, immutable_labels)
        low_model, high_model = resolve_model_names(operation.low_model, operation.high_model)
        self._ensure_vlm_clients(operation, low_model, high_model)
        prompt = self._advisor_prompt(operation, base_result, immutable_labels, first_pass_report)
        patches = []
        if operation.specialist_advisor_mode in {"low", "both"}:
            patches.append(self.low_client.complete_json(image_path, prompt, temperature=0.0))
        if operation.specialist_advisor_mode in {"high", "both"}:
            review_prompt = prompt
            if patches:
                review_prompt += f"\nLow-model patch candidate: {json.dumps(patches[-1], ensure_ascii=False)}"
            patches.append(self.high_client.complete_json(image_path, review_prompt, temperature=0.0))
        return self._sanitize_advisor_patch(patches[-1] if patches else {}, immutable_labels)

    def _plugin_config_overrides(self, patch: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
        def adjusted(current: float, delta_key: str) -> float:
            return max(0.0, min(1.0, current + float(patch.get(delta_key, 0.0))))

        config = {
            "box_threshold": adjusted(0.45, "box_threshold_delta"),
            "text_threshold": adjusted(0.30, "text_threshold_delta"),
            "nms_iou": adjusted(0.60, "nms_iou_delta"),
            "min_confidence": adjusted(0.20, "min_confidence_delta"),
        }
        return {
            "grounding_dino": dict(config),
            "grounded_sam2": dict(config),
        }

    def _augmented_image_path(self, image_path: str, patch: Dict[str, Any]) -> Tuple[str, Optional[str]]:
        augmentation = patch.get("augmentation") or {}
        if not augmentation.get("enabled"):
            return image_path, None
        image = Image.open(image_path).convert("RGB")
        image = ImageEnhance.Brightness(image).enhance(float(augmentation.get("brightness", 1.0)))
        image = ImageEnhance.Contrast(image).enhance(float(augmentation.get("contrast", 1.0)))
        handle = tempfile.NamedTemporaryFile(prefix="autolabel-rerun-", suffix=".jpg", delete=False)
        handle.close()
        image.save(handle.name)
        return handle.name, handle.name

    def _specialist_consistency_rerun(
        self,
        image_path: str,
        operation: OperationPlan,
        base_result: DetectionResult,
        first_pass_report: Dict[str, Any],
    ) -> Dict[str, Any]:
        if operation.specialist_consistency_runs <= 0 or count_result_labels(base_result) <= 0:
            return {"enabled": False, "reason": "disabled_or_empty"}
        immutable_labels = self._candidate_labels_for_generation(operation) or _result_labels(base_result)
        patch = self._advisor_patch(image_path, operation, base_result, immutable_labels, first_pass_report)
        rerun_prompt = f"{patch.get('prompt_prefix', '')} {operation.prompt} {patch.get('prompt_suffix', '')}".strip()
        rerun_image, temporary_path = self._augmented_image_path(image_path, patch)
        try:
            rerun_result, rerun_records = self.plugin_orchestrator.process(
                rerun_image,
                rerun_prompt,
                operation.task_type,
                DetectionResult(task_type=operation.task_type),
                config_overrides=self._plugin_config_overrides(patch),
            )
        finally:
            if temporary_path and os.path.exists(temporary_path):
                os.unlink(temporary_path)
        consistency = compute_result_consistency(base_result, rerun_result)
        bbox = _bbox_agreement(base_result, rerun_result)
        return {
            "enabled": True,
            "advisor_mode": operation.specialist_advisor_mode,
            "runs": 1,
            "patch": patch,
            "result": rerun_result.model_dump(),
            "records": rerun_records,
            "result_consistency": consistency,
            "bbox_agreement": bbox,
        }

    def _llm_consistency_rerun(
        self,
        image_path: str,
        operation: OperationPlan,
        base_result: DetectionResult,
    ) -> Dict[str, Any]:
        mode = operation.llm_consistency_mode
        if mode == "none" or count_result_labels(base_result) <= 0:
            return {"enabled": False, "reason": "disabled_or_empty"}
        low_model, high_model = resolve_model_names(operation.low_model, operation.high_model)
        self._ensure_vlm_clients(operation, low_model, high_model)
        selected = []
        if mode in {"low", "both"}:
            selected.append(("low", self.low_client))
        if mode in {"high", "both"}:
            selected.append(("high", self.high_client))

        comparisons = []
        for level, client in selected:
            before = client.api_attempts
            started = time.perf_counter()
            llm_result = client.predict(
                image_path,
                operation.prompt,
                temperature=0.0,
                task_type=operation.task_type,
            )
            llm_result.source_model = client.model_name
            bbox = _bbox_agreement(base_result, llm_result)
            consistency = compute_result_consistency(base_result, llm_result)
            agreement = bbox.get("agreement", consistency)
            review_required = agreement < operation.threshold
            comparisons.append({
                "level": level,
                "model": client.model_name,
                "api_attempts": client.api_attempts - before,
                "elapsed_sec": time.perf_counter() - started,
                "result": llm_result.model_dump(),
                "result_consistency": consistency,
                "bbox_agreement": bbox,
                "review_required": review_required,
                "threshold": operation.threshold,
            })

        agreements = [
            item["bbox_agreement"].get("agreement")
            for item in comparisons
            if item.get("bbox_agreement", {}).get("agreement") is not None
        ]
        return {
            "enabled": True,
            "mode": mode,
            "threshold": operation.threshold,
            "comparisons": comparisons,
            "mean_bbox_agreement": sum(agreements) / len(agreements) if agreements else None,
            "review_required": any(item.get("review_required") for item in comparisons),
        }

    def generate_drafts(self, image_path: str, operation: OperationPlan) -> Dict[str, Any]:
        self._ensure_generation_for_operation(operation)
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
        self._ensure_plugins(operation)
        if not self.plugin_orchestrator:
            return {
                "result": result,
                "records": [],
                "elapsed_sec": 0.0,
                "first_pass_report": self._first_pass_report(result, [], operation),
                "specialist_consistency": {"enabled": False},
            }
        started = time.perf_counter()
        merged, records = self.plugin_orchestrator.process(
            image_path,
            operation.prompt,
            operation.task_type,
            result,
        )
        first_pass_report = self._first_pass_report(merged, records, operation)
        specialist_consistency = self._specialist_consistency_rerun(image_path, operation, merged, first_pass_report)
        llm_consistency = self._llm_consistency_rerun(image_path, operation, merged)
        if llm_consistency.get("enabled"):
            specialist_consistency = dict(specialist_consistency)
            specialist_consistency["llm_consistency"] = llm_consistency
        return {
            "result": merged,
            "records": records,
            "elapsed_sec": time.perf_counter() - started,
            "first_pass_report": first_pass_report,
            "specialist_consistency": specialist_consistency,
        }

    def needs_high_verification(
        self,
        operation: OperationPlan,
        result: DetectionResult,
        plugin_records: List[dict],
        issues: List[str],
    ) -> Tuple[bool, str]:
        self._ensure_generation_for_operation(operation)
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
            resolved_formats = resolve_export_formats(result, formats, operation.task_type)
            label_paths = writer.save(result, image_path, formats=resolved_formats)
            export_records.append({
                "image": record["image"],
                "paths": label_paths,
                "issues": auditor.audit_record(label_paths),
            })
            vis_path = visualize_boxes(image_path, result, operation.vis_dir)
            first_pass_report = record.get("first_pass_report") or {}
            specialist_consistency = record.get("specialist_consistency") or {}
            bbox_agreement = specialist_consistency.get("bbox_agreement") or {}
            llm_consistency = specialist_consistency.get("llm_consistency") or {}
            llm_comparisons = llm_consistency.get("comparisons") or []
            llm_agreements = [
                (item.get("bbox_agreement") or {}).get("agreement")
                for item in llm_comparisons
                if (item.get("bbox_agreement") or {}).get("agreement") is not None
            ]
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
                "plugin_scores": json_io.dumps(result.plugin_scores, ensure_ascii=False),
                "plugin_records": json_io.dumps(record.get("plugin_records", []), ensure_ascii=False),
                "first_pass_report": json_io.dumps(first_pass_report, ensure_ascii=False),
                "first_pass_total_labels": first_pass_report.get("total_labels"),
                "first_pass_mean_confidence": (first_pass_report.get("confidence") or {}).get("mean"),
                "first_pass_low_confidence_count": (first_pass_report.get("confidence") or {}).get("low_confidence_count"),
                "specialist_consistency_enabled": bool(specialist_consistency.get("enabled")),
                "specialist_advisor_mode": specialist_consistency.get("advisor_mode", operation.specialist_advisor_mode),
                "specialist_result_consistency": specialist_consistency.get("result_consistency"),
                "specialist_bbox_agreement": bbox_agreement.get("agreement"),
                "specialist_mean_matched_iou": bbox_agreement.get("mean_matched_iou"),
                "specialist_rerun_records": json_io.dumps(specialist_consistency.get("records", []), ensure_ascii=False),
                "specialist_rerun_patch": json_io.dumps(specialist_consistency.get("patch", {}), ensure_ascii=False),
                "llm_consistency_enabled": bool(llm_consistency.get("enabled")),
                "llm_consistency_mode": llm_consistency.get("mode"),
                "llm_consistency_threshold": llm_consistency.get("threshold"),
                "llm_mean_bbox_agreement": (
                    sum(llm_agreements) / len(llm_agreements)
                    if llm_agreements else None
                ),
                "llm_review_required": bool(llm_consistency.get("review_required")),
                "llm_consistency_comparisons": json_io.dumps(llm_comparisons, ensure_ascii=False),
                "validation_issues": json_io.dumps(record.get("issues", []), ensure_ascii=False),
                "low_api_attempts": record.get("low_api_attempts", 0),
                "high_api_attempts": record.get("high_api_attempts", 0),
                "elapsed_sec": record.get("elapsed_sec", 0.0),
                "label_path": label_paths.get("yolo") or next(iter(label_paths.values()), ""),
                "label_paths": json_io.dumps(label_paths, ensure_ascii=False),
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
                    f.write(json_io.dumps(row, ensure_ascii=False) + "\n")
        evaluation = None
        if operation.gt_dir and os.path.isdir(operation.gt_dir) and "yolo" in writer.used_formats:
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
        specialist_consistency_records = [
            record.get("specialist_consistency")
            for record in records
            if (record.get("specialist_consistency") or {}).get("enabled")
        ]
        specialist_bbox_agreements = [
            (item.get("bbox_agreement") or {}).get("agreement")
            for item in specialist_consistency_records
            if (item.get("bbox_agreement") or {}).get("agreement") is not None
        ]
        llm_consistency_records = [
            item.get("llm_consistency")
            for item in (
                record.get("specialist_consistency") or {}
                for record in records
            )
            if (item or {}).get("enabled")
        ]
        llm_bbox_agreements = [
            (comparison.get("bbox_agreement") or {}).get("agreement")
            for item in llm_consistency_records
            for comparison in item.get("comparisons", [])
            if (comparison.get("bbox_agreement") or {}).get("agreement") is not None
        ]
        llm_review_rows = _llm_review_queue(records)
        llm_review_required = len(llm_review_rows)
        first_pass_reports = [
            record.get("first_pass_report")
            for record in records
            if record.get("first_pass_report")
        ]
        first_pass_label_counts = [
            int(report.get("total_labels", 0))
            for report in first_pass_reports
        ]
        first_pass_confidences = [
            (report.get("confidence") or {}).get("mean")
            for report in first_pass_reports
            if (report.get("confidence") or {}).get("mean") is not None
        ]
        summary = {
            "report_version": "2.0",
            "action": "generate",
            "images": len(records),
            "task_type": operation.task_type,
            "image_dir": operation.img_dir,
            "output_dir": operation.out_dir,
            "classes_path": operation.classes_path,
            "consistency_metric": consistency_metric_name(operation.task_type),
            "formats": list(dict.fromkeys(writer.used_formats or formats)),
            "total_labels": sum(row["objects"] for row in metric_rows),
            "total_elapsed_sec": total_elapsed,
            "low_api_attempts": low_attempts,
            "high_api_attempts": high_attempts,
            "escalation_count": escalation_count,
            "plugins": plugins,
            "plugin_prepare_records": self.plugin_prepare_records,
            "first_pass_report": {
                "images": len(first_pass_reports),
                "total_labels": sum(first_pass_label_counts),
                "mean_labels_per_image": (
                    sum(first_pass_label_counts) / len(first_pass_label_counts)
                    if first_pass_label_counts else 0.0
                ),
                "mean_confidence": (
                    sum(first_pass_confidences) / len(first_pass_confidences)
                    if first_pass_confidences else None
                ),
                "records": [
                    {
                        "image": record["image"],
                        "report": record.get("first_pass_report"),
                    }
                    for record in records
                    if record.get("first_pass_report")
                ],
            },
            "specialist_consistency": {
                "enabled_images": len(specialist_consistency_records),
                "advisor_mode": operation.specialist_advisor_mode,
                "requested_runs": operation.specialist_consistency_runs,
                "mean_bbox_agreement": (
                    sum(specialist_bbox_agreements) / len(specialist_bbox_agreements)
                    if specialist_bbox_agreements else None
                ),
                "llm_enabled_images": len(llm_consistency_records),
                "llm_mode": operation.llm_consistency_mode,
                "llm_review_required_images": llm_review_required,
                "llm_review_images": [row["image"] for row in llm_review_rows if row.get("image")],
                "llm_review_records": llm_review_rows,
                "llm_mean_bbox_agreement": (
                    sum(llm_bbox_agreements) / len(llm_bbox_agreements)
                    if llm_bbox_agreements else None
                ),
            },
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
        json_io.dump_file(user_action_report, user_action_path, ensure_ascii=False, indent=2)
        json_io.dump_file(summary, summary_path, ensure_ascii=False, indent=2)
        return summary

    def load_conversion(self, operation: OperationPlan, source_format: str) -> Dict[str, Any]:
        batch = import_labels_with_report(
            operation.input_path,
            operation.img_dir,
            source_format=source_format,
            classes_path=operation.classes_path,
            duplicate_iou=operation.duplicate_iou,
            custom_mapping_spec=operation.custom_label_mapping,
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
                issue.startswith("image_open_failed:")
                or issue == "invalid_image_size"
                for issue in issues
            )
            if blocking or (operation.strict and issues):
                continue
            image_path = find_image_path(operation.img_dir, record["image"])
            result = DetectionResult.model_validate(record["result"])
            resolved_formats = resolve_export_formats(result, writer.formats, operation.task_type or result.task_type)
            paths = writer.save(result, image_path, formats=resolved_formats)
            export_issues = auditor.audit_record(paths)
            missing_formats = sorted(set(resolved_formats) - set(paths))
            if missing_formats:
                reason = "missing_image" if any(issue.startswith("missing_image:") for issue in issues) else "missing_required_metadata"
                export_issues.extend(
                    f"{fmt}:{reason}:{image_path}"
                    for fmt in missing_formats
                )
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
            "image_dir": operation.img_dir,
            "output_dir": operation.out_dir,
            "classes_path": operation.classes_path,
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
        json_io.dump_file(user_action_report, user_action_path, ensure_ascii=False, indent=2)
        json_io.dump_file(report, report_path, ensure_ascii=False, indent=2)
        return report

    def conversion_schema_candidates(self, operation: OperationPlan) -> List[str]:
        return schema_candidates(operation.input_path, operation.source_format)

    def evaluate(self, operation: OperationPlan) -> Dict[str, Any]:
        report = build_experiment_report(operation.runs, gt_dir=operation.gt_dir)
        paths = save_experiment_report(report, operation.out_dir)
        return {"action": "evaluate", "rows": report, "artifacts": paths}

    def prepare_model_dataset(self, operation: OperationPlan) -> Dict[str, Any]:
        model_name = (operation.model_name or "custom").lower()
        usage_mode = operation.usage_mode or "library"
        framework = (operation.framework or "custom").lower()
        purpose = operation.dataset_purpose or "training"
        out_dir = operation.out_dir

        if usage_mode == "official_repo":
            directories = [
                "datasets",
                "configs",
                "checkpoints",
                "outputs",
            ]
            required_files = ["dataset_layout.json"]
            layout = operation.output_layout or f"{model_name}_official_repo_{framework}"
        elif model_name == "segformer" and framework == "mmsegmentation":
            directories = [
                "images/train",
                "images/val",
                "annotations/train",
                "annotations/val",
            ]
            required_files = ["dataset_layout.json", "dataset_meta.json"]
            layout = "segformer_mmsegmentation"
        elif model_name == "segformer" and framework == "huggingface":
            directories = [
                "images/train",
                "images/validation",
                "masks/train",
                "masks/validation",
            ]
            required_files = ["dataset_layout.json", "dataset_info.json"]
            layout = "segformer_huggingface"
        elif model_name in {"maskdino", "mask2former"}:
            directories = [
                "images/train",
                "images/val",
                "annotations",
            ]
            required_files = [
                "dataset_layout.json",
                "annotations/instances_train.json",
                "annotations/instances_val.json",
            ]
            layout = f"{model_name}_{framework}"
        else:
            directories = ["images", "annotations"]
            required_files = ["dataset_layout.json"]
            layout = operation.output_layout or f"{model_name}_{framework}"

        created_directories = []
        for relative in directories:
            path = os.path.join(out_dir, relative)
            os.makedirs(path, exist_ok=True)
            created_directories.append(path)

        split = {
            "train": operation.split_train,
            "val": operation.split_val,
            "test": operation.split_test,
        }
        metadata = {
            "action": "prepare_model_dataset",
            "model_name": model_name,
            "usage_mode": usage_mode,
            "framework": framework,
            "repo_url": operation.repo_url,
            "repo_path": operation.repo_path,
            "purpose": purpose,
            "task_type": operation.task_type,
            "source_format": operation.source_format,
            "layout": layout,
            "split": split,
            "directories": directories,
            "required_files": required_files,
            "notes": [
                "이 단계는 모델이 요구하는 폴더 구조와 준비 리포트를 생성합니다.",
                "official_repo 방식은 repo를 자동 clone하지 않고, 공식 repo 기준으로 사용할 데이터셋 준비 구조와 메타정보만 기록합니다.",
                "실제 라벨 변환은 입력 라벨 형식과 모델 요구 라벨 형식이 호환될 때 별도 변환 단계로 수행해야 합니다.",
            ],
        }
        os.makedirs(out_dir, exist_ok=True)
        report_path = os.path.join(out_dir, "dataset_layout.json")
        json_io.dump_file(metadata, report_path, ensure_ascii=False, indent=2)

        return {
            **metadata,
            "created_directories": created_directories,
            "artifacts": {"layout_report": report_path},
            "report_path": report_path,
        }

    def save_history(self, output_dir: str, history: List[Dict[str, Any]]) -> str:
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, "workflow_history.json")
        json_io.dump_file(history, path, ensure_ascii=False, indent=2)
        return path
