import os
import time
import csv
import json
import argparse
from tqdm import tqdm
from dotenv import load_dotenv

from src.core.llm_client import VisionLLMClient
from src.agents.verification_agent import HierarchicalVerificationAgent
from src.agents.insight_agent import DatasetInsightAgent
from src.utils.format_converter import LabelExportWriter, normalize_label_formats
from src.utils.visualize import visualize_boxes
from src.utils.evaluation import evaluate_yolo_dirs
from src.plugins.orchestrator import TaskPluginOrchestrator
from src.plugins.registry import create_default_registry

load_dotenv()

DEFAULT_BEDROCK_LOW_MODEL_ID = "us.anthropic.claude-haiku-4-5-20251001-v1:0"
DEFAULT_BEDROCK_HIGH_MODEL_ID = "us.anthropic.claude-sonnet-4-5-20250929-v1:0"
SUPPORTED_TASK_TYPES = {
    "classification",
    "object_detection",
    "segmentation",
    "pose_estimation",
    "ocr",
    "tracking",
    "all",
}

def is_bedrock_model(model_name: str) -> bool:
    return model_name.startswith("bedrock:") or model_name.startswith("anthropic.claude")

def as_bedrock_model(model_id: str | None) -> str | None:
    if not model_id:
        return None
    return model_id if model_id.startswith("bedrock:") else f"bedrock:{model_id}"

def model_capacity_rank(model_name: str) -> int:
    normalized = model_name.lower()
    if "haiku" in normalized:
        return 1
    if "sonnet" in normalized:
        return 2
    if "opus" in normalized:
        return 3
    return 0

def resolve_model_names(args) -> tuple[str, str]:
    low_model = (
        args.low_model
        or os.getenv("LOW_MODEL")
        or as_bedrock_model(os.getenv("AWS_BEDROCK_LOW_MODEL_ID"))
    )
    high_model = (
        args.high_model
        or os.getenv("HIGH_MODEL")
        or as_bedrock_model(os.getenv("AWS_BEDROCK_HIGH_MODEL_ID"))
        or as_bedrock_model(os.getenv("AWS_BEDROCK_MODEL_ID"))
    )

    if not low_model and high_model and is_bedrock_model(high_model):
        low_model = as_bedrock_model(DEFAULT_BEDROCK_LOW_MODEL_ID)
    if not high_model and low_model and is_bedrock_model(low_model):
        high_model = as_bedrock_model(DEFAULT_BEDROCK_HIGH_MODEL_ID)

    return low_model or "gpt-4o-mini", high_model or "gpt-4o"

def validate_cascade_setup(low_model_name: str, high_model_name: str, allow_same_model: bool) -> bool:
    low_rank = model_capacity_rank(low_model_name)
    high_rank = model_capacity_rank(high_model_name)

    if low_model_name == high_model_name:
        print("[!] ERROR: LOW_MODEL and HIGH_MODEL are identical.")
        print("[!] This does not support the paper claim of a heterogeneous capacity cascade.")
        print("[!] For AWS Bedrock Claude, use a lightweight model for LOW_MODEL and a stronger model for HIGH_MODEL.")
        print(f"    Example LOW_MODEL=bedrock:{DEFAULT_BEDROCK_LOW_MODEL_ID}")
        print(f"    Example HIGH_MODEL=bedrock:{DEFAULT_BEDROCK_HIGH_MODEL_ID}")
        if allow_same_model:
            print("[!] Continuing because --allow_same_model was set.")
            return True
        return False

    if low_rank and high_rank and low_rank >= high_rank:
        print("[!] WARNING: LOW_MODEL does not appear lighter than HIGH_MODEL.")
        print("[!] Recommended order for the cascade is Haiku < Sonnet < Opus.")

    return True

def count_labels(result) -> int:
    return (
        len(result.classifications)
        + len(result.boxes)
        + len(result.segments)
        + len(result.poses)
        + len(result.texts)
        + len(result.tracks)
    )

def main():
    parser = argparse.ArgumentParser(description="Agentic Auto-Labeling System - Paper Experiment Mode")
    parser.add_argument("--img_dir", type=str, default="data/raw", help="Directory with images to label")
    parser.add_argument("--out_dir", type=str, default="data/labeled", help="Output directory for exported labels")
    parser.add_argument("--vis_dir", type=str, default="data/visualized", help="Output directory for visualizations")
    parser.add_argument("--prompt", type=str, default="Detect and classify all prominent objects in this image. Output strictly as JSON.", help="Labeling prompt")
    parser.add_argument(
        "--task_type",
        type=str,
        default="object_detection",
        choices=sorted(SUPPORTED_TASK_TYPES),
        help="Vision labeling task: classification, object_detection, segmentation, pose_estimation, ocr, tracking, or all",
    )
    parser.add_argument("--threshold", type=float, default=0.75, help="Consistency threshold for high-level model escalation")
    parser.add_argument("--low_model", type=str, default=None, help="Low-capacity draft model. Overrides LOW_MODEL env.")
    parser.add_argument("--high_model", type=str, default=None, help="High-capacity verification model. Overrides HIGH_MODEL env.")
    parser.add_argument("--inference_count", type=int, default=3, help="Repeated low-model inferences per image for consistency scoring")
    parser.add_argument("--draft_temperature", type=float, default=0.7, help="Temperature for repeated low-model draft inference")
    parser.add_argument("--allow_same_model", action="store_true", help="Allow LOW_MODEL and HIGH_MODEL to be identical for debugging only")
    parser.add_argument("--gt_dir", type=str, default=None, help="Optional directory with ground-truth YOLO labels for precision/recall evaluation")
    parser.add_argument("--eval_iou", type=float, default=0.5, help="IoU threshold for optional ground-truth evaluation")
    parser.add_argument(
        "--label_formats",
        type=str,
        default="yolo",
        help="Comma-separated export formats: yolo, pascal_voc, coco, vision_json, custom, or all",
    )
    parser.add_argument(
        "--custom_label_template",
        type=str,
        default=None,
        help="Template file for custom label export. Uses placeholders such as {image_name} and {objects_json}.",
    )
    parser.add_argument(
        "--custom_label_extension",
        type=str,
        default=".json",
        help="File extension for custom label exports.",
    )
    parser.add_argument(
        "--plugin_config",
        type=str,
        default=None,
        help="Optional JSON configuration for task-specific specialist model plugins.",
    )
    parser.add_argument(
        "--plugin_fail_fast",
        action="store_true",
        help="Stop the run when a specialist plugin fails instead of recording the error.",
    )
    args = parser.parse_args()

    os.makedirs(args.img_dir, exist_ok=True)
    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(args.vis_dir, exist_ok=True)

    low_model_name, high_model_name = resolve_model_names(args)
    if not validate_cascade_setup(low_model_name, high_model_name, args.allow_same_model):
        return

    try:
        requested_formats = args.label_formats
        if args.label_formats == "yolo" and args.task_type != "object_detection":
            requested_formats = "vision_json"
        label_formats = normalize_label_formats(requested_formats)
        label_writer = LabelExportWriter(
            args.out_dir,
            formats=label_formats,
            custom_template_path=args.custom_label_template,
            custom_extension=args.custom_label_extension,
        )
    except Exception as e:
        print(f"[!] ERROR: Invalid label export setup: {e}")
        return

    plugin_orchestrator = None
    if args.plugin_config:
        try:
            plugin_registry = create_default_registry()
            plugins = plugin_registry.load_config(args.plugin_config)
            plugin_orchestrator = TaskPluginOrchestrator(plugins, fail_fast=args.plugin_fail_fast)
        except Exception as e:
            print(f"[!] ERROR: Invalid plugin setup: {e}")
            return
    
    required_keys = []
    if "gpt" in low_model_name.lower() or "gpt" in high_model_name.lower():
        required_keys.append("OPENAI_API_KEY")
    if (
        ("claude" in low_model_name.lower() and not is_bedrock_model(low_model_name))
        or ("claude" in high_model_name.lower() and not is_bedrock_model(high_model_name))
    ):
        required_keys.append("ANTHROPIC_API_KEY")

    missing_keys = [key for key in required_keys if not os.getenv(key)]
    if missing_keys:
        print(f"[!] ERROR: Missing API key(s): {', '.join(missing_keys)}")
        print("[!] Please add them to .env and run the script again.")
        return

    try:
        low_client = VisionLLMClient(low_model_name)
        high_client = VisionLLMClient(high_model_name)
    except Exception as e:
        print(f"Error initializing API clients: {e}")
        return
        
    verifier = HierarchicalVerificationAgent(
        low_client,
        high_client,
        threshold=args.threshold,
        inference_count=args.inference_count,
        draft_temperature=args.draft_temperature,
    )
    insighter = DatasetInsightAgent()

    images = sorted(f for f in os.listdir(args.img_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg')))
    if not images:
        print(f"No images found in {args.img_dir}.")
        print("Running sample setup script automatically...")
        from src.utils.setup_samples import download_samples
        download_samples()
        images = sorted(f for f in os.listdir(args.img_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg')))

    print(f"\n🚀 Starting Agentic Auto-Labeling Experiment ({len(images)} images)")
    print(f"   - Low-Level (Draft): {low_model_name}")
    print(f"   - High-Level (Verify): {high_model_name}")
    print(f"   - Draft Repeats: {args.inference_count} at temperature {args.draft_temperature}")
    print(f"   - Task Type: {args.task_type}")
    print(f"   - Uncertainty Threshold: {verifier.threshold} (Escalate if Consistency < {verifier.threshold})")
    print(f"   - Label Formats: {', '.join(label_writer.formats)}")
    if plugin_orchestrator:
        print(f"   - Specialist Plugins: {', '.join(plugin_orchestrator.names) or 'none'}")
    print("-" * 60)
    
    # Metrics for Paper
    total_auto_time = 0.0
    manual_time_per_image = 45.0 # Literature benchmark
    escalation_count = 0
    total_objects_found = 0
    run_records = []

    # Process with progress bar
    pbar = tqdm(images, desc="Labeling", unit="img")
    for img_name in pbar:
        img_path = os.path.join(args.img_dir, img_name)
        
        start_time = time.time()
        low_attempts_before = low_client.api_attempts
        high_attempts_before = high_client.api_attempts
        try:
            # Core Agentic Workflow
            result, status = verifier.process(img_path, args.prompt, task_type=args.task_type)
            plugin_records = []
            if plugin_orchestrator:
                result, plugin_records = plugin_orchestrator.process(
                    img_path,
                    args.prompt,
                    args.task_type,
                    result,
                )
            insighter.add_result(result)
            
            # Save Labels & Visualizations
            label_paths = label_writer.save(result, img_path)
            vis_path = visualize_boxes(img_path, result, args.vis_dir)
            
            elapsed = time.time() - start_time
            total_auto_time += elapsed
            label_count = count_labels(result)
            total_objects_found += label_count
            
            if status == "Escalated":
                escalation_count += 1
                
            # Update progress bar text
            reliability = 1.0 - result.uncertainty_score if result.uncertainty_score is not None else 1.0
            pbar.set_postfix({"Status": status, "Reliability": f"{reliability:.2f}", "Labels": label_count})

            run_records.append({
                "image": img_name,
                "status": status,
                "source_model": result.source_model,
                "low_model": low_model_name,
                "high_model": high_model_name,
                "task_type": result.task_type,
                "objects": label_count,
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
                "plugin_records": json.dumps(plugin_records, ensure_ascii=False),
                "low_api_attempts": low_client.api_attempts - low_attempts_before,
                "high_api_attempts": high_client.api_attempts - high_attempts_before,
                "elapsed_sec": elapsed,
                "label_path": label_paths.get("yolo") or next(iter(label_paths.values()), ""),
                "label_paths": json.dumps(label_paths, ensure_ascii=False),
                "visualization_path": vis_path,
            })
            
        except Exception as e:
            print(f"\n[!] Error processing {img_name}: {e}")

    export_artifacts = label_writer.finalize()

    metrics_csv_path = os.path.join(args.out_dir, "run_metrics.csv")
    metrics_jsonl_path = os.path.join(args.out_dir, "run_metrics.jsonl")
    if run_records:
        with open(metrics_csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(run_records[0].keys()))
            writer.writeheader()
            writer.writerows(run_records)
        with open(metrics_jsonl_path, "w", encoding="utf-8") as f:
            for record in run_records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

    # Calculate Paper Metrics
    total_manual_time = len(images) * manual_time_per_image
    time_saved_pct = ((total_manual_time - total_auto_time) / total_manual_time) * 100 if total_manual_time > 0 else 0
    cost_reduction_pct = ((len(images) - escalation_count) / len(images)) * 100 if images else 0
    evaluation = None
    if args.gt_dir and os.path.isdir(args.gt_dir):
        if "yolo" not in label_writer.formats:
            print("\n[!] Skipping ground-truth evaluation because --gt_dir currently requires YOLO predictions.")
        else:
            evaluation = evaluate_yolo_dirs(args.out_dir, args.gt_dir, args.eval_iou)

    run_summary = {
        "images": len(images),
        "task_type": args.task_type,
        "label_formats": label_writer.formats,
        "total_labels": total_objects_found,
        "total_elapsed_sec": total_auto_time,
        "avg_elapsed_sec": total_auto_time / len(images) if images else 0.0,
        "manual_time_per_image_sec": manual_time_per_image,
        "estimated_manual_time_sec": total_manual_time,
        "time_saved_pct": time_saved_pct,
        "low_model": low_model_name,
        "high_model": high_model_name,
        "low_api_attempts": low_client.api_attempts,
        "high_api_attempts": high_client.api_attempts,
        "escalation_count": escalation_count,
        "escalation_rate": escalation_count / len(images) if images else 0.0,
        "cost_reduction_pct": cost_reduction_pct,
        "plugins": plugin_orchestrator.names if plugin_orchestrator else [],
        "evaluation": evaluation,
    }
    summary_path = os.path.join(args.out_dir, "run_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(run_summary, f, ensure_ascii=False, indent=2)

    print("\n" + "="*60)
    print(" 📊 ABLATION & PERFORMANCE REPORT (For Paper Table 1)")
    print("="*60)
    print(f"1. Dataset Processed:      {len(images)} images")
    print(f"2. Total Objects Found:    {total_objects_found} objects")
    print(f"\n[Time Efficiency]")
    print(f" - Estimated Manual Time:  {total_manual_time:.1f} sec (Assuming 45s/img)")
    print(f" - Agentic Pipeline Time:  {total_auto_time:.1f} sec")
    print(f" => Time Saved:            {time_saved_pct:.1f}% ▼ (Significant reduction)")
    
    print(f"\n[Hierarchical Cascade Efficiency (Ablation)]")
    print(f" - Low Model:              {low_model_name}")
    print(f" - High Model:             {high_model_name}")
    print(f" - Low API Attempts:       {low_client.api_attempts}")
    print(f" - High API Attempts:      {high_client.api_attempts}")
    print(f" - Low-Level Only Count:   {len(images) - escalation_count} imgs (Passed Consistency Check)")
    print(f" - Escalated to High-Level: {escalation_count} imgs (Failed Consistency Check)")
    print(f" => API Cost Saved:        {cost_reduction_pct:.1f}% ▼ compared to using High-Level model for all images")
    
    if evaluation:
        print(f"\n[Ground Truth Evaluation @ IoU {args.eval_iou:.2f}]")
        print(f" - Precision:             {evaluation['precision']:.3f}")
        print(f" - Recall:                {evaluation['recall']:.3f}")
        print(f" - F1:                    {evaluation['f1']:.3f}")
        print(f" - Mean IoU:              {evaluation['mean_iou']:.3f}")

    print(insighter.get_report())
    
    print("\n" + "="*60)
    print(f"📁 Outputs Saved:")
    print(f" - Labels:         {os.path.abspath(args.out_dir)}")
    print(f" - Visualizations: {os.path.abspath(args.vis_dir)} (Use these for Paper Figures!)")
    if export_artifacts:
        for name, path in export_artifacts.items():
            print(f" - {name}: {os.path.abspath(path)}")
    if run_records:
        print(f" - Run Metrics CSV: {os.path.abspath(metrics_csv_path)}")
        print(f" - Run Metrics JSONL: {os.path.abspath(metrics_jsonl_path)}")
        print(f" - Run Summary JSON: {os.path.abspath(summary_path)}")
    print("="*60)

if __name__ == "__main__":
    main()
