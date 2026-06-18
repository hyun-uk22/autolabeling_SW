# Changelog

## 2026-05-22 13:20:06 +09:00

- Renamed project instruction file from `Agents.md` to `AGENTS.md` so Codex can detect it consistently.
- Added this `changelog.md` file to follow the project instruction that requested changes be accumulated with timestamps.

## 2026-06-17 19:48:36 +09:00

- Added multi-format label export support for YOLO, Pascal VOC XML, COCO JSON, and template-based custom formats.
- Added CLI options `--label_formats`, `--custom_label_template`, and `--custom_label_extension`.
- Updated run metrics to record both a representative `label_path` and full format-specific `label_paths`.
- Documented the new export formats, custom template placeholders, and updated output behavior in `README.md`.

## 2026-06-17 20:50:33 +09:00

- Reviewed `README.md` against the current code workflow.
- Updated the CLI option description for `--out_dir` to reflect multi-format label exports.
- Expanded the workflow section to include model resolution, export setup validation, API key checks, per-image processing, export finalization, metrics writing, and YOLO-only ground-truth evaluation behavior.
- Updated the JSONL metrics example to include the `label_paths` field.

## 2026-06-17 20:57:32 +09:00

- Added `--task_type` support for `classification`, `object_detection`, `segmentation`, `pose_estimation`, `ocr`, `tracking`, and `all`.
- Extended internal label models to store classifications, bounding boxes, polygon segments, pose keypoints, OCR text regions, and tracking instances.
- Updated LLM prompting and response parsing to use task-specific JSON schemas.
- Extended consistency scoring beyond bbox IoU with segmentation bbox IoU and Jaccard-style matching for non-geometric labels.
- Added `vision_json` export for preserving all task outputs and expanded COCO export to include segmentation polygons.
- Updated visualization and dataset insight reporting for the new task result types.
- Documented multi-task labeling behavior, new CLI options, and Vision JSONL output in `README.md`.

## 2026-06-17 21:04:15 +09:00

- Added `.gitignore` for GitHub publishing.
- Excluded local secrets, virtual environments, Python caches, generated datasets, generated labels, visualizations, metrics, model checkpoints, logs, and temporary files.
- Kept `.env.example` trackable while ignoring real `.env` files.

## 2026-06-18 14:25:03 +09:00

- Added `convert_labels.py` for converting existing external labels into supported target formats.
- Added `src/utils/label_importer.py` with YOLO, Pascal VOC, COCO, Vision JSONL, CSV, and generic JSON import support.
- Added `src/utils/label_validator.py` to detect empty conversions, missing images, invalid coordinates, malformed boxes, incomplete polygons, and missing label fields.
- Extended detection evaluation with GT-only false negative handling and F1 score.
- Added run summary JSON output from `main.py`.
- Added `evaluate_experiments.py` and experiment report utilities for low-only, high-only, and cascade ablation comparisons.
- Updated `.gitignore` and `README.md` for conversion outputs, validation reports, and quantitative experiment reports.

## 2026-06-18 15:49:30 +09:00

- Added an internal task specialist plugin architecture with a common `VisionTaskPlugin` interface, dynamic registry, JSON configuration loader, and result orchestrator.
- Added built-in lazy-loading adapters for CLIP classification, Grounding DINO detection, SAM segmentation, Ultralytics pose estimation, EasyOCR, and Ultralytics tracking.
- Added cross-model result merging, plugin confidence/agreement scores, plugin provenance metadata, and uncertainty recalculation.
- Added `--plugin_config` and `--plugin_fail_fast` options to `main.py` while preserving the existing VLM-only default path.
- Added `configs/plugins.example.json` and `requirements-specialists.txt` for optional specialist model setup.
- Sorted input image names so frame-sequence tracking has deterministic ordering.
- Documented built-in task chains, configuration, failure behavior, and external plugin registration in `README.md`.
