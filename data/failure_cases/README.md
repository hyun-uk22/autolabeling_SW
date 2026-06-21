# Failure Case Fixtures

These fixtures are for demo and regression checks around conversion validation.

## empty_yolo

Input YOLO txt exists but has no label rows.

Expected:
- `records_converted`: 0
- `user_action_report.status`: `needs_review`
- issue: `empty_result`

```powershell
python convert_labels.py --input data\failure_cases\empty_yolo\labels --img_dir data\failure_cases\empty_yolo\images --out_dir data\failure_cases\empty_yolo\converted --source_format auto --target_formats yolo --yes
```

## missing_image

YOLO label exists, but the corresponding image is missing.

Expected:
- `records_converted`: 0
- issue: `missing_image`

```powershell
python convert_labels.py --input data\failure_cases\missing_image\labels --img_dir data\failure_cases\missing_image\images --out_dir data\failure_cases\missing_image\converted --source_format auto --target_formats yolo --yes
```

## bad_yolo_coordinates

YOLO row has numeric values, but coordinates are outside the normalized 0-1 range.

Expected:
- `records_converted`: 0
- issues: `coordinate_out_of_range`, `empty_output_file`

```powershell
python convert_labels.py --input data\failure_cases\bad_yolo_coordinates\labels --img_dir data\failure_cases\bad_yolo_coordinates\images --out_dir data\failure_cases\bad_yolo_coordinates\converted --source_format auto --target_formats yolo --yes
```

## segmentation_to_yolo

Vision JSONL contains segmentation only, but the requested output is YOLO detection txt.

Expected:
- `records_converted`: 0
- issue: `empty_output_file`

```powershell
python convert_labels.py --input data\failure_cases\segmentation_to_yolo\labels --img_dir data\failure_cases\segmentation_to_yolo\images --out_dir data\failure_cases\segmentation_to_yolo\converted --source_format auto --target_formats yolo --yes
```
