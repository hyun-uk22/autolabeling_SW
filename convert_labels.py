import argparse
import json
import os

from src.utils.format_converter import LabelExportWriter
from src.utils.label_importer import find_image_path, import_labels_with_report
from src.utils.label_validator import summarize_validation, validate_result


def main():
    parser = argparse.ArgumentParser(description="Convert existing vision labels into target formats.")
    parser.add_argument("--input", required=True, help="Input label file or directory")
    parser.add_argument("--img_dir", required=True, help="Directory containing source images")
    parser.add_argument("--out_dir", required=True, help="Output directory for converted labels")
    parser.add_argument(
        "--source_format",
        default="auto",
        choices=["auto", "yolo", "pascal_voc", "coco", "vision_json", "csv", "generic_json"],
        help="Input label format. Use auto for heuristic detection.",
    )
    parser.add_argument(
        "--target_formats",
        default="yolo",
        help="Comma-separated target formats: yolo, pascal_voc, coco, vision_json, custom, all",
    )
    parser.add_argument("--classes", default=None, help="Optional classes.txt for YOLO input")
    parser.add_argument("--custom_label_template", default=None, help="Template path for custom output")
    parser.add_argument("--custom_label_extension", default=".json", help="Extension for custom output")
    parser.add_argument(
        "--duplicate_iou",
        type=float,
        default=0.85,
        help="IoU threshold for merging duplicate labels from mixed input formats",
    )
    parser.add_argument("--strict", action="store_true", help="Skip records with validation issues")
    args = parser.parse_args()
    if not 0.0 < args.duplicate_iou <= 1.0:
        parser.error("--duplicate_iou must be greater than 0 and at most 1")

    import_batch = import_labels_with_report(
        args.input,
        args.img_dir,
        source_format=args.source_format,
        classes_path=args.classes,
        duplicate_iou=args.duplicate_iou,
    )
    records = import_batch.records
    writer = LabelExportWriter(
        args.out_dir,
        formats=args.target_formats,
        custom_template_path=args.custom_label_template,
        custom_extension=args.custom_label_extension,
    )

    validation_records = []
    converted = 0
    for image_name, result in records:
        image_path = find_image_path(args.img_dir, image_name)
        issues = validate_result(result, image_path)
        validation_records.append({"image": image_name, "issues": issues})
        blocking_issues = [
            issue for issue in issues
            if issue.startswith("missing_image:")
            or issue.startswith("image_open_failed:")
            or issue == "invalid_image_size"
        ]
        if blocking_issues or (args.strict and issues):
            continue
        writer.save(result, image_path)
        converted += 1

    artifacts = writer.finalize()
    validation_summary = summarize_validation(validation_records)
    report = {
        "input": args.input,
        "source_format": args.source_format,
        "input_summary": import_batch.report,
        "target_formats": writer.formats,
        "records_read": len(records),
        "records_converted": converted,
        "validation": validation_summary,
        "artifacts": artifacts,
        "records": validation_records,
    }

    os.makedirs(args.out_dir, exist_ok=True)
    report_path = os.path.join(args.out_dir, "conversion_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"Converted {converted}/{len(records)} records")
    print(f"Validation failed records: {validation_summary['failed_records']}")
    print(f"Report: {os.path.abspath(report_path)}")


if __name__ == "__main__":
    main()
