import argparse
import os
import sys

from src.workflow.models import OperationPlan
from src.workflow.runtime import WorkflowRuntime


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
    parser.add_argument("--classes", "--classes_path", dest="classes", default=None, help="Optional classes.txt for YOLO input")
    parser.add_argument("--custom_label_template", default=None, help="Template path for custom output")
    parser.add_argument("--custom_label_extension", default=".json", help="Extension for custom output")
    parser.add_argument(
        "--duplicate_iou",
        type=float,
        default=0.85,
        help="IoU threshold for merging duplicate labels from mixed input formats",
    )
    parser.add_argument("--strict", action="store_true", help="Skip records with validation issues")
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Approve excluding records with validation issues without an interactive prompt.",
    )
    args = parser.parse_args()
    if not 0.0 < args.duplicate_iou <= 1.0:
        parser.error("--duplicate_iou must be greater than 0 and at most 1")

    operation = OperationPlan(
        action="convert",
        input_path=args.input,
        img_dir=args.img_dir,
        out_dir=args.out_dir,
        source_format=args.source_format,
        formats=args.target_formats,
        classes_path=args.classes,
        custom_label_template=args.custom_label_template,
        custom_label_extension=args.custom_label_extension,
        duplicate_iou=args.duplicate_iou,
        strict=args.strict,
        require_approval=False,
    )
    runtime = WorkflowRuntime()
    loaded = runtime.load_conversion(operation, args.source_format)
    records = loaded["records"]
    validations = runtime.validate_conversion(operation, records)
    issue_records = [record for record in validations if record.get("issues")]
    if issue_records and not args.yes:
        print("Validation issues were found:")
        for record in issue_records:
            print(f" - {record['image']}: {', '.join(record['issues'])}")
        print("These records will be excluded from conversion.")
        try:
            answer = input("Continue excluding problematic records? [y/N]: ").strip().lower()
        except EOFError:
            answer = ""
        if answer not in {"y", "yes"}:
            print("Conversion cancelled. No output was exported.")
            sys.exit(2)
    repaired = runtime.repair_conversion(records)
    report = runtime.export_conversion(
        operation,
        repaired,
        validations,
        loaded["resolved_source_format"],
        loaded["input_summary"],
    )

    print(f"Converted {report['records_converted']}/{report['records_read']} records")
    print(f"Validation failed records: {report['validation']['failed_records']}")
    print(f"Export failed records: {report['export_validation']['failed_records']}")
    print(f"Report: {os.path.abspath(report['report_path'])}")
    print(f"User action report: {os.path.abspath(report['user_action_report_path'])}")


if __name__ == "__main__":
    main()
