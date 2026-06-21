import argparse
import json
import os

from src.agents.conversion_agent import ConversionQualityAgent
from src.utils.format_converter import LabelExportWriter
from src.utils.label_importer import find_image_path, import_labels
from src.utils.label_validator import summarize_validation


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
    parser.add_argument("--strict", action="store_true", help="Skip records with validation issues")
    args = parser.parse_args()

    records = import_labels(
        args.input,
        args.img_dir,
        source_format=args.source_format,
        classes_path=args.classes,
    )
    writer = LabelExportWriter(
        args.out_dir,
        formats=args.target_formats,
        custom_template_path=args.custom_label_template,
        custom_extension=args.custom_label_extension,
    )
    conversion_agent = ConversionQualityAgent()

    validation_records = []
    export_records = []
    converted = 0
    for image_name, result in records:
        image_path = find_image_path(args.img_dir, image_name)
        repaired_result = conversion_agent.repair_detection_result(result)
        issues = conversion_agent.validate_input(repaired_result, image_path)
        blocking_issues = conversion_agent.blocking_input_issues(issues, strict=args.strict)
        validation_records.append({
            "image": image_name,
            "issues": issues,
            "blocking_issues": blocking_issues,
        })
        if blocking_issues or (args.strict and issues):
            continue
        resolved_formats = conversion_agent.resolve_export_formats(repaired_result, writer.formats)
        label_paths = writer.save(repaired_result, image_path, formats=resolved_formats)
        export_issues = conversion_agent.audit_record_exports(label_paths)
        export_records.append({
            "image": image_name,
            "paths": label_paths,
            "issues": export_issues,
            "recovery": conversion_agent.summarize_recovery(writer.formats, resolved_formats, export_issues),
        })
        if export_issues:
            continue
        converted += 1

    artifacts = writer.finalize()
    artifact_issues = conversion_agent.audit_final_artifacts(artifacts)
    validation_summary = summarize_validation(validation_records)
    report = {
        "input": args.input,
        "source_format": args.source_format,
        "target_formats": writer.formats,
        "records_read": len(records),
        "records_converted": converted,
        "validation": validation_summary,
        "export_validation": {
            "failed_records": sum(bool(record["issues"]) for record in export_records),
            "artifact_issues": artifact_issues,
        },
        "artifacts": artifacts,
        "records": validation_records,
        "exports": export_records,
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
