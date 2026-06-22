import re
import os
from collections import Counter, defaultdict
from typing import Any, Dict, Iterable, List, Optional


ISSUE_CATALOG = {
    "empty_result": ("critical", "input_data", "라벨 데이터가 비어 있습니다", "원본 데이터 확인 후 다시 라벨링하세요."),
    "missing_image": ("critical", "file_system", "이미지 파일을 찾을 수 없습니다", "이미지 경로를 확인하거나 파일을 복원하세요."),
    "image_open_failed": ("critical", "file_format", "이미지 파일을 열 수 없습니다", "이미지 손상 여부와 지원 형식을 확인하세요."),
    "invalid_image_size": ("critical", "file_format", "이미지 크기가 유효하지 않습니다", "올바른 이미지 파일로 교체하세요."),
    "missing_label": ("high", "label_quality", "클래스 라벨이 누락되었습니다", "누락된 객체의 클래스 라벨을 지정하세요."),
    "missing_name": ("high", "pose", "키포인트 이름이 누락되었습니다", "키포인트 이름을 지정하세요."),
    "missing_text": ("high", "ocr", "OCR 문자열이 비어 있습니다", "텍스트 영역을 다시 인식하거나 수정하세요."),
    "missing_track_id": ("high", "tracking", "Track ID가 누락되었습니다", "객체 추적 ID를 다시 생성하세요."),
    "coordinate_out_of_range": ("high", "label_quality", "좌표가 0~1 범위를 벗어났습니다", "픽셀 좌표와 정규화 좌표 설정을 확인하세요."),
    "invalid_box_order": ("high", "label_quality", "박스 좌표 순서가 잘못되었습니다", "xmin < xmax, ymin < ymax가 되도록 수정하세요."),
    "confidence_out_of_range": ("medium", "label_quality", "Confidence가 0~1 범위를 벗어났습니다", "Confidence 값을 0~1로 정규화하세요."),
    "too_few_points": ("high", "segmentation", "Polygon point가 3개 미만입니다", "세 점 이상의 polygon으로 다시 라벨링하세요."),
    "empty_keypoints": ("high", "pose", "Pose keypoint가 비어 있습니다", "키포인트를 추가하거나 잘못된 pose를 제거하세요."),
    "missing_output_path": ("critical", "output_format", "출력 경로가 생성되지 않았습니다", "출력 포맷과 디렉터리 설정을 확인하세요."),
    "missing_output_file": ("critical", "output_format", "출력 파일이 생성되지 않았습니다", "쓰기 권한과 출력 경로를 확인하세요."),
    "empty_output_file": ("critical", "output_format", "출력 파일이 비어 있습니다", "입력 라벨과 선택한 출력 포맷의 호환성을 확인하세요."),
    "output_parse_failed": ("critical", "output_format", "출력 파일을 다시 읽을 수 없습니다", "생성된 파일의 구조와 인코딩을 확인하세요."),
    "no_label_rows": ("critical", "output_format", "YOLO 라벨 행이 없습니다", "Bounding box가 있는지 확인하거나 Vision JSON을 사용하세요."),
    "invalid_row_shape": ("high", "output_format", "YOLO 행 구조가 올바르지 않습니다", "각 행이 class_id와 좌표 4개인지 확인하세요."),
    "non_numeric_row": ("high", "output_format", "YOLO 행에 숫자가 아닌 값이 있습니다", "YOLO 좌표와 class_id를 숫자로 수정하세요."),
    "negative_class_id": ("high", "output_format", "YOLO class ID가 음수입니다", "0 이상의 class ID를 사용하세요."),
    "invalid_box_size": ("high", "output_format", "출력 박스 크기가 유효하지 않습니다", "너비와 높이가 0보다 크도록 수정하세요."),
    "no_objects": ("critical", "output_format", "Pascal VOC 객체가 없습니다", "Bounding box가 있는지 확인하세요."),
    "invalid_object": ("high", "output_format", "Pascal VOC 객체 구조가 잘못되었습니다", "name과 bndbox를 확인하세요."),
    "non_numeric_box": ("high", "output_format", "Pascal VOC 좌표가 숫자가 아닙니다", "좌표 값을 숫자로 수정하세요."),
    "no_images": ("critical", "output_format", "COCO image 항목이 없습니다", "원본 이미지 연결을 확인하세요."),
    "no_annotations": ("critical", "output_format", "COCO annotation이 없습니다", "선택한 task가 COCO에서 표현 가능한지 확인하세요."),
    "no_categories": ("critical", "output_format", "COCO category가 없습니다", "클래스 라벨을 확인하세요."),
    "invalid_bbox": ("high", "output_format", "COCO bbox 구조가 잘못되었습니다", "bbox가 [x, y, width, height]인지 확인하세요."),
    "non_numeric_bbox": ("high", "output_format", "COCO bbox가 숫자가 아닙니다", "bbox 값을 숫자로 수정하세요."),
    "invalid_bbox_size": ("high", "output_format", "COCO bbox 크기가 유효하지 않습니다", "width와 height가 0보다 크도록 수정하세요."),
    "missing_category_id": ("high", "output_format", "COCO category_id가 누락되었습니다", "annotation의 category_id를 확인하세요."),
    "no_label_records": ("critical", "output_format", "Vision JSON에 유효한 라벨이 없습니다", "입력 결과와 task를 확인하세요."),
    "no_labels": ("critical", "output_format", "클래스 목록이 비어 있습니다", "생성된 객체의 클래스 이름을 확인하세요."),
    "unrecognized_txt_schema": ("medium", "input_data", "라벨로 해석할 수 없는 txt 파일입니다", "라벨 파일이 아니면 제외하고, YOLO 라벨이면 class_id x_center y_center width height 형식으로 수정하세요."),
    "unrecognized_json_schema": ("medium", "input_data", "라벨로 해석할 수 없는 JSON 파일입니다", "라벨 파일이 아니면 입력 폴더에서 제외하고, COCO JSON이면 images/annotations/categories 구조를 확인하세요."),
    "unrecognized_yaml_schema": ("medium", "input_data", "지원하지 않는 YAML 구조입니다", "YAML이 데이터셋 설정 파일인지 확인하고 라벨 입력 폴더와 분리하세요."),
    "yaml_config_not_label_source": ("medium", "input_data", "YAML 설정 파일은 라벨 변환 대상이 아닙니다", "dataset.yaml은 설정 파일로 보관하고 라벨 입력 폴더에서는 제외하세요."),
    "yaml_invalid_class_id": ("high", "input_data", "YAML 클래스 ID가 올바르지 않습니다", "dataset.yaml의 names 키가 0 이상의 정수인지 확인하세요."),
    "yaml_path_missing": ("high", "file_system", "YAML 내부 데이터 경로를 찾을 수 없습니다", "dataset.yaml의 path/val 경로가 현재 PC의 실제 데이터 경로와 일치하는지 확인하세요."),
    "schema_read_failed": ("high", "input_data", "파일 구조를 읽는 중 오류가 발생했습니다", "파일 인코딩, JSON/XML/YAML 문법, 깨진 문자를 확인하세요."),
}


def issue_code(issue: str) -> str:
    for part in str(issue).split(":"):
        normalized = re.sub(r"\[[^]]*\]", "", part).split(".")[-1].strip()
        if normalized in ISSUE_CATALOG:
            return normalized
    return "unknown"


def categorize_issue(issue: str) -> Dict[str, Any]:
    code = issue_code(issue)
    severity, category, message, action = ISSUE_CATALOG.get(
        code,
        ("unknown", "unknown", str(issue), "이슈와 workflow 로그를 수동으로 검토하세요."),
    )
    return {
        "code": code,
        "severity": severity,
        "category": category,
        "message": message,
        "user_action": action,
        "original_issue": str(issue),
    }


def _path_from_issue(issue: str) -> Optional[str]:
    text = str(issue)
    code = issue_code(text)
    prefixes = {
        "missing_image",
        "missing_output_file",
        "empty_output_file",
        "output_parse_failed",
        "no_label_rows",
        "coordinate_out_of_range",
        "invalid_box_size",
        "yaml_path_missing",
        "yaml_invalid_class_id",
        "schema_read_failed",
    }
    if code in prefixes and text.startswith(f"{code}:"):
        return text[len(code) + 1:]
    return None


def _affected_path(code: str, issue: str, metadata: Dict[str, Any]) -> str:
    input_paths = metadata.get("input_paths") or []
    output_paths = metadata.get("output_paths") or {}
    if code.startswith("yaml_") or code.startswith("unrecognized_") or code == "schema_read_failed":
        return input_paths[0] if input_paths else metadata.get("image_path", "")
    explicit = _path_from_issue(issue)
    if explicit:
        return explicit
    if code == "missing_image":
        return metadata.get("image_path", "")
    if code in {"empty_output_file", "missing_output_file", "missing_output_path"}:
        return next((path for path in output_paths.values() if path), "")
    if code in {"empty_result", "coordinate_out_of_range", "invalid_box_order", "missing_label"}:
        return input_paths[0] if input_paths else metadata.get("image_path", "")
    return input_paths[0] if input_paths else metadata.get("image_path", "")


def _fix_instruction(code: str, path: str) -> str:
    target_name = os.path.basename(path) if path else ""
    target = f" `{target_name}`" if target_name else ""
    instructions = {
        "empty_result": f"라벨 파일{target}에 객체 라벨 행 또는 annotation을 추가하거나, 해당 샘플을 제외하고 다시 변환하세요.",
        "missing_image": (
            f"이미지 파일{target}을 찾을 수 없습니다. 이미지가 실제로 없으면 복원하고, "
            "라벨 파일명만 바꾼 경우 라벨 파일명을 실제 이미지 파일명과 같은 stem으로 되돌리세요. "
            "Pascal VOC/COCO/JSON 계열이면 라벨 내부 filename/file_name도 실제 이미지명과 일치해야 합니다."
        ),
        "coordinate_out_of_range": f"라벨 파일{target}의 좌표를 0~1 정규화 YOLO 좌표로 수정하세요. 픽셀 좌표라면 이미지 width/height로 나눠 정규화하세요.",
        "invalid_box_order": f"라벨 파일{target}에서 xmin < xmax, ymin < ymax가 되도록 박스 좌표 순서를 수정하세요.",
        "empty_output_file": f"출력 파일{target}이 비어 있습니다. task와 출력 포맷 호환성을 확인하고, segmentation/pose/OCR이면 Vision JSON 또는 COCO 등 표현 가능한 포맷으로 다시 변환하세요.",
        "no_label_rows": f"YOLO 출력 파일{target}에 class_id x_center y_center width height 행이 생기도록 bounding box 라벨을 추가하세요.",
        "missing_output_file": f"출력 파일{target}이 생성되지 않았습니다. 출력 디렉터리 권한과 경로를 확인한 뒤 다시 실행하세요.",
        "unrecognized_txt_schema": f"txt 파일{target}이 라벨 형식이 아닙니다. 라벨 파일이 아니면 입력 폴더에서 제외하고, YOLO 라벨이면 `class_id x_center y_center width height` 형식으로 수정하세요.",
        "unrecognized_json_schema": f"JSON 파일{target}이 라벨 형식이 아닙니다. 이전 결과 리포트 파일이면 입력 폴더에서 제외하세요.",
        "yaml_config_not_label_source": f"YAML 파일{target}은 라벨 변환 대상이 아닙니다. dataset 설정 파일로만 사용하고 라벨 입력 폴더에서는 제외하세요.",
        "yaml_invalid_class_id": f"YAML 파일{target}의 클래스 번호를 0 이상의 정수로 수정하세요.",
        "yaml_path_missing": f"YAML 파일{target}의 path/val 경로가 현재 PC에 존재하지 않습니다. 실제 이미지/라벨 폴더 경로로 수정하세요.",
        "schema_read_failed": f"파일{target}의 문법 또는 인코딩을 확인하세요. 깨진 JSON/XML/YAML이면 수정하거나 입력에서 제외하세요.",
    }
    return instructions.get(code, f"파일{target}의 원본 이슈를 확인하고 라벨 구조 또는 출력 포맷 설정을 수정하세요.")


def _record_report(image: str, issues: Iterable[str], metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    metadata = metadata or {}
    details = []
    for issue in dict.fromkeys(issues):
        detail = categorize_issue(issue)
        path = _affected_path(detail["code"], str(issue), metadata)
        detail["affected_path"] = path
        detail["fix_instruction"] = _fix_instruction(detail["code"], path)
        details.append(detail)
    severity_counts = Counter(item["severity"] for item in details)
    category_counts = Counter(item["category"] for item in details)
    if severity_counts["critical"]:
        status = "blocked"
    elif severity_counts["high"]:
        status = "needs_attention"
    else:
        status = "warning"
    priority = [
        item["fix_instruction"]
        for item in details
    ]
    return {
        "image": image,
        "status": status,
        "total_issues": len(details),
        "issues_by_severity": dict(severity_counts),
        "issues_by_category": dict(category_counts),
        "priority_actions": list(dict.fromkeys(priority))[:3],
        "detailed_issues": details,
    }


def build_user_action_report(
    validation_records: List[Dict[str, Any]],
    export_records: Optional[List[Dict[str, Any]]] = None,
    artifact_issues: Optional[List[str]] = None,
    total_records: Optional[int] = None,
) -> Dict[str, Any]:
    issues_by_image = defaultdict(list)
    metadata_by_image: Dict[str, Dict[str, Any]] = defaultdict(dict)
    for record in [*validation_records, *(export_records or [])]:
        image = str(record.get("image", "unknown"))
        issues_by_image[image].extend(record.get("issues", []))
        if record.get("image_path"):
            metadata_by_image[image]["image_path"] = record["image_path"]
        if record.get("input_paths"):
            metadata_by_image[image]["input_paths"] = record["input_paths"]
        if record.get("paths"):
            metadata_by_image[image]["output_paths"] = record["paths"]
    problem_images = {
        image: issues for image, issues in issues_by_image.items() if issues
    }
    records = [
        _record_report(image, issues, metadata_by_image.get(image, {}))
        for image, issues in sorted(problem_images.items())
    ]
    artifact_details = []
    for issue in (artifact_issues or []):
        detail = categorize_issue(issue)
        path = _path_from_issue(issue) or ""
        detail["affected_path"] = path
        detail["fix_instruction"] = _fix_instruction(detail["code"], path)
        artifact_details.append(detail)
    all_details = [item for record in records for item in record["detailed_issues"]] + artifact_details
    code_counts = Counter(item["code"] for item in all_details)
    category_counts = Counter(item["category"] for item in all_details)

    total = total_records if total_records is not None else len(validation_records)
    affected = min(len(problem_images), total)
    clean = max(total - affected, 0)
    has_critical = any(item["severity"] == "critical" for item in all_details)
    if not all_details:
        status = "success"
    elif clean > 0:
        status = "partial_success"
    else:
        status = "needs_review" if has_critical else "partial_success"

    top_issues = [
        {
            "issue_type": code,
            "count": count,
            "user_action": ISSUE_CATALOG.get(code, (None, None, None, "수동 검토하세요."))[3],
        }
        for code, count in code_counts.most_common(5)
    ]
    recommended = []
    if category_counts["input_data"]:
        recommended.append("비어 있거나 유효 라벨이 없는 입력 데이터를 먼저 확인하세요.")
    if category_counts["file_system"] or category_counts["file_format"]:
        recommended.append("누락되거나 손상된 이미지 파일을 먼저 복구하세요.")
    if category_counts["label_quality"]:
        recommended.append("좌표 범위와 클래스 라벨을 검토한 뒤 다시 실행하세요.")
    if category_counts["output_format"]:
        recommended.append("선택한 task와 출력 포맷의 호환성 및 생성 artifact를 확인하세요.")
    if not recommended:
        recommended.append("모든 데이터가 정상적으로 처리되었습니다.")

    return {
        "report_version": "2.0",
        "status": status,
        "summary": {
            "total_records": total,
            "clean": clean,
            "needs_review": affected,
            "blocked": sum(record["status"] == "blocked" for record in records),
            "needs_attention": sum(record["status"] == "needs_attention" for record in records),
            "warning": sum(record["status"] == "warning" for record in records),
            "artifact_issues": len(artifact_details),
        },
        "completion_rate": (clean / total * 100.0) if total else 100.0,
        "top_issues": top_issues,
        "recommended_actions": recommended,
        "artifact_issues": artifact_details,
        "detailed_records": records,
    }
