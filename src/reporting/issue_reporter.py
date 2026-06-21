import re
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
}

ISSUE_SUGGESTIONS = {
    "empty_result": [
        "입력 라벨 파일에 객체 annotation이 실제로 있는지 확인하세요.",
        "이미지만 있고 라벨이 없는 샘플이라면 변환 대상에서 제외하세요.",
    ],
    "missing_image": [
        "라벨에 기록된 이미지 파일명과 실제 파일명이 일치하는지 확인하세요.",
        "누락된 이미지를 원본 데이터셋에서 복원한 뒤 다시 변환하세요.",
    ],
    "image_open_failed": [
        "이미지가 손상되지 않았는지 확인하세요.",
        "지원 형식(PNG, JPG, JPEG, WEBP, BMP)으로 다시 저장하세요.",
    ],
    "invalid_image_size": [
        "이미지 width와 height가 0보다 큰지 확인하세요.",
        "손상되었거나 비정상적인 이미지 파일은 교체하세요.",
    ],
    "missing_label": [
        "클래스명이 비어 있는 객체를 찾아 실제 클래스명으로 채우세요.",
        "YOLO 입력이라면 data.yaml 또는 classes.txt의 클래스 매핑을 확인하세요.",
    ],
    "coordinate_out_of_range": [
        "정규화 좌표는 0~1 범위여야 합니다.",
        "픽셀 좌표를 사용했다면 이미지 width/height로 나눠 정규화하세요.",
    ],
    "invalid_box_order": [
        "xmin은 xmax보다 작아야 하고 ymin은 ymax보다 작아야 합니다.",
        "좌표 순서가 뒤집힌 박스를 수정하거나 제거하세요.",
    ],
    "confidence_out_of_range": [
        "Confidence는 0~1 범위의 실수여야 합니다.",
        "퍼센트 값이라면 100으로 나눠 저장하세요.",
    ],
    "too_few_points": [
        "Polygon은 최소 3개 point가 필요합니다.",
        "점이 부족한 segmentation은 다시 라벨링하거나 제외하세요.",
    ],
    "empty_keypoints": [
        "Pose task라면 최소 한 개 이상의 keypoint가 필요합니다.",
        "잘못 생성된 pose 객체는 삭제하거나 keypoint를 추가하세요.",
    ],
    "no_label_rows": [
        "YOLO txt에 class_id와 bbox 좌표 행이 있는지 확인하세요.",
        "bbox가 없는 task라면 Vision JSON 등 표현 가능한 출력 포맷을 선택하세요.",
    ],
    "invalid_row_shape": [
        "YOLO 행은 class_id, x_center, y_center, width, height 순서여야 합니다.",
        "세그멘테이션/포즈 YOLO라면 현재 변환 task와 호환되는지 확인하세요.",
    ],
    "non_numeric_row": [
        "YOLO 좌표와 class_id는 숫자여야 합니다.",
        "클래스명은 txt 행이 아니라 data.yaml 또는 classes.txt에 기록하세요.",
    ],
    "negative_class_id": [
        "YOLO class_id는 0 이상의 정수여야 합니다.",
        "클래스 매핑 파일과 라벨 txt의 class_id 범위를 맞추세요.",
    ],
    "invalid_box_size": [
        "bbox width와 height는 0보다 커야 합니다.",
        "너비나 높이가 0인 박스는 수정하거나 제거하세요.",
    ],
    "no_objects": [
        "Pascal VOC XML에 object 항목이 있는지 확인하세요.",
        "객체가 없는 이미지라면 변환 대상에서 제외하세요.",
    ],
    "invalid_object": [
        "Pascal VOC object에는 name과 bndbox가 필요합니다.",
        "bndbox의 xmin, ymin, xmax, ymax 값을 확인하세요.",
    ],
    "non_numeric_box": [
        "Pascal VOC bbox 좌표는 숫자여야 합니다.",
        "문자열 또는 빈 좌표 값을 숫자로 수정하세요.",
    ],
    "no_images": [
        "COCO JSON의 images 배열이 비어 있지 않은지 확인하세요.",
        "annotation의 image_id와 images 항목이 연결되는지 확인하세요.",
    ],
    "no_annotations": [
        "COCO JSON의 annotations 배열이 비어 있지 않은지 확인하세요.",
        "객체가 없는 샘플은 제외하거나 task에 맞는 포맷을 선택하세요.",
    ],
    "no_categories": [
        "COCO JSON의 categories 배열에 클래스 목록을 추가하세요.",
        "annotation의 category_id가 categories의 id와 연결되는지 확인하세요.",
    ],
    "invalid_bbox": [
        "COCO bbox는 [x, y, width, height] 구조여야 합니다.",
        "bbox 배열 길이와 순서를 확인하세요.",
    ],
    "non_numeric_bbox": [
        "COCO bbox 값은 모두 숫자여야 합니다.",
        "빈 문자열이나 클래스명이 bbox에 섞이지 않았는지 확인하세요.",
    ],
    "invalid_bbox_size": [
        "COCO bbox의 width와 height는 0보다 커야 합니다.",
        "크기가 0 이하인 annotation은 수정하거나 제거하세요.",
    ],
    "missing_category_id": [
        "COCO annotation에는 category_id가 필요합니다.",
        "category_id가 categories 배열의 id와 일치하는지 확인하세요.",
    ],
    "no_label_records": [
        "Vision JSON에 변환 가능한 라벨 record가 있는지 확인하세요.",
        "선택한 task와 입력 JSON 구조가 맞는지 확인하세요.",
    ],
    "no_labels": [
        "생성 결과 또는 입력 데이터에 클래스명이 있는지 확인하세요.",
        "YOLO 입력이라면 클래스 매핑 파일을 함께 제공하세요.",
    ],
}

FIX_INSTRUCTIONS = {
    "empty_result": "라벨 파일{target}에 객체 라벨 행 또는 annotation을 추가하거나 해당 샘플을 제외하고 다시 변환하세요.",
    "missing_image": "이미지 파일{target}을 복원하거나 라벨 파일의 이미지 이름이 실제 파일명과 일치하도록 수정하세요.",
    "image_open_failed": "이미지 파일{target}이 손상되지 않았는지 확인하고 지원 형식(PNG, JPG, JPEG, WEBP, BMP)으로 다시 저장하세요.",
    "invalid_image_size": "이미지 파일{target}을 정상적인 크기의 이미지로 교체하세요.",
    "missing_label": "라벨 파일{target}에서 클래스명이 비어 있는 객체에 실제 클래스명을 지정하세요.",
    "missing_name": "Pose 라벨{target}의 keypoint name 필드를 채우세요.",
    "missing_text": "OCR 라벨{target}의 text 필드를 채우거나 해당 OCR 객체를 제거하세요.",
    "missing_track_id": "Tracking 라벨{target}에 객체별 track_id를 부여하세요.",
    "coordinate_out_of_range": "라벨 파일{target}의 좌표를 0~1 정규화 좌표로 수정하세요. 픽셀 좌표라면 이미지 width/height로 나누세요.",
    "invalid_box_order": "라벨 파일{target}의 박스가 xmin < xmax, ymin < ymax 순서를 만족하도록 좌표를 수정하세요.",
    "confidence_out_of_range": "라벨 파일{target}의 confidence 값을 0~1 범위로 정규화하세요.",
    "too_few_points": "Segmentation 라벨{target}을 세 점 이상의 polygon으로 다시 작성하세요.",
    "empty_keypoints": "Pose 라벨{target}에 keypoint를 추가하거나 잘못된 pose 객체를 제거하세요.",
    "missing_output_path": "출력 경로 설정을 확인하고 쓰기 가능한 디렉터리로 다시 실행하세요.",
    "missing_output_file": "출력 파일{target}이 생성되지 않았으므로 출력 디렉터리 권한과 포맷 설정을 확인하세요.",
    "empty_output_file": "출력 파일{target}이 비어 있으므로 입력 라벨에 변환 가능한 객체가 있는지 확인하세요.",
    "output_parse_failed": "출력 파일{target}의 JSON/XML/TXT 구조와 인코딩을 확인한 뒤 다시 생성하세요.",
    "no_label_rows": "YOLO 출력 파일{target}에 class_id와 bbox 좌표 4개로 구성된 라벨 행이 생성되도록 입력 bbox를 확인하세요.",
    "invalid_row_shape": "YOLO 출력 파일{target}의 각 행을 class_id x_center y_center width height 구조로 수정하세요.",
    "non_numeric_row": "YOLO 출력 파일{target}의 class_id와 좌표 값을 숫자로 수정하세요.",
    "negative_class_id": "YOLO 출력 파일{target}의 class_id를 0 이상의 정수로 수정하세요.",
    "invalid_box_size": "출력 라벨{target}의 bbox width와 height가 0보다 크도록 좌표를 수정하세요.",
    "no_objects": "Pascal VOC 출력 파일{target}에 object가 생성되도록 입력 bbox와 task 설정을 확인하세요.",
    "invalid_object": "Pascal VOC 출력 파일{target}의 object에 name과 bndbox를 채우세요.",
    "non_numeric_box": "Pascal VOC 출력 파일{target}의 bndbox 좌표를 숫자로 수정하세요.",
    "no_images": "COCO 출력 파일{target}의 images 배열과 image_id 연결을 확인하세요.",
    "no_annotations": "COCO 출력 파일{target}의 annotations 배열이 생성되도록 입력 bbox와 task 설정을 확인하세요.",
    "no_categories": "COCO 출력 파일{target}의 categories 배열에 클래스 목록을 추가하세요.",
    "invalid_bbox": "COCO 출력 파일{target}의 bbox를 [x, y, width, height] 배열로 수정하세요.",
    "non_numeric_bbox": "COCO 출력 파일{target}의 bbox 값을 숫자로 수정하세요.",
    "invalid_bbox_size": "COCO 출력 파일{target}의 bbox width와 height가 0보다 크도록 수정하세요.",
    "missing_category_id": "COCO 출력 파일{target}의 annotation에 category_id를 추가하고 categories.id와 맞추세요.",
    "no_label_records": "Vision JSON 출력 파일{target}에 task와 호환되는 라벨 record가 생성되도록 입력 데이터를 확인하세요.",
    "no_labels": "클래스 목록{target}이 비어 있으므로 입력 라벨의 클래스명 또는 YOLO 클래스 매핑 파일을 확인하세요.",
}


def issue_code(issue: str) -> str:
    for part in str(issue).split(":"):
        normalized = re.sub(r"\[[^]]*\]", "", part).split(".")[-1].strip()
        if normalized in ISSUE_CATALOG:
            return normalized
    return "unknown"


def _affected_path(issue: str) -> str:
    text = str(issue)
    parts = text.split(":")
    code = issue_code(text)
    if code == "unknown" or len(parts) < 2:
        return ""
    for index, part in enumerate(parts):
        normalized = re.sub(r"\[[^]]*\]", "", part).split(".")[-1].strip()
        if normalized == code and index + 1 < len(parts):
            return ":".join(parts[index + 1:]).strip()
    if parts[0] == code:
        return ":".join(parts[1:]).strip()
    return ""


def _format_fix_instruction(code: str, affected_path: str, fallback: str) -> str:
    template = FIX_INSTRUCTIONS.get(code)
    if not template:
        return fallback
    target = f"({affected_path})" if affected_path else ""
    return template.format(target=target)


def categorize_issue(issue: str) -> Dict[str, Any]:
    code = issue_code(issue)
    severity, category, message, action = ISSUE_CATALOG.get(
        code,
        ("unknown", "unknown", str(issue), "이슈와 workflow 로그를 수동으로 검토하세요."),
    )
    affected_path = _affected_path(str(issue))
    return {
        "code": code,
        "severity": severity,
        "category": category,
        "message": message,
        "user_action": action,
        "suggestions": ISSUE_SUGGESTIONS.get(code, [action]),
        "affected_path": affected_path,
        "fix_instruction": _format_fix_instruction(code, affected_path, action),
        "original_issue": str(issue),
    }


def _record_report(image: str, issues: Iterable[str]) -> Dict[str, Any]:
    details = [categorize_issue(issue) for issue in dict.fromkeys(issues)]
    severity_counts = Counter(item["severity"] for item in details)
    category_counts = Counter(item["category"] for item in details)
    if severity_counts["critical"]:
        status = "blocked"
    elif severity_counts["high"]:
        status = "needs_attention"
    else:
        status = "warning"
    priority = [
        item.get("fix_instruction") or item["user_action"]
        for item in details
        if item["severity"] in {"critical", "high"}
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
    for record in [*validation_records, *(export_records or [])]:
        issues_by_image[str(record.get("image", "unknown"))].extend(record.get("issues", []))
    problem_images = {
        image: issues for image, issues in issues_by_image.items() if issues
    }
    records = [_record_report(image, issues) for image, issues in sorted(problem_images.items())]
    artifact_details = [categorize_issue(issue) for issue in (artifact_issues or [])]
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
