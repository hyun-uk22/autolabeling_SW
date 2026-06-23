from collections import Counter
from typing import Any, Dict, Iterable, List, Optional

from .issue_reporter import ISSUE_CATALOG, issue_code as normalize_issue_code


def _notice(severity: str, code: str, message: str, action: str, **extra) -> Dict[str, Any]:
    payload = {
        "severity": severity,
        "code": code,
        "message": message,
        "user_action": action,
    }
    payload.update(extra)
    return payload


def _issue_code(issue: str) -> str:
    return normalize_issue_code(str(issue))


def _validation_notice(code: str, count: int) -> Optional[Dict[str, Any]]:
    if not count:
        return None
    severity, category, message, action = ISSUE_CATALOG.get(
        code,
        ("warning", "input_data", code, "입력 데이터 문제 상세를 확인하세요."),
    )
    # Preflight should summarize data quality problems without newly blocking
    # conversion unless the issue is already handled as critical above.
    notice_severity = "warning" if severity in {"critical", "high", "medium"} else "info"
    return _notice(
        notice_severity,
        code,
        f"{message} 항목이 {count}개 있습니다.",
        action,
        count=count,
        category=category,
    )


def build_conversion_preflight(
    input_summary: Optional[Dict[str, Any]],
    target_formats: Iterable[str],
    validation_records: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Describe missing or risky information before/while converting labels."""
    summary = input_summary or {}
    targets = list(target_formats or [])
    notices = []

    if summary.get("sources_discovered", 0) == 0:
        notices.append(_notice(
            "critical",
            "no_label_sources",
            "변환 가능한 라벨 소스를 찾지 못했습니다.",
            "입력 경로와 source format을 확인하세요.",
        ))
    if summary.get("records_after_merge", 0) == 0:
        notices.append(_notice(
            "critical",
            "no_records",
            "변환할 수 있는 라벨 데이터가 없습니다.",
            "입력 라벨 파일이 비어 있거나 지원 파일 형식인지 확인하세요.",
        ))

    failed_files = summary.get("failed_files", [])
    if failed_files:
        notices.append(_notice(
            "warning",
            "import_failed_files",
            f"{len(failed_files)}개 라벨 파일을 읽지 못했습니다.",
            "검증 이슈 상세의 실패 원인을 확인해 라벨 형식이 실제 파일 내용과 맞는지 점검하고, 리포트/metrics 파일은 라벨 입력 폴더에서 분리하세요.",
            count=len(failed_files),
        ))

    skipped_files = summary.get("skipped_files", [])
    if skipped_files:
        notices.append(_notice(
            "info",
            "skipped_unrecognized_files",
            f"{len(skipped_files)}개 파일은 지원 라벨 파일 형식으로 판별되지 않아 건너뛰었습니다.",
            "의도한 라벨 파일이면 입력 파일 형식을 명시하거나 파일 구조를 지원 형식에 맞게 정리하세요.",
            count=len(skipped_files),
        ))

    for source in summary.get("processed_files", []):
        if source.get("format") != "yolo":
            continue
        mapping = source.get("class_mapping", {})
        if mapping.get("status") == "missing":
            notices.append(_notice(
                "warning",
                "missing_yolo_class_mapping",
                "YOLO class mapping 파일을 찾지 못해 class id를 문자열 라벨로 처리했습니다.",
                "data.yaml, dataset.yaml 또는 classes.txt를 입력 라벨 폴더에 두거나 --classes/classes_path로 지정하세요.",
                source=source.get("path"),
                searched=mapping.get("searched", []),
            ))

    merge = summary.get("merge", {})
    conflicts = merge.get("conflicts", [])
    if conflicts:
        notices.append(_notice(
            "warning",
            "label_conflicts",
            f"동일 위치 라벨에서 클래스명이 다른 충돌 {len(conflicts)}건을 발견했습니다.",
            "input_summary.merge.conflicts를 확인해 taxonomy나 class alias를 정리하세요.",
            count=len(conflicts),
        ))
    label_normalizations = merge.get("label_normalizations", [])
    if label_normalizations:
        notices.append(_notice(
            "info",
            "numeric_label_normalized",
            f"겹치는 라벨을 기준으로 숫자 클래스 라벨 {len(label_normalizations)}건을 실제 클래스명으로 정규화했습니다.",
            "input_summary.merge.label_normalizations를 확인해 추론된 class id와 클래스명이 의도와 일치하는지 검토하세요.",
            count=len(label_normalizations),
        ))

    if validation_records:
        issue_counts = Counter(
            _issue_code(issue)
            for record in validation_records
            for issue in record.get("issues", [])
        )
        if issue_counts.get("missing_image"):
            notices.append(_notice(
                "warning",
                "missing_images",
                f"이미지 파일을 찾지 못한 데이터가 {issue_counts['missing_image']}개 있습니다.",
                "라벨 파일만 점검하는 경우 계속 진행할 수 있지만, COCO/Pascal VOC처럼 이미지 크기가 필요한 출력은 이미지 디렉터리 연결 후 다시 실행하세요.",
                count=issue_counts["missing_image"],
            ))
        if issue_counts.get("image_open_failed") or issue_counts.get("invalid_image_size"):
            count = issue_counts.get("image_open_failed", 0) + issue_counts.get("invalid_image_size", 0)
            notices.append(_notice(
                "critical",
                "invalid_images",
                f"이미지 크기를 읽을 수 없는 데이터가 {count}개 있습니다.",
                "손상 이미지 또는 지원하지 않는 이미지 형식을 교체하세요.",
                count=count,
            ))
        if issue_counts.get("missing_label"):
            notices.append(_notice(
                "warning",
                "missing_labels",
                f"클래스 라벨이 누락된 항목이 {issue_counts['missing_label']}개 있습니다.",
                "라벨 taxonomy를 확인하고 누락 클래스명을 채우세요.",
                count=issue_counts["missing_label"],
            ))
        summarized_codes = {
            "missing_image",
            "image_open_failed",
            "invalid_image_size",
            "missing_label",
        }
        for code in sorted(set(issue_counts) - summarized_codes):
            notice = _validation_notice(code, issue_counts[code])
            if notice:
                notices.append(notice)

    if "yolo" in targets and not summary.get("class_list"):
        notices.append(_notice(
            "warning",
            "empty_class_list",
            "YOLO 출력에 사용할 클래스 목록이 비어 있습니다.",
            "객체 라벨이 있는지 확인하거나 YOLO class mapping 파일을 제공하세요.",
        ))

    severity_order = {"critical": 0, "warning": 1, "info": 2}
    notices.sort(key=lambda item: (severity_order.get(item["severity"], 99), item["code"]))
    return {
        "status": "blocked" if any(item["severity"] == "critical" for item in notices)
        else "needs_attention" if any(item["severity"] == "warning" for item in notices)
        else "ok",
        "target_formats": targets,
        "notices": notices,
        "missing_required": [item for item in notices if item["severity"] == "critical"],
        "missing_recommended": [item for item in notices if item["severity"] == "warning"],
    }
