from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


def _conversion_impact(severity: str, code: str) -> str:
    if code in {"no_label_sources", "no_records"}:
        return "변환 불가"
    if code in {"missing_images", "invalid_images", "import_failed_files"}:
        return "해당 데이터 제외 후 진행"
    if severity == "critical":
        return "해당 데이터 제외 후 진행"
    return "변환은 그대로 진행"


def _notice(severity: str, code: str, message: str, action: str, **extra) -> Dict[str, Any]:
    payload = {
        "severity": severity,
        "code": code,
        "message": message,
        "user_action": action,
        "conversion_impact": extra.pop("conversion_impact", _conversion_impact(severity, code)),
    }
    payload.update(extra)
    return payload


def _issue_code(issue: str) -> str:
    return str(issue).split(":", 1)[0].split(".")[-1]


def _short_path(value: Any) -> str:
    return Path(str(value)).name if value else ""


def _sample_paths(items: Iterable[Dict[str, Any]], key: str = "path", limit: int = 5) -> List[str]:
    samples = []
    for item in items:
        value = item.get(key)
        if value:
            samples.append(_short_path(value))
        if len(samples) >= limit:
            break
    return samples


def _conflict_examples(conflicts: Iterable[Dict[str, Any]], limit: int = 5) -> List[str]:
    examples = []
    for conflict in conflicts:
        image = conflict.get("image", "unknown")
        existing = conflict.get("existing_label", "?")
        incoming = conflict.get("incoming_label", "?")
        iou = conflict.get("iou")
        suffix = f" (IoU {iou:.3f})" if isinstance(iou, (int, float)) else ""
        examples.append(f"{image}: {existing} vs {incoming}{suffix}")
        if len(examples) >= limit:
            break
    return examples


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
            "변환할 수 있는 라벨 레코드가 없습니다.",
            "입력 라벨 파일이 비어 있거나 지원 schema인지 확인하세요.",
        ))

    failed_files = summary.get("failed_files", [])
    if failed_files:
        files = _sample_paths(failed_files)
        notices.append(_notice(
            "warning",
            "import_failed_files",
            f"{len(failed_files)}개 라벨 파일을 읽지 못했습니다.",
            f"확인할 파일: {', '.join(files)}" if files else "conversion_report.json의 input_summary.failed_files를 확인하세요.",
            count=len(failed_files),
            files=files,
            details=failed_files[:5],
        ))

    skipped_files = summary.get("skipped_files", [])
    if skipped_files:
        files = _sample_paths(skipped_files)
        notices.append(_notice(
            "info",
            "skipped_unrecognized_files",
            f"{len(skipped_files)}개 파일은 지원 라벨 schema로 판별되지 않아 건너뛰었습니다.",
            f"건너뛴 파일: {', '.join(files)}. 의도한 라벨 파일이면 포맷을 명시하거나 schema를 지원 형식에 맞게 정리하세요.",
            count=len(skipped_files),
            files=files,
            details=skipped_files[:5],
        ))

    for source in summary.get("processed_files", []):
        if source.get("format") != "yolo":
            continue
        duplicate_rows = source.get("duplicate_label_rows") or []
        if duplicate_rows:
            files = _sample_paths(duplicate_rows)
            examples = []
            for item in duplicate_rows[:5]:
                row = (item.get("rows") or [{}])[0]
                line = row.get("line")
                first_line = row.get("first_line")
                location = f"{_short_path(item.get('path'))}: line {line}"
                if first_line:
                    location += f" duplicates line {first_line}"
                examples.append(location)
            notices.append(_notice(
                "warning",
                "duplicate_label_rows",
                f"동일한 YOLO 라벨 row 중복이 {sum(item.get('count', 0) for item in duplicate_rows)}건 발견됐습니다.",
                f"중복 라벨을 확인하세요: {', '.join(examples)}. 의도한 중복이 아니면 하나만 남기세요.",
                count=sum(item.get("count", 0) for item in duplicate_rows),
                files=files,
                examples=examples,
                details=duplicate_rows[:5],
            ))
        mapping = source.get("class_mapping", {})
        if mapping.get("status") == "missing":
            source_name = _short_path(source.get("path"))
            notices.append(_notice(
                "warning",
                "missing_yolo_class_mapping",
                f"YOLO class mapping 파일을 찾지 못해 {source_name}의 class id를 문자 라벨로 처리했습니다.",
                f"{source_name}와 같은 폴더에 data.yaml, dataset.yaml 또는 classes.txt를 두거나 classes_path로 지정하세요.",
                source=source.get("path"),
                searched=mapping.get("searched", []),
                files=[source_name] if source_name else [],
            ))

    merge = summary.get("merge", {})
    conflicts = merge.get("conflicts", [])
    if conflicts:
        examples = _conflict_examples(conflicts)
        notices.append(_notice(
            "warning",
            "label_conflicts",
            f"동일 위치 라벨에서 클래스명이 다른 충돌 {len(conflicts)}건을 발견했습니다.",
            f"충돌 데이터: {', '.join(examples)}. taxonomy나 class alias를 정리하세요.",
            count=len(conflicts),
            examples=examples,
            details=conflicts[:5],
        ))

    if validation_records:
        issue_counts = Counter(
            _issue_code(issue)
            for record in validation_records
            for issue in record.get("issues", [])
        )
        if issue_counts.get("missing_image"):
            examples = [
                record.get("image", "")
                for record in validation_records
                if any(_issue_code(issue) == "missing_image" for issue in record.get("issues", []))
            ][:5]
            notices.append(_notice(
                "critical",
                "missing_images",
                f"이미지 파일을 찾지 못한 레코드가 {issue_counts['missing_image']}개 있습니다.",
                f"이미지 파일을 확인하세요: {', '.join(examples)}. 라벨 filename/stem과 이미지 디렉터리 구성을 맞춘 뒤 다시 실행하세요.",
                count=issue_counts["missing_image"],
                examples=examples,
            ))
        if issue_counts.get("image_open_failed") or issue_counts.get("invalid_image_size"):
            count = issue_counts.get("image_open_failed", 0) + issue_counts.get("invalid_image_size", 0)
            notices.append(_notice(
                "critical",
                "invalid_images",
                f"이미지 크기를 읽을 수 없는 레코드가 {count}개 있습니다.",
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
