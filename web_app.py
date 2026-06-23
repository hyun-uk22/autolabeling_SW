import html
import json
import copy
import uuid
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components

from src.core.user_settings import (
    ENV_FIELDS,
    SECRET_FIELDS,
    load_user_environment,
    read_user_settings,
    save_user_settings,
)
from src.core.workspace import (
    WORKSPACE_DEFAULTS,
    load_workspace,
    resolve_workspace_path,
    save_workspace,
)
from src.reporting import build_conversion_preflight
from src.reporting.issue_reporter import categorize_issue
from src.utils.label_importer import find_image_path, import_labels_with_report
from src.utils.label_validator import summarize_validation, validate_result
from src.workflow.service import execute_workflow_plan
from src.workflow.conversation import (
    describe_plan,
    describe_result,
)
from src.workflow.conversation_router import handle_conversation
from src.workflow.plan_patcher import revise_pending_proposal


st.set_page_config(page_title="AutoLabel", page_icon=":material/dataset:", layout="wide")
load_user_environment()

st.markdown(
    """
    <style>
    :root {
        --ink: #1d232b;
        --muted: #66717f;
        --line: #dfe3e8;
        --surface: #ffffff;
        --canvas: #f4f6f8;
        --blue: #2563eb;
        --blue-soft: #eff5ff;
        --green: #15803d;
    }
    html, body, [class*="css"] {
        font-family: "Pretendard", "Malgun Gothic", "Segoe UI", sans-serif;
    }
    .stApp { background: var(--canvas); color: var(--ink); }
    [data-testid="stHeader"] { background: transparent; height: 0; }
    [data-testid="stToolbar"], #MainMenu, footer { display: none; }
    .block-container { max-width: 1220px; padding-top: 1.25rem; padding-bottom: 3rem; }
    h1, h2, h3, p { letter-spacing: 0 !important; }
    .app-header {
        display: flex;
        align-items: center;
        gap: 14px;
        min-height: 68px;
        padding: 0 2px 16px 2px;
        border-bottom: 1px solid var(--line);
        margin-bottom: 18px;
    }
    .brand-mark {
        display: grid;
        place-items: center;
        width: 40px;
        height: 40px;
        border-radius: 6px;
        background: #1f2937;
        color: white;
        font-size: 13px;
        font-weight: 800;
    }
    .brand-name { color: var(--ink); font-size: 19px; font-weight: 750; line-height: 1.15; }
    .brand-meta { color: var(--muted); font-size: 12px; margin-top: 4px; }
    .runtime-state {
        display: flex;
        align-items: center;
        gap: 7px;
        margin-left: auto;
        color: #3d4957;
        font-size: 12px;
        font-weight: 600;
    }
    .runtime-dot { width: 8px; height: 8px; border-radius: 50%; background: #22a447; }
    .workspace-path {
        color: #4b5563;
        font-size: 12px;
        font-weight: 600;
        margin: -8px 0 16px 0;
        overflow-wrap: anywhere;
    }
    .panel-title { margin: 6px 0 2px 0; color: var(--ink); font-size: 18px; font-weight: 750; }
    .panel-meta { margin: 0 0 14px 0; color: var(--muted); font-size: 12px; }
    .form-section {
        color: #4b5563;
        font-size: 11px;
        font-weight: 750;
        letter-spacing: .08em;
        margin: 2px 0 10px 0;
        text-transform: uppercase;
    }
    div[data-testid="stForm"] {
        background: var(--surface);
        border: 1px solid var(--line);
        border-radius: 7px;
        padding: 22px 24px 20px 24px;
        box-shadow: 0 1px 2px rgba(20, 30, 45, .04);
    }
    div[data-baseweb="input"] > div,
    div[data-baseweb="select"] > div,
    div[data-baseweb="textarea"] > div {
        background: #fbfcfd;
        border-color: #cfd5dc;
        border-radius: 5px;
    }
    div[data-baseweb="input"] > div:focus-within,
    div[data-baseweb="select"] > div:focus-within,
    div[data-baseweb="textarea"] > div:focus-within {
        border-color: var(--blue);
        box-shadow: 0 0 0 1px var(--blue);
    }
    label[data-testid="stWidgetLabel"] p { color: #394351; font-size: 12px; font-weight: 650; }
    div[data-testid="stFormSubmitButton"] { display: flex; justify-content: flex-end; padding-top: 8px; }
    div[data-testid="stVerticalBlockBorderWrapper"]:has(.path-panel-marker),
    div[data-testid="stVerticalBlockBorderWrapper"]:has(.path-panel-marker) > div,
    div[data-testid="stVerticalBlockBorderWrapper"]:has(.path-panel-marker) [data-testid="stVerticalBlock"] {
        background: var(--surface);
        border-radius: 7px;
    }
    .path-panel-marker {
        display: none;
    }
    div[data-testid="stFormSubmitButton"] > div { display: flex; justify-content: flex-end; width: 100%; }
    div[data-testid="stFormSubmitButton"] button { margin-left: auto; }
    .stButton > button, .stFormSubmitButton > button {
        min-height: 38px;
        border-radius: 5px;
        font-weight: 700;
        padding: 0 18px;
    }
    .stFormSubmitButton > button[kind="primary"] {
        background: var(--blue);
        border-color: var(--blue);
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 4px;
        border-bottom: 1px solid var(--line);
        margin-bottom: 18px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 42px;
        padding: 0 16px;
        color: #5c6673;
        font-size: 13px;
        font-weight: 650;
        background: transparent;
    }
    .stTabs [aria-selected="true"] { color: var(--blue) !important; }
    .stTabs [data-baseweb="tab-highlight"] { background: var(--blue); height: 2px; }
    div[data-testid="stAlert"] { border-radius: 6px; border-width: 1px; }
    div[data-testid="stCode"] { border-radius: 5px; }
    @media (max-width: 760px) {
        .block-container { padding: .8rem .8rem 2rem .8rem; }
        .brand-meta, .runtime-state { display: none; }
        div[data-testid="stForm"] { padding: 16px 14px; }
        .stTabs [data-baseweb="tab"] { padding: 0 9px; font-size: 12px; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


FORMAT_OPTIONS = ["yolo", "pascal_voc", "coco", "vision_json"]
SOURCE_OPTIONS = ["auto", "yolo", "pascal_voc", "coco", "vision_json", "csv", "generic_json"]
TASK_OPTIONS = ["object_detection", "classification", "segmentation", "pose_estimation", "ocr", "tracking", "all"]
STATUS_LABELS = {
    "blocked": "확인 필요",
    "needs_attention": "주의 필요",
    "warning": "주의",
    "partial_success": "일부 완료",
    "needs_review": "검토 필요",
    "success": "정상",
    "ok": "정상",
    "completed": "완료",
    "completed_with_errors": "오류 있음",
    "failed": "실패",
}
ISSUE_REASON_LABELS = {
    "empty_result": "라벨 파일을 읽었지만 변환할 객체 라벨이 없습니다.",
    "missing_image": "라벨과 연결되는 원본 이미지 파일을 찾지 못했습니다.",
    "image_open_failed": "이미지 파일을 열 수 없습니다. 파일이 손상됐거나 지원하지 않는 형식일 수 있습니다.",
    "invalid_image_size": "이미지 크기를 읽을 수 없거나 크기 값이 올바르지 않습니다.",
    "missing_label": "객체의 클래스명이 비어 있습니다.",
    "coordinate_out_of_range": "좌표값이 유효 범위를 벗어났습니다.",
    "invalid_box_order": "박스 좌표 순서가 올바르지 않습니다.",
    "confidence_out_of_range": "confidence 값이 0~1 범위를 벗어났습니다.",
    "import_failed_files": "라벨 후보 파일을 읽는 중 오류가 발생했습니다.",
    "skipped_unrecognized_files": "지원하는 라벨 파일 형식으로 판별되지 않아 건너뛰었습니다.",
    "unrecognized_csv_schema": "CSV 파일 구조가 지원하는 라벨 파일 형식과 다릅니다.",
    "unrecognized_json_schema": "JSON 파일 구조가 지원하는 라벨 파일 형식과 다릅니다.",
    "unrecognized_jsonl_schema": "JSONL 파일 구조가 지원하는 라벨 파일 형식과 다릅니다.",
    "unrecognized_txt_schema": "TXT 파일 구조가 YOLO 라벨 형식과 다릅니다.",
    "unrecognized_xml_schema": "XML 파일 구조가 Pascal VOC 라벨 형식과 다릅니다.",
    "schema_read_failed": "파일 내용을 읽거나 해석하는 중 오류가 발생했습니다.",
}


def display_status(value):
    return STATUS_LABELS.get(str(value), str(value or "-"))


def display_issue_reason(code, fallback=""):
    text = str(code or "")
    key = text.split(":", 1)[0]
    return ISSUE_REASON_LABELS.get(key, fallback or text or "-")


def _relative_workspace_path(workspace, path):
    root = Path(workspace).expanduser().resolve()
    target = Path(path).expanduser().resolve()
    try:
        return target.relative_to(root).as_posix()
    except ValueError:
        return str(target)


def workspace_dir_candidates(workspace, names, fallback):
    root = Path(workspace).expanduser().resolve()
    wanted = {name.lower() for name in names}
    ignored = {
        ".git",
        ".streamlit",
        "__pycache__",
        "artifacts",
        "converted",
        "outputs",
        "reports",
        "runs",
        "visualized",
        "visualizations",
    }
    candidates = []
    if root.exists():
        for path in root.rglob("*"):
            if not path.is_dir():
                continue
            parts = {part.lower() for part in path.parts}
            if parts & ignored:
                continue
            if path.name.lower() in wanted:
                candidates.append(_relative_workspace_path(root, path))
    candidates = sorted(dict.fromkeys(candidates), key=lambda value: (len(value), value.lower()))
    if candidates:
        if fallback not in candidates:
            candidates.append(fallback)
        return candidates
    return [fallback]


def _plan_actions(plan):
    return [
        operation.get("action")
        for operation in plan.get("operations", [])
        if operation.get("action")
    ]


def execute_plan(plan, auto_approve=False):
    actions = _plan_actions(plan)
    has_generation = "generate" in actions
    label = "라벨 생성 실행 중" if has_generation else "작업 실행 중"
    with st.status(label, expanded=True) as status:
        st.write("Workflow를 준비하고 있습니다.")
        if has_generation:
            st.write(
                "Specialist vision model을 준비합니다. 최초 실행이거나 캐시가 비어 있으면 "
                "Grounding DINO/SAM 계열 가중치 다운로드 때문에 시간이 오래 걸릴 수 있습니다."
            )
        elif "convert" in actions:
            st.write("라벨 파일을 읽고 변환/검증을 진행합니다.")
        elif "evaluate" in actions:
            st.write("평가 대상과 리포트 출력을 준비합니다.")
        try:
            result = execute_workflow_plan(
                plan,
                auto_approve=auto_approve,
                thread_id=f"streamlit-{uuid.uuid4()}",
            )
        except Exception:
            status.update(label="작업 실패", state="error", expanded=True)
            raise
        status.update(label="작업 완료", state="complete", expanded=False)
        return result


def _download_report_files(output, report_index, key_prefix):
    paths = {}
    for key in ("report_path", "summary_path", "user_action_report_path"):
        if output.get(key):
            paths[key] = output[key]
    paths.update(output.get("artifacts", {}))
    available = [
        (name, Path(path))
        for name, path in paths.items()
        if path and Path(path).is_file()
    ]
    if not available:
        return
    st.markdown("**결과 파일**")
    columns = st.columns(min(3, len(available)))
    for file_index, (name, path) in enumerate(available):
        with columns[file_index % len(columns)]:
            st.download_button(
                f"{name} 다운로드",
                data=path.read_bytes(),
                file_name=path.name,
                key=f"{key_prefix}-download-{report_index}-{file_index}-{name}",
                width="stretch",
            )


def render_workflow_report(result, key_prefix="report"):
    outputs = result.get("operation_outputs", [])
    for index, output in enumerate(outputs):
        action = output.get("action", "unknown")
        action_label = {"generate": "라벨 생성", "convert": "형식 변환", "evaluate": "평가"}.get(
            action,
            action,
        )
        st.markdown(f"### {action_label} 결과 리포트")
        st.caption(
            "입력 데이터 문제: 입력/변환 전후 검증 단계에서 발견된 문제 수 · "
            "결과 파일 문제: 저장된 결과물에서 발견된 문제 수 · "
            "검토 필요: 문제가 있는 실제 데이터 파일 수"
        )
        if action == "generate":
            performance = output.get("performance", {})
            columns = st.columns(4)
            columns[0].metric("처리 이미지", output.get("images", 0))
            columns[1].metric("생성 라벨", output.get("total_labels", 0))
            columns[2].metric("평균 처리 시간", f"{performance.get('avg_elapsed_sec', 0.0):.2f}초")
            columns[3].metric("Escalation 비율", f"{performance.get('escalation_rate', 0.0) * 100:.1f}%")
            if performance.get("estimation_notice"):
                st.caption(performance["estimation_notice"])
            first_pass = output.get("first_pass_report", {})
            if first_pass.get("images"):
                fp_columns = st.columns(3)
                fp_columns[0].metric("1차 추론 이미지", first_pass.get("images", 0))
                fp_columns[1].metric("1차 라벨", first_pass.get("total_labels", 0))
                mean_confidence = first_pass.get("mean_confidence")
                fp_columns[2].metric(
                    "1차 평균 Confidence",
                    "-" if mean_confidence is None else f"{mean_confidence:.2f}",
                )
            specialist_consistency = output.get("specialist_consistency", {})
            if specialist_consistency.get("enabled_images"):
                sc_columns = st.columns(3)
                sc_columns[0].metric("재추론 이미지", specialist_consistency.get("enabled_images", 0))
                sc_columns[1].metric("Advisor", specialist_consistency.get("advisor_mode", "none"))
                mean_agreement = specialist_consistency.get("mean_bbox_agreement")
                sc_columns[2].metric(
                    "BBox Agreement",
                    "-" if mean_agreement is None else f"{mean_agreement * 100:.1f}%",
                )
        elif action == "convert":
            validation = output.get("validation", {})
            columns = st.columns(4)
            columns[0].metric("읽은 데이터", output.get("records_read", 0))
            columns[1].metric("변환 완료", output.get("records_converted", 0))
            columns[2].metric("입력 데이터 문제", validation.get("failed_records", 0))
            columns[3].metric(
                "결과 파일 문제",
                output.get("export_validation", {}).get("failed_records", 0),
            )
        elif action == "evaluate":
            rows = output.get("rows", [])
            st.metric("평가 실행", len(rows))
            if rows:
                st.dataframe(rows, width="stretch")

        user_report = output.get("user_action_report", {})
        if user_report:
            summary = user_report.get("summary", {})
            columns = st.columns(4)
            columns[0].metric("완료율", f"{user_report.get('completion_rate', 0.0):.1f}%")
            columns[1].metric("정상", summary.get("clean", 0))
            columns[2].metric("검토 필요", summary.get("needs_review", 0))
            columns[3].metric("결과 파일 문제", summary.get("artifact_issues", 0))
            for action_text in user_report.get("recommended_actions", []):
                st.info(action_text)
            detailed = user_report.get("detailed_records", [])
            if detailed:
                st.markdown("**검토가 필요한 데이터**")
                st.dataframe([
                    {
                        "파일": record.get("image", ""),
                        "상태": record.get("status", ""),
                        "이슈 수": record.get("total_issues", 0),
                        "우선 조치": " / ".join(record.get("priority_actions", [])),
                    }
                    for record in detailed
                ], width="stretch")

        insight = output.get("dataset_insight", {})
        distribution = insight.get("distribution", {})
        if distribution:
            st.markdown("**Dataset Insight**")
            st.dataframe([
                {
                    "클래스": label,
                    "개수": values.get("count", 0),
                    "비율(%)": round(values.get("percentage", 0.0), 2),
                }
                for label, values in distribution.items()
            ], width="stretch")
            for suggestion in insight.get("suggestions", []):
                st.warning(suggestion)

        _download_report_files(output, index, key_prefix)


def render_manual_result(result, result_key):
    if result.get("status") == "completed":
        st.success("작업이 완료되었습니다.")
    else:
        st.warning(f"작업 상태: {result.get('status', 'unknown')}")
    render_workflow_report(result, key_prefix=f"manual-{result_key}")
    with st.expander(f"{result_key} 원본 workflow 결과"):
        st.json({
            "status": result.get("status"),
            "outputs": result.get("operation_outputs", []),
            "errors": result.get("errors", []),
            "history_path": result.get("history_path", ""),
        })


def render_recent_result_summary(label, result):
    outputs = result.get("operation_outputs", [])
    output = outputs[0] if outputs else {}
    report_path = output.get("report_path") or output.get("summary_path") or output.get("user_action_report_path")
    output_dir = str(Path(report_path).parent) if report_path else "-"
    action = output.get("action", "")
    formats = ", ".join(output.get("target_formats") or output.get("formats") or [])
    summary = f"최근 작업: {label}"
    if action == "convert" and formats:
        summary += f" | 출력 포맷: {formats}"
    if output_dir != "-":
        summary += f" | 출력 경로: {output_dir}"
    st.caption(summary)


def run_plan(plan, auto_approve=False, result_key="workflow", first_pass_plan_key=None):
    try:
        result = execute_plan(plan, auto_approve=auto_approve)
        if "manual_results" not in st.session_state:
            st.session_state.manual_results = {}
        st.session_state.manual_results[result_key] = result
        st.session_state.last_manual_result_key = result_key
        if first_pass_plan_key:
            st.session_state[first_pass_plan_key] = (
                plan if has_generate_operation(plan) else None
            )
        st.session_state.open_result_report = True
        st.rerun()
    except Exception as exc:
        st.error(f"작업 실패: {exc}")


def parse_runs(value):
    runs = {}
    for raw_line in value.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if "=" not in line:
            raise ValueError("실험 경로는 이름=경로 형식이어야 합니다.")
        name, path = line.split("=", 1)
        if not name.strip() or not path.strip():
            raise ValueError("실험 이름과 경로를 모두 입력하세요.")
        runs[name.strip()] = path.strip()
    if not runs:
        raise ValueError("실험 경로를 하나 이상 입력하세요.")
    return runs


def build_convert_preflight_preview(
    input_path,
    image_dir,
    source_format,
    target_formats,
    classes_path=None,
    duplicate_iou=0.85,
):
    batch = import_labels_with_report(
        input_path,
        image_dir,
        source_format=source_format,
        classes_path=classes_path,
        duplicate_iou=duplicate_iou,
    )
    validations = []
    for image_name, result in batch.records:
        image_path = find_image_path(image_dir, image_name)
        conversion_sources = result.plugin_metadata.get("conversion_sources", {})
        validations.append({
            "image": image_name,
            "image_path": image_path,
            "label_paths": conversion_sources.get("paths", []),
            "issues": validate_result(result, image_path),
        })
    return {
        "input_summary": batch.report,
        "validation": summarize_validation(validations),
        "preflight": build_conversion_preflight(batch.report, target_formats, validations),
        "records": validations,
    }


def preflight_user_summary(preview):
    preflight = preview.get("preflight", {})
    validation = preview.get("validation", {})
    summary = preview.get("input_summary", {})
    status = preflight.get("status", "unknown")
    total = summary.get("records_after_merge", 0)
    failed = validation.get("failed_records", 0)
    convertible = max(total - failed, 0)
    notices = preflight.get("notices", [])
    critical_count = sum(1 for notice in notices if notice.get("severity") == "critical")
    warning_count = sum(1 for notice in notices if notice.get("severity") == "warning")

    if status == "blocked" or total == 0 or critical_count:
        return (
            "error",
            "실행 전에 수정이 필요합니다.",
            "변환 가능한 라벨을 찾지 못했거나 필수 정보가 부족합니다. 아래 확인사항을 먼저 해결한 뒤 다시 사전 점검을 실행하세요.",
        )
    if failed:
        return (
            "warning",
            "주의가 필요하지만 실행할 수 있습니다.",
            f"{total}개 데이터 중 {convertible}개는 변환 가능하고, {failed}개는 검토가 필요합니다.",
        )
    if warning_count:
        return (
            "warning",
            "주의사항이 있지만 실행할 수 있습니다.",
            f"{total}개 데이터를 변환할 수 있습니다. 일부 중복, 충돌, 건너뛴 파일이 있을 수 있으니 아래 확인사항만 검토하세요.",
        )
    return (
        "success",
        "바로 실행할 수 있습니다.",
        f"{total}개 데이터에서 변환을 막는 문제가 발견되지 않았습니다.",
    )


def render_preflight_user_summary(preview):
    alert, title, body = preflight_user_summary(preview)
    message = f"**{title}**\n\n{body}"
    if alert == "error":
        st.error(message)
    elif alert == "warning":
        st.warning(message)
    else:
        st.success(message)


def render_convert_preflight_preview(preview):
    preflight = preview.get("preflight", {})
    validation = preview.get("validation", {})
    summary = preview.get("input_summary", {})
    columns = st.columns(3)
    columns[0].metric("발견 소스", summary.get("sources_discovered", 0))
    columns[1].metric("변환 대상", summary.get("records_after_merge", 0))
    columns[2].metric("입력 데이터 문제", validation.get("failed_records", 0))
    render_preflight_user_summary(preview)
    if preflight.get("notices"):
        st.markdown("**실행 전 확인 사항**")
        st.dataframe([
            {
                "심각도": display_status(notice.get("severity", "")),
                "항목": notice.get("code", ""),
                "내용": notice.get("message", ""),
                "필요 조치": notice.get("user_action", ""),
            }
            for notice in preflight.get("notices", [])
        ], width="stretch")
    else:
        st.success("변환 전에 확인된 필수 누락 정보가 없습니다.")
    detailed = preflight_detail_rows(preview)
    if detailed:
        st.markdown("**입력 데이터 문제 상세**")
        st.dataframe(detailed, width="stretch")


def preflight_detail_rows(preview):
    summary = preview.get("input_summary", {})
    grouped = {}

    def add_detail(file_name, issue_code, label_path):
        key = label_path or file_name
        if key not in grouped:
            grouped[key] = {
                "파일": file_name,
                "항목": [],
                "라벨 파일 경로": label_path,
                "원인": [],
            }
        row = grouped[key]
        if file_name and file_name not in str(row["파일"]).split("\n"):
            row["파일"] = "\n".join([str(row["파일"]), file_name]) if row["파일"] else file_name
        if issue_code and issue_code not in row["항목"]:
            row["항목"].append(issue_code)

    for item in summary.get("failed_files", []):
        path = item.get("path", "")
        add_detail(Path(path).name, "import_failed_files", path)
        if item.get("error") and item["error"] not in grouped[path]["원인"]:
            grouped[path]["원인"].append(display_issue_reason("import_failed_files", item["error"]))
    for item in summary.get("skipped_files", []):
        path = item.get("path", "")
        add_detail(Path(path).name, "skipped_unrecognized_files", path)
        if item.get("reason") and item["reason"] not in grouped[path]["원인"]:
            grouped[path]["원인"].append(display_issue_reason(item["reason"], item["reason"]))
    for record in preview.get("records", []):
        label_paths = record.get("label_paths", [])
        for issue in record.get("issues", []):
            detail = categorize_issue(issue)
            if label_paths:
                for label_path in label_paths:
                    add_detail(record.get("image", ""), detail.get("code", ""), label_path)
                    reason = display_issue_reason(detail.get("code"), detail.get("message", ""))
                    if reason not in grouped[label_path]["원인"]:
                        grouped[label_path]["원인"].append(reason)
            else:
                add_detail(record.get("image", ""), detail.get("code", ""), "")
                key = record.get("image", "")
                reason = display_issue_reason(detail.get("code"), detail.get("message", ""))
                if reason not in grouped[key]["원인"]:
                    grouped[key]["원인"].append(reason)
    return [
        {
            "파일": row["파일"],
            "항목": "\n".join(row["항목"]),
            "원인": "\n".join(row["원인"]),
            "라벨 파일 경로": row["라벨 파일 경로"],
        }
        for row in grouped.values()
    ]


def build_chat_preflight_preview(proposal):
    operation = (proposal.get("plan", {}).get("operations") or [{}])[0]
    if operation.get("action") != "convert":
        return None
    return build_convert_preflight_preview(
        operation.get("input_path"),
        operation.get("img_dir"),
        operation.get("source_format", "auto"),
        operation.get("formats", []),
        classes_path=operation.get("classes_path"),
        duplicate_iou=operation.get("duplicate_iou", 0.85),
    )


def render_chat_preflight_for_proposal(proposal):
    try:
        return build_chat_preflight_preview(proposal)
    except Exception as exc:
        return {
            "preflight": {
                "status": "blocked",
                "notices": [{
                    "severity": "warning",
                    "code": "preflight_failed",
                    "message": f"실행 전 점검을 생성하지 못했습니다: {exc}",
                    "user_action": "입력 라벨 경로, 이미지 경로, source format을 확인하세요.",
                }],
            },
            "validation": {"failed_records": 0},
            "input_summary": {"sources_discovered": 0, "records_after_merge": 0},
            "records": [],
        }


def describe_plan_revision(revision, workspace):
    if revision.get("kind") == "cancel":
        return f"현재 실행 계획을 취소했습니다. {revision.get('reason', '')}".strip()
    if revision.get("kind") == "new_plan":
        return (
            "현재 실행 계획과 다른 새 작업 요청으로 판단했습니다. "
            "기존 계획을 취소한 뒤 새 요청을 다시 입력해 주세요."
        )
    if revision.get("kind") == "clarify":
        return f"계획 수정 요청을 명확히 해 주세요. {revision.get('reason', '')}".strip()
    lines = ["**실행 계획을 수정했습니다.**"]
    if revision.get("reason"):
        lines.append(f"- 수정 근거: {revision['reason']}")
    for change in revision.get("changes", []):
        lines.append(f"- 변경: {change}")
    lines.append("")
    lines.append(describe_plan(revision["proposal"], workspace))
    return "\n".join(lines)


def first_pass_chat_proposal(proposal):
    prepared = copy.deepcopy(proposal)
    for operation in prepared.get("plan", {}).get("operations", []):
        if operation.get("action") == "generate":
            operation["specialist_consistency_runs"] = 0
            operation["specialist_advisor_mode"] = "none"
    return prepared


def has_generate_operation(plan):
    return any(
        operation.get("action") == "generate"
        for operation in plan.get("operations", [])
    )


def chat_rerun_plan(first_pass_plan, advisor_mode):
    plan = copy.deepcopy(first_pass_plan)
    for operation in plan.get("operations", []):
        if operation.get("action") == "generate":
            operation["specialist_consistency_runs"] = 1
            operation["specialist_advisor_mode"] = advisor_mode
    return plan


def render_selective_rerun_controls(first_pass_plan, key_prefix, on_complete):
    if not first_pass_plan or not has_generate_operation(first_pass_plan):
        return
    st.divider()
    st.markdown("#### 1차 추론 이후 선택 재추론")
    st.caption("위 결과를 확인한 뒤 필요할 때만 동일 작업 설정으로 specialist 재추론을 실행합니다.")
    advisor_mode = st.selectbox(
        "재추론 Advisor",
        ["none", "low", "high", "both"],
        index=0,
        key=f"{key_prefix}_specialist_advisor_mode",
        help="none은 LLM 없이 고정 파라미터로 재추론하고, low/high/both는 LLM이 threshold, augmentation, prompt 보조 파라미터를 제안합니다.",
    )
    if st.button("Specialist 재추론 1회 실행", type="secondary", key=f"{key_prefix}_specialist_rerun"):
        try:
            result = execute_plan(chat_rerun_plan(first_pass_plan, advisor_mode), auto_approve=True)
            on_complete(result)
        except Exception as exc:
            st.error(f"선택 재추론 실행 중 오류가 발생했습니다: {exc}")
        st.rerun()


def complete_chat_rerun(result, workspace):
    response = "선택 재추론을 완료했습니다.\n\n" + describe_result(result, workspace)
    st.session_state.last_chat_result = result
    st.session_state.chat_messages.append({"role": "assistant", "content": response})


def render_result_report(workspace):
    st.markdown('<div class="panel-title">결과 리포트</div>', unsafe_allow_html=True)
    manual_results = st.session_state.get("manual_results", {})
    manual_key = st.session_state.get("last_manual_result_key")
    if manual_key and manual_key in manual_results:
        manual_label = {
            "convert": "라벨 형식 변환",
            "generate": "자동 라벨 생성",
            "evaluate": "실험 결과 평가",
        }.get(manual_key, manual_key)
        render_recent_result_summary(manual_label, manual_results[manual_key])
        render_manual_result(manual_results[manual_key], manual_key)
        if manual_key == "generate":
            def complete_generate_rerun(result):
                if "manual_results" not in st.session_state:
                    st.session_state.manual_results = {}
                st.session_state.manual_results["generate"] = result
                st.session_state.last_manual_result_key = "generate"
                st.session_state.open_result_report = True

            render_selective_rerun_controls(
                st.session_state.get("last_generate_first_pass_plan"),
                "generate",
                complete_generate_rerun,
            )
        return

    result = st.session_state.get("last_chat_result")
    if not result:
        st.info("아직 작업 결과가 없습니다.")
        return
    render_recent_result_summary("대화형 작업", result)
    render_workflow_report(result, key_prefix="chat")
    render_selective_rerun_controls(
        st.session_state.get("last_chat_first_pass_plan"),
        "chat",
        lambda rerun_result: complete_chat_rerun(rerun_result, workspace),
    )


def render_conversation(workspace):
    st.markdown('<div class="panel-title">대화형 작업</div>', unsafe_allow_html=True)
    st.caption("원하는 데이터 작업을 자연어로 입력하세요. Workspace를 탐색한 뒤 실행 계획을 먼저 보여드립니다.")

    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = [{
            "role": "assistant",
            "content": (
                "어떤 작업을 진행할까요? (예: 현재 데이터셋의 라벨링 형식을 MS COCO 형식으로 바꿔줘)"
            ),
        }]
    if "pending_proposal" not in st.session_state:
        st.session_state.pending_proposal = None
    if "last_chat_result" not in st.session_state:
        st.session_state.last_chat_result = None
    if "last_chat_first_pass_plan" not in st.session_state:
        st.session_state.last_chat_first_pass_plan = None
    if "pending_preflight_preview" not in st.session_state:
        st.session_state.pending_preflight_preview = None

    for message in st.session_state.chat_messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if st.session_state.last_chat_result:
        st.info("최근 대화형 작업 결과는 `결과 리포트` 탭에서 확인하세요.")

    pending_proposal = st.session_state.pending_proposal
    if pending_proposal:
        pending_preflight = st.session_state.get("pending_preflight_preview")
        if pending_preflight:
            render_convert_preflight_preview(pending_preflight)
        approve_column, cancel_column, _ = st.columns([1, 1, 4])
        with approve_column:
            execute_chat_plan = st.button("계획 실행", type="primary", use_container_width=True)
        with cancel_column:
            cancel_chat_plan = st.button("취소", use_container_width=True)
        if execute_chat_plan:
            try:
                result = execute_plan(pending_proposal["plan"], auto_approve=True)
                response = describe_result(result, workspace)
                st.session_state.last_chat_result = result
                st.session_state.last_chat_first_pass_plan = (
                    pending_proposal["plan"]
                    if has_generate_operation(pending_proposal["plan"])
                    else None
                )
                st.session_state.open_result_report = True
            except Exception as exc:
                response = f"작업 실행 중 오류가 발생했습니다: {exc}"
            st.session_state.chat_messages.append({"role": "assistant", "content": response})
            st.session_state.pending_proposal = None
            st.session_state.pending_preflight_preview = None
            st.rerun()
        if cancel_chat_plan:
            st.session_state.chat_messages.append({"role": "assistant", "content": "작업을 취소했습니다."})
            st.session_state.pending_proposal = None
            st.session_state.pending_preflight_preview = None
            st.rerun()

    chat_placeholder = (
        "현재 실행 계획을 어떻게 수정할까요? 예: 출력 위치를 data/converted_test로 바꿔줘"
        if st.session_state.pending_proposal
        else "예: 현재 데이터셋의 라벨링 형식을 MS COCO 형식으로 바꿔줘"
    )
    chat_request = st.chat_input(chat_placeholder)
    if chat_request:
        st.session_state.chat_messages.append({"role": "user", "content": chat_request})
        st.session_state.last_chat_result = None
        st.session_state.last_chat_first_pass_plan = None
        st.session_state.pending_preflight_preview = None
        try:
            if st.session_state.pending_proposal:
                with st.status("실행 계획 수정 중", expanded=True) as status:
                    st.write("기존 실행 계획과 수정 요청을 LLM patcher에 전달합니다.")
                    try:
                        revision = revise_pending_proposal(
                            chat_request,
                            st.session_state.pending_proposal,
                            workspace,
                        )
                    except Exception:
                        status.update(label="실행 계획 수정 실패", state="error", expanded=True)
                        raise
                    status.update(label="수정안 생성 완료", state="complete", expanded=False)
                response = describe_plan_revision(revision, workspace)
                if revision["kind"] == "patch":
                    proposal = revision["proposal"]
                    st.session_state.pending_proposal = proposal
                    st.session_state.pending_preflight_preview = render_chat_preflight_for_proposal(proposal)
                elif revision["kind"] in {"cancel", "new_plan"}:
                    st.session_state.pending_proposal = None
                    st.session_state.pending_preflight_preview = None
            else:
                with st.status("요청 분석 중", expanded=True) as status:
                    st.write("Workspace를 탐색하고 요청 의도를 분석합니다.")
                    try:
                        routed = handle_conversation(chat_request, workspace)
                    except Exception:
                        status.update(label="요청 분석 실패", state="error", expanded=True)
                        raise
                    status.update(label="실행 계획 생성 완료", state="complete", expanded=False)
                if routed["kind"] == "plan":
                    proposal = first_pass_chat_proposal(routed["proposal"])
                    response = describe_plan(proposal, workspace)
                    st.session_state.pending_proposal = proposal
                    st.session_state.pending_preflight_preview = render_chat_preflight_for_proposal(proposal)
                else:
                    response = routed["response"]
        except (OSError, RuntimeError, ValueError) as exc:
            response = f"요청을 실행 계획으로 만들지 못했습니다: {exc}"
        st.session_state.chat_messages.append({"role": "assistant", "content": response})
        st.rerun()


st.markdown(
    """
    <div class="app-header">
      <div class="brand-mark">AL</div>
      <div>
        <div class="brand-name">AutoLabel</div>
        <div class="brand-meta">Vision dataset workspace</div>
      </div>
      <div class="runtime-state"><span class="runtime-dot"></span>Local runtime</div>
    </div>
    """,
    unsafe_allow_html=True,
)

if "workspace" not in st.session_state:
    saved_workspace = load_workspace()
    st.markdown('<div class="panel-title">Workspace 선택</div>', unsafe_allow_html=True)
    st.caption(
        "표준 폴더 구조 생성 체크 시 workspace 아래에 다음 폴더가 자동 생성됩니다.  \n"
        "`data/raw` — 원본 이미지  \n"
        "`data/labeled` — 생성된 라벨  \n"
        "`data/visualized` — 라벨 시각화 결과  \n"
        "`data/converted` — 형식 변환 라벨  \n"
        "`data/ground_truth` — 평가용 정답 라벨  \n"
        "`data/reports` — 평가 리포트  \n"
        "`configs/plugins.json` — 플러그인 설정 파일"
    )
    create_layout = st.checkbox(
        "표준 폴더 구조 생성",
        value=not bool(saved_workspace),
        help="이미 작업 중인 디렉터리를 그대로 사용하려면 체크 해제하세요.",
    )
    with st.form("workspace_form"):
        selected_workspace = st.text_input(
            "Workspace 경로",
            value=str(saved_workspace or (Path.home() / "Documents" / "AutoLabelWorkspace")),
            placeholder="D:/AutoLabelWorkspace",
        )
        workspace_submit = st.form_submit_button(
            "Workspace 적용",
            type="primary",
            icon=":material/folder_open:",
        )
    if workspace_submit:
        if not selected_workspace.strip():
            st.error("Workspace 경로를 입력하세요.")
        else:
            try:
                st.session_state.workspace = str(save_workspace(selected_workspace, create_layout=create_layout))
                st.rerun()
            except OSError as exc:
                st.error(f"Workspace 생성 실패: {exc}")
    st.stop()

workspace = st.session_state.workspace
st.markdown(f'<div class="workspace-path">{html.escape(workspace)}</div>', unsafe_allow_html=True)
chat_tab, convert_tab, generate_tab, evaluate_tab, result_tab, settings_tab = st.tabs(
    ["대화형 작업", "형식 변환", "라벨 생성", "평가", "결과 리포트", "설정"]
)

if st.session_state.pop("open_result_report", False):
    components.html(
        """
        <script>
        const labels = Array.from(window.parent.document.querySelectorAll('[role="tab"]'));
        const target = labels.find((el) => el.textContent.includes('결과 리포트'));
        if (target) target.click();
        </script>
        """,
        height=0,
    )

with chat_tab:
    render_conversation(workspace)

with result_tab:
    render_result_report(workspace)

with convert_tab:
    st.markdown('<div class="panel-title">라벨 형식 변환</div>', unsafe_allow_html=True)
    label_path_candidates = workspace_dir_candidates(
        workspace,
        {"labels", "labeled"},
        WORKSPACE_DEFAULTS["labels"],
    )
    image_dir_candidates = workspace_dir_candidates(
        workspace,
        {"raw", "images"},
        WORKSPACE_DEFAULTS["images"],
    )
    direct_option = "직접 입력..."
    label_path_options = [*label_path_candidates, direct_option]
    image_dir_options = [*image_dir_candidates, direct_option]
    with st.container(border=True):
        st.markdown('<span class="path-panel-marker"></span>', unsafe_allow_html=True)
        st.markdown('<div class="form-section">데이터 경로</div>', unsafe_allow_html=True)
        path_left, path_right = st.columns(2)
        with path_left:
            selected_label_path = st.selectbox("라벨 폴더 선택", label_path_options)
            input_path = (
                st.text_input("라벨 폴더 직접 입력", value=WORKSPACE_DEFAULTS["labels"])
                if selected_label_path == direct_option
                else selected_label_path
            )
            st.caption(
                "라벨 파일이 들어 있는 폴더만 지정하세요. "
                "결과 폴더(converted/reports/visualized)가 섞이면 불필요한 파일도 검사될 수 있습니다."
            )
        with path_right:
            selected_image_dir = st.selectbox("이미지 폴더 선택", image_dir_options)
            image_dir = (
                st.text_input("이미지 폴더 직접 입력", value=WORKSPACE_DEFAULTS["images"])
                if selected_image_dir == direct_option
                else selected_image_dir
            )
            st.caption("라벨 파일명과 같은 원본 이미지가 들어 있는 폴더를 지정하세요.")
    with st.form("convert_form"):
        left, right = st.columns(2)
        with left:
            st.markdown('<div class="form-section">출력 설정</div>', unsafe_allow_html=True)
            output_dir = st.text_input("출력 디렉터리", value=WORKSPACE_DEFAULTS["converted"])
            convert_classes = st.text_input(
                "클래스 매핑 파일",
                value="",
                placeholder="YOLO data.yaml, dataset.yaml 또는 classes.txt",
            )
        with right:
            st.markdown('<div class="form-section">변환 규칙</div>', unsafe_allow_html=True)
            source_format = st.selectbox("입력 파일 형식", SOURCE_OPTIONS)
            target_formats = st.multiselect("출력 포맷", FORMAT_OPTIONS, default=["yolo"])
            duplicate_iou = st.slider("중복 IoU", 0.01, 1.0, 0.85, 0.01)
            convert_insight_ratio = st.slider(
                "불균형 비율 기준",
                1.1,
                10.0,
                3.0,
                0.1,
                key="convert_insight_ratio",
            )
            strict = st.checkbox("입력 데이터 문제가 있는 데이터 제외")
        preflight_submit = st.form_submit_button("변환 사전 점검", icon=":material/rule:")
        convert_submit = st.form_submit_button("변환 실행", type="primary", icon=":material/sync_alt:")
    if preflight_submit:
        if not input_path or not image_dir or not target_formats:
            st.error("입력 라벨 경로, 이미지 디렉터리, 출력 포맷을 지정하세요.")
        else:
            try:
                st.session_state.convert_preflight_preview = build_convert_preflight_preview(
                    resolve_workspace_path(workspace, input_path),
                    resolve_workspace_path(workspace, image_dir),
                    source_format,
                    target_formats,
                    classes_path=resolve_workspace_path(workspace, convert_classes) if convert_classes else None,
                    duplicate_iou=duplicate_iou,
                )
            except Exception as exc:
                st.error(f"사전 점검 실패: {exc}")
    if st.session_state.get("convert_preflight_preview"):
        render_convert_preflight_preview(st.session_state.convert_preflight_preview)
    if convert_submit:
        if not input_path or not image_dir or not output_dir or not target_formats:
            st.error("입력, 이미지, 출력 경로와 출력 포맷을 모두 지정하세요.")
        else:
            run_plan({
                "request_summary": "Streamlit label conversion",
                "operations": [{
                    "action": "convert",
                    "input_path": resolve_workspace_path(workspace, input_path),
                    "img_dir": resolve_workspace_path(workspace, image_dir),
                    "out_dir": resolve_workspace_path(workspace, output_dir),
                    "source_format": source_format,
                    "formats": target_formats,
                    "classes_path": resolve_workspace_path(workspace, convert_classes) if convert_classes else None,
                    "duplicate_iou": duplicate_iou,
                    "insight_imbalance_ratio": convert_insight_ratio,
                    "strict": strict,
                }],
            }, result_key="convert")
with generate_tab:
    st.markdown('<div class="panel-title">자동 라벨 생성</div>', unsafe_allow_html=True)
    with st.form("generate_form"):
        left, right = st.columns(2)
        with left:
            st.markdown('<div class="form-section">데이터 경로</div>', unsafe_allow_html=True)
            generation_images = st.text_input("이미지 디렉터리", value=WORKSPACE_DEFAULTS["images"], key="generation_images")
            generation_output = st.text_input("라벨 출력", value=WORKSPACE_DEFAULTS["labels"])
            visualization_output = st.text_input("시각화 출력", value=WORKSPACE_DEFAULTS["visualized"])
            plugin_config = st.text_input("Plugin 설정 파일", value=WORKSPACE_DEFAULTS["plugin_config"])
            generation_classes = st.text_input("클래스 매핑 파일", value="", placeholder="data.yaml 또는 classes.txt")
        with right:
            st.markdown('<div class="form-section">생성 규칙</div>', unsafe_allow_html=True)
            task_type = st.selectbox("태스크", TASK_OPTIONS)
            generation_formats = st.multiselect("출력 포맷", FORMAT_OPTIONS, default=["yolo"], key="generation_formats")
            threshold = st.slider("신뢰도 기준", 0.0, 1.0, 0.75, 0.01)
            generation_insight_ratio = st.slider(
                "불균형 비율 기준",
                1.1,
                10.0,
                3.0,
                0.1,
                key="generation_insight_ratio",
            )
            inference_count = st.number_input("초안 추론 횟수", 1, 10, 3)
            st.caption("Specialist 재추론은 1차 결과 확인 후 결과 리포트 아래에서 선택 실행합니다.")
        prompt = st.text_area(
            "프롬프트",
            value="",
            placeholder="이미지에서 눈에 띄는 모든 객체를 탐지하고 분류해 주세요.",
        )
        approve_expensive = st.checkbox("고비용 모델 API 호출 승인")
        generate_submit = st.form_submit_button("라벨 생성 실행", type="primary", icon=":material/auto_awesome:")
    if generate_submit:
        if not generation_images or not generation_output or not visualization_output or not generation_formats:
            st.error("이미지, 라벨 출력, 시각화 출력 경로와 출력 포맷을 모두 지정하세요.")
        elif not prompt.strip():
            st.error("라벨 생성 프롬프트를 입력하세요.")
        elif not approve_expensive:
            st.error("모델 호출 승인이 필요합니다.")
        else:
            generate_plan = {
                "request_summary": "Streamlit automatic label generation",
                "operations": [{
                    "action": "generate",
                    "task_type": task_type,
                    "img_dir": resolve_workspace_path(workspace, generation_images),
                    "out_dir": resolve_workspace_path(workspace, generation_output),
                    "vis_dir": resolve_workspace_path(workspace, visualization_output),
                    "formats": generation_formats,
                    "threshold": threshold,
                    "insight_imbalance_ratio": generation_insight_ratio,
                    "inference_count": int(inference_count),
                    "prompt": prompt,
                    "generation_strategy": "specialist_first",
                    "specialist_consistency_runs": 0,
                    "specialist_advisor_mode": "none",
                    "classes_path": resolve_workspace_path(workspace, generation_classes) if generation_classes else None,
                    "plugin_config": resolve_workspace_path(workspace, plugin_config) if plugin_config else None,
                    "require_approval": True,
                }],
            }
            run_plan(
                generate_plan,
                auto_approve=True,
                result_key="generate",
                first_pass_plan_key="last_generate_first_pass_plan",
            )
with evaluate_tab:
    st.markdown('<div class="panel-title">실험 결과 평가</div>', unsafe_allow_html=True)
    with st.form("evaluate_form"):
        ground_truth = st.text_input("Ground truth 디렉터리", value=WORKSPACE_DEFAULTS["ground_truth"])
        report_output = st.text_input("리포트 출력", value=WORKSPACE_DEFAULTS["reports"])
        run_lines = st.text_area(
            "실험 경로",
            placeholder="baseline=D:/runs/baseline\ncascade=D:/runs/cascade",
        )
        evaluate_submit = st.form_submit_button("평가 실행", type="primary", icon=":material/analytics:")
    if evaluate_submit:
        try:
            runs = parse_runs(run_lines)
            if not report_output:
                raise ValueError("리포트 출력 경로를 입력하세요.")
        except ValueError as exc:
            st.error(str(exc))
        else:
            run_plan({
                "request_summary": "Streamlit experiment evaluation",
                "operations": [{
                    "action": "evaluate",
                    "gt_dir": resolve_workspace_path(workspace, ground_truth) if ground_truth else None,
                    "out_dir": resolve_workspace_path(workspace, report_output),
                    "runs": {
                        name: resolve_workspace_path(workspace, path)
                        for name, path in runs.items()
                    },
                }],
            }, result_key="evaluate")
with settings_tab:
    st.markdown('<div class="panel-title">사용자 설정</div>', unsafe_allow_html=True)
    settings = read_user_settings()
    labels = {
        "AWS_REGION": "AWS Region",
        "AWS_PROFILE": "AWS Profile",
        "AWS_ACCESS_KEY_ID": "AWS Access Key ID",
        "AWS_SECRET_ACCESS_KEY": "AWS Secret Access Key",
        "AWS_SESSION_TOKEN": "AWS Session Token",
        "OPENAI_API_KEY": "OpenAI API Key",
        "ANTHROPIC_API_KEY": "Anthropic API Key",
        "LOW_MODEL": "Low Model",
        "HIGH_MODEL": "High Model",
        "PLANNER_MODEL": "Planner Model",
        "INTENT_ROUTER_MODEL": "Intent Router Model",
        "CHAT_MODEL": "Chat Model",
    }
    with st.form("settings_form"):
        workspace_value = st.text_input("Workspace 경로", value=workspace, key="setting_workspace")
        columns = st.columns(2)
        values = {}
        groups = [
            ("AWS 인증", ENV_FIELDS[:5]),
            ("API 및 모델", ENV_FIELDS[5:]),
        ]
        for column, (group_name, keys) in zip(columns, groups):
            with column:
                st.markdown(f'<div class="form-section">{group_name}</div>', unsafe_allow_html=True)
                for key in keys:
                    values[key] = st.text_input(
                        labels[key],
                        value=settings.get(key, ""),
                        type="password" if key in SECRET_FIELDS else "default",
                        key=f"setting_{key}",
                    )
        settings_submit = st.form_submit_button("설정 저장", type="primary", icon=":material/save:")
    if settings_submit:
        if not workspace_value.strip():
            st.error("Workspace 경로를 입력하세요.")
        else:
            try:
                st.session_state.workspace = str(save_workspace(workspace_value, create_layout=False))
                saved_path = save_user_settings(values)
                st.success(f"설정을 저장했습니다: {saved_path}")
            except OSError as exc:
                st.error(f"설정 저장 실패: {exc}")
