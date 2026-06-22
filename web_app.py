import html
import json
import copy
import uuid
from pathlib import Path

import streamlit as st

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
from src.utils.label_importer import find_image_path, import_labels_with_report
from src.utils.label_validator import summarize_validation, validate_result
from src.workflow.service import execute_workflow_plan
from src.workflow.conversation import (
    describe_plan,
    describe_result,
)
from src.workflow.conversation_router import handle_conversation


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


def execute_plan(plan, auto_approve=False):
    with st.spinner("작업 실행 중"):
        return execute_workflow_plan(
            plan,
            auto_approve=auto_approve,
            thread_id=f"streamlit-{uuid.uuid4()}",
        )


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
            columns[0].metric("읽은 레코드", output.get("records_read", 0))
            columns[1].metric("변환 완료", output.get("records_converted", 0))
            columns[2].metric("검증 이슈", validation.get("failed_records", 0))
            columns[3].metric(
                "출력 이슈",
                output.get("export_validation", {}).get("failed_records", 0),
            )
            preflight = output.get("preflight", {})
            if preflight.get("notices"):
                st.markdown("**변환 전 확인 사항**")
                st.dataframe([
                    {
                        "심각도": notice.get("severity", ""),
                        "항목": notice.get("code", ""),
                        "내용": notice.get("message", ""),
                        "필요 조치": notice.get("user_action", ""),
                    }
                    for notice in preflight.get("notices", [])
                ], width="stretch")
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
            columns[3].metric("Artifact 이슈", summary.get("artifact_issues", 0))
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


def run_plan(plan, auto_approve=False, result_key="workflow", first_pass_plan_key=None):
    try:
        result = execute_plan(plan, auto_approve=auto_approve)
        if "manual_results" not in st.session_state:
            st.session_state.manual_results = {}
        st.session_state.manual_results[result_key] = result
        if first_pass_plan_key:
            st.session_state[first_pass_plan_key] = (
                plan if has_generate_operation(plan) else None
            )
        render_manual_result(result, result_key)
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
        validations.append({
            "image": image_name,
            "issues": validate_result(result, image_path),
        })
    return {
        "input_summary": batch.report,
        "validation": summarize_validation(validations),
        "preflight": build_conversion_preflight(batch.report, target_formats, validations),
        "records": validations,
    }


def render_convert_preflight_preview(preview):
    preflight = preview.get("preflight", {})
    validation = preview.get("validation", {})
    summary = preview.get("input_summary", {})
    columns = st.columns(4)
    columns[0].metric("발견 소스", summary.get("sources_discovered", 0))
    columns[1].metric("변환 대상", summary.get("records_after_merge", 0))
    columns[2].metric("검증 이슈", validation.get("failed_records", 0))
    columns[3].metric("상태", preflight.get("status", "unknown"))
    if preflight.get("notices"):
        st.markdown("**실행 전 확인 사항**")
        st.dataframe([
            {
                "심각도": notice.get("severity", ""),
                "항목": notice.get("code", ""),
                "내용": notice.get("message", ""),
                "필요 조치": notice.get("user_action", ""),
            }
            for notice in preflight.get("notices", [])
        ], width="stretch")
    else:
        st.success("변환 전에 확인된 필수 누락 정보가 없습니다.")
    detailed = [
        {
            "파일": record.get("image", ""),
            "이슈": " / ".join(record.get("issues", [])),
        }
        for record in preview.get("records", [])
        if record.get("issues")
    ]
    if detailed:
        with st.expander("검증 이슈 상세"):
            st.dataframe(detailed, width="stretch")


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

    for message in st.session_state.chat_messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    pending_proposal = st.session_state.pending_proposal
    if pending_proposal:
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
            except Exception as exc:
                response = f"작업 실행 중 오류가 발생했습니다: {exc}"
            st.session_state.chat_messages.append({"role": "assistant", "content": response})
            st.session_state.pending_proposal = None
            st.rerun()
        if cancel_chat_plan:
            st.session_state.chat_messages.append({"role": "assistant", "content": "작업을 취소했습니다."})
            st.session_state.pending_proposal = None
            st.rerun()

    chat_request = st.chat_input(
        "예: 현재 데이터셋의 라벨링 형식을 MS COCO 형식으로 바꿔줘",
        disabled=bool(st.session_state.pending_proposal),
    )
    if chat_request:
        st.session_state.chat_messages.append({"role": "user", "content": chat_request})
        st.session_state.last_chat_result = None
        st.session_state.last_chat_first_pass_plan = None
        try:
            routed = handle_conversation(chat_request, workspace)
            if routed["kind"] == "plan":
                proposal = first_pass_chat_proposal(routed["proposal"])
                response = describe_plan(proposal, workspace)
                st.session_state.pending_proposal = proposal
            else:
                response = routed["response"]
        except (OSError, ValueError) as exc:
            response = f"요청을 실행 계획으로 만들지 못했습니다: {exc}"
        st.session_state.chat_messages.append({"role": "assistant", "content": response})
        st.rerun()

    if st.session_state.last_chat_result:
        render_workflow_report(st.session_state.last_chat_result, key_prefix="chat")
        def complete_chat_rerun(result):
            response = "선택 재추론을 완료했습니다.\n\n" + describe_result(result, workspace)
            st.session_state.last_chat_result = result
            st.session_state.chat_messages.append({"role": "assistant", "content": response})

        render_selective_rerun_controls(
            st.session_state.last_chat_first_pass_plan,
            "chat",
            complete_chat_rerun,
        )


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
chat_tab, convert_tab, generate_tab, evaluate_tab, settings_tab = st.tabs(
    ["대화형 작업", "형식 변환", "라벨 생성", "평가", "설정"]
)

with chat_tab:
    render_conversation(workspace)

with convert_tab:
    st.markdown('<div class="panel-title">라벨 형식 변환</div>', unsafe_allow_html=True)
    with st.form("convert_form"):
        left, right = st.columns(2)
        with left:
            st.markdown('<div class="form-section">데이터 경로</div>', unsafe_allow_html=True)
            input_path = st.text_input("입력 라벨 경로", value=WORKSPACE_DEFAULTS["labels"])
            image_dir = st.text_input("이미지 디렉터리", value=WORKSPACE_DEFAULTS["images"])
            output_dir = st.text_input("출력 디렉터리", value=WORKSPACE_DEFAULTS["converted"])
            convert_classes = st.text_input(
                "클래스 매핑 파일",
                value="",
                placeholder="YOLO data.yaml, dataset.yaml 또는 classes.txt",
            )
        with right:
            st.markdown('<div class="form-section">변환 규칙</div>', unsafe_allow_html=True)
            source_format = st.selectbox("입력 포맷", SOURCE_OPTIONS)
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
            strict = st.checkbox("검증 이슈가 있는 레코드 제외")
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
    elif st.session_state.get("manual_results", {}).get("convert"):
        render_manual_result(st.session_state.manual_results["convert"], "convert")

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
    elif st.session_state.get("manual_results", {}).get("generate"):
        render_manual_result(st.session_state.manual_results["generate"], "generate")
    if st.session_state.get("manual_results", {}).get("generate"):
        def complete_generate_rerun(result):
            if "manual_results" not in st.session_state:
                st.session_state.manual_results = {}
            st.session_state.manual_results["generate"] = result

        render_selective_rerun_controls(
            st.session_state.get("last_generate_first_pass_plan"),
            "generate",
            complete_generate_rerun,
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
    elif st.session_state.get("manual_results", {}).get("evaluate"):
        render_manual_result(st.session_state.manual_results["evaluate"], "evaluate")

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
