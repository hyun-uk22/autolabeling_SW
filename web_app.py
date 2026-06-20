import html
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
from src.workflow.service import execute_workflow_plan
from src.workflow.conversation import (
    build_conversation_plan,
    describe_plan,
    describe_result,
)


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


def run_plan(plan, auto_approve=False):
    try:
        result = execute_plan(plan, auto_approve=auto_approve)
        if result.get("status") == "completed":
            st.success("작업이 완료되었습니다.")
        else:
            st.warning(f"작업 상태: {result.get('status', 'unknown')}")
        st.json({
            "status": result.get("status"),
            "outputs": result.get("operation_outputs", []),
            "errors": result.get("errors", []),
            "history_path": result.get("history_path", ""),
        })
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

st.markdown('<div class="panel-title">대화형 작업</div>', unsafe_allow_html=True)
st.caption("원하는 데이터 작업을 자연어로 입력하세요. Workspace를 탐색한 뒤 실행 계획을 먼저 보여드립니다.")

if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = [{
        "role": "assistant",
        "content": (
            "어떤 작업을 진행할까요? 예: `현재 데이터셋의 라벨링 형식을 MS COCO 형식으로 바꿔줘`"
        ),
    }]
if "pending_proposal" not in st.session_state:
    st.session_state.pending_proposal = None

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
    try:
        proposal = build_conversation_plan(chat_request, workspace)
        response = describe_plan(proposal, workspace)
        st.session_state.pending_proposal = proposal
    except (OSError, ValueError) as exc:
        response = f"요청을 실행 계획으로 만들지 못했습니다: {exc}"
    st.session_state.chat_messages.append({"role": "assistant", "content": response})
    st.rerun()

st.divider()
st.markdown('<div class="panel-title">고급 작업</div>', unsafe_allow_html=True)
st.caption("경로와 실행 옵션을 직접 지정하려면 아래 기능을 사용하세요.")
convert_tab, generate_tab, evaluate_tab, settings_tab = st.tabs(["형식 변환", "라벨 생성", "평가", "설정"])

with convert_tab:
    st.markdown('<div class="panel-title">라벨 형식 변환</div>', unsafe_allow_html=True)
    with st.form("convert_form"):
        left, right = st.columns(2)
        with left:
            st.markdown('<div class="form-section">데이터 경로</div>', unsafe_allow_html=True)
            input_path = st.text_input("입력 라벨 경로", value=WORKSPACE_DEFAULTS["labels"])
            image_dir = st.text_input("이미지 디렉터리", value=WORKSPACE_DEFAULTS["images"])
            output_dir = st.text_input("출력 디렉터리", value=WORKSPACE_DEFAULTS["converted"])
        with right:
            st.markdown('<div class="form-section">변환 규칙</div>', unsafe_allow_html=True)
            source_format = st.selectbox("입력 포맷", SOURCE_OPTIONS)
            target_formats = st.multiselect("출력 포맷", FORMAT_OPTIONS, default=["yolo"])
            duplicate_iou = st.slider("중복 IoU", 0.01, 1.0, 0.85, 0.01)
            strict = st.checkbox("검증 이슈가 있는 레코드 제외")
        convert_submit = st.form_submit_button("변환 실행", type="primary", icon=":material/sync_alt:")
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
                    "duplicate_iou": duplicate_iou,
                    "strict": strict,
                }],
            })

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
        with right:
            st.markdown('<div class="form-section">생성 규칙</div>', unsafe_allow_html=True)
            task_type = st.selectbox("태스크", TASK_OPTIONS)
            generation_formats = st.multiselect("출력 포맷", FORMAT_OPTIONS, default=["yolo"], key="generation_formats")
            threshold = st.slider("신뢰도 기준", 0.0, 1.0, 0.75, 0.01)
            inference_count = st.number_input("초안 추론 횟수", 1, 10, 3)
        prompt = st.text_area(
            "프롬프트",
            "Detect and classify all prominent objects in this image. Output strictly as JSON.",
        )
        approve_expensive = st.checkbox("고비용 모델 API 호출 승인")
        generate_submit = st.form_submit_button("라벨 생성 실행", type="primary", icon=":material/auto_awesome:")
    if generate_submit:
        if not generation_images or not generation_output or not visualization_output or not generation_formats:
            st.error("이미지, 라벨 출력, 시각화 출력 경로와 출력 포맷을 모두 지정하세요.")
        elif not approve_expensive:
            st.error("모델 호출 승인이 필요합니다.")
        else:
            run_plan({
                "request_summary": "Streamlit automatic label generation",
                "operations": [{
                    "action": "generate",
                    "task_type": task_type,
                    "img_dir": resolve_workspace_path(workspace, generation_images),
                    "out_dir": resolve_workspace_path(workspace, generation_output),
                    "vis_dir": resolve_workspace_path(workspace, visualization_output),
                    "formats": generation_formats,
                    "threshold": threshold,
                    "inference_count": int(inference_count),
                    "prompt": prompt,
                    "plugin_config": resolve_workspace_path(workspace, plugin_config) if plugin_config else None,
                    "require_approval": True,
                }],
            }, auto_approve=True)

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
            })

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
