import html
import json
import copy
import os
import uuid
from pathlib import Path

import streamlit as st
from PIL import Image


def patch_streamlit_drawable_canvas_image_url():
    try:
        import streamlit.elements.image as st_image
        from streamlit.elements.lib import image_utils
        from streamlit.elements.lib.layout_utils import LayoutConfig
    except Exception:
        return
    if hasattr(st_image, "image_to_url"):
        return

    def image_to_url(image, width, clamp, channels, output_format, image_id):
        return image_utils.image_to_url(
            image,
            LayoutConfig(width=width),
            clamp,
            channels,
            output_format,
            image_id,
        )

    st_image.image_to_url = image_to_url


patch_streamlit_drawable_canvas_image_url()

try:
    from streamlit_drawable_canvas import st_canvas
except Exception:  # pragma: no cover - optional Streamlit component
    st_canvas = None

from src.core.models import (
    BoundingBox,
    ClassificationLabel,
    DetectionResult,
    Keypoint,
    Point,
    PolygonSegment,
    PoseInstance,
    TextRegion,
    TrackInstance,
)
from src.core.model_config import resolve_planner_model
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
from src.utils.format_converter import LabelExportWriter
from src.utils.custom_label_mapper import (
    CUSTOM_MAPPING_FORMAT,
    infer_custom_mapping_spec_from_sample,
    sample_custom_label_file,
)
from src.utils.label_importer import find_image_path, import_labels_with_report
from src.utils.label_validator import summarize_validation, validate_result
from src.workflow.service import execute_workflow_plan
from src.workflow.conversation import (
    describe_plan,
    describe_result,
)
from src.workflow.conversation_router import handle_conversation
from src.workflow.plan_patcher import revise_pending_proposal
from src.ui.polygon_editor import polygon_vertex_editor


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
SOURCE_OPTIONS = ["auto", "yolo", "pascal_voc", "coco", "vision_json", "csv", "generic_json", CUSTOM_MAPPING_FORMAT]
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


def editable_path_selectbox(label, candidates, state_key, help_text):
    current = st.session_state.get(state_key) or candidates[0]
    options = list(dict.fromkeys([current] + candidates))
    selected = st.selectbox(
        label,
        options,
        index=options.index(current),
        key=state_key,
        accept_new_options=True,
        help=help_text,
    )
    return selected or current


def split_issue_file_cell(value):
    return [part.strip() for part in str(value or "").splitlines() if part.strip()]


def unique_issue_files(values):
    files = []
    for value in values:
        for file_name in split_issue_file_cell(value):
            if file_name and file_name not in files:
                files.append(file_name)
    return files


def editor_context_signature(context):
    keys = ["label_path", "image_dir", "source_format", "classes_path", "task_type"]
    return json.dumps({key: context.get(key, "") for key in keys}, ensure_ascii=False, sort_keys=True)


def start_label_editor_issue_queue(workspace, files, context):
    queue = unique_issue_files(files)
    if not queue:
        st.warning("라벨 편집으로 보낼 문제 파일이 없습니다.")
        return
    context = dict(context or {})
    context.setdefault("source_format", "auto")
    context.setdefault("task_type", "object_detection")
    context.setdefault("output_dir", WORKSPACE_DEFAULTS["labels"])
    for key, state_key in {
        "label_path": "editor_label_path",
        "image_dir": "editor_image_dir",
        "source_format": "editor_source_format",
        "classes_path": "editor_classes",
        "task_type": "editor_task",
        "output_dir": "editor_output",
    }.items():
        value = context.get(key)
        if value:
            st.session_state[state_key] = _relative_workspace_path(workspace, value) if Path(str(value)).is_absolute() else str(value)
    st.session_state.editor_issue_queue = queue
    st.session_state.editor_issue_index = 0
    st.session_state.editor_issue_context = context
    st.session_state.editor_loaded_context_signature = ""
    st.session_state.editor_selected_image_select = queue[0]
    st.session_state.open_label_editor = True
    st.rerun()


def start_label_editor_context(workspace, context):
    context = dict(context or {})
    context.setdefault("source_format", "auto")
    context.setdefault("task_type", "object_detection")
    context.setdefault("output_dir", WORKSPACE_DEFAULTS["labels"])
    if context.get("source_format") not in SOURCE_OPTIONS:
        context["source_format"] = "auto"
    for key, state_key in {
        "label_path": "editor_label_path",
        "image_dir": "editor_image_dir",
        "source_format": "editor_source_format",
        "classes_path": "editor_classes",
        "task_type": "editor_task",
        "output_dir": "editor_output",
    }.items():
        value = context.get(key)
        if value:
            st.session_state[state_key] = _relative_workspace_path(workspace, value) if Path(str(value)).is_absolute() else str(value)
    st.session_state.editor_issue_queue = []
    st.session_state.editor_issue_index = 0
    st.session_state.editor_issue_context = {}
    st.session_state.editor_loaded_context_signature = ""
    st.session_state.editor_auto_load = True
    st.session_state.open_label_editor = True


def render_issue_editor_launcher(files, context, key_prefix, title="문제 파일 라벨 편집", filter_to_issue_queue=False):
    issue_files = unique_issue_files(files)
    if not issue_files:
        return
    context = dict(context or {})
    if filter_to_issue_queue:
        context["filter_to_issue_queue"] = True
    with st.container(border=True):
        st.markdown(f"**{title}**")
        st.caption(
            "이미지 파일이 연결되지 않은 라벨은 표 편집만 가능합니다. "
            "bbox/polygon을 마우스로 수정하려면 라벨과 연결되는 원본 이미지 경로를 함께 지정하세요."
        )
        selected = st.multiselect(
            "라벨 편집으로 보낼 파일",
            issue_files,
            default=issue_files,
            key=f"{key_prefix}-issue-editor-files",
            help="선택한 파일 순서대로 라벨 편집 탭에서 이전/다음 버튼으로 검토합니다.",
        )
        if st.button("선택 파일을 라벨 편집에서 열기", key=f"{key_prefix}-open-editor", icon=":material/edit:"):
            start_label_editor_issue_queue(st.session_state.workspace, selected, context)


def workflow_editor_context(output):
    action = output.get("action")
    if action == "convert":
        label_path = output.get("input") or output.get("input_path") or ""
        image_dir = output.get("img_dir") or output.get("image_dir") or ""
        if not image_dir and label_path:
            image_dir = str(Path(label_path).parent)
        source_format = output.get("resolved_source_format") or output.get("source_format") or "auto"
        if source_format in {"mixed", "unknown"}:
            source_format = "auto"
        return {
            "label_path": label_path,
            "image_dir": image_dir,
            "source_format": source_format,
            "classes_path": output.get("classes_path", ""),
            "task_type": output.get("task_type", "object_detection"),
            "output_dir": output.get("output_dir") or WORKSPACE_DEFAULTS["labels"],
        }
    report_path = output.get("summary_path") or output.get("report_path") or output.get("user_action_report_path") or ""
    output_dir = str(Path(report_path).parent) if report_path else output.get("output_dir", WORKSPACE_DEFAULTS["labels"])
    return {
        "label_path": output_dir,
        "image_dir": output.get("image_dir") or output.get("img_dir") or output.get("source_dir") or WORKSPACE_DEFAULTS["images"],
        "source_format": "auto",
        "classes_path": output.get("classes_path", ""),
        "task_type": output.get("task_type", "object_detection"),
        "output_dir": output_dir,
    }


def generation_editor_context_options(output):
    options = [("1차 Vision Model 라벨", workflow_editor_context(output))]
    llm_exports = output.get("llm_exports") or {}
    output_root = llm_exports.get("output_root")
    artifacts = llm_exports.get("artifacts") or {}
    if output_root:
        for level in ("low", "high"):
            if level in artifacts:
                context = workflow_editor_context(output)
                context["label_path"] = str(Path(output_root) / level)
                context["output_dir"] = context["label_path"]
                options.append((f"{level.upper()} LMM 재생성 라벨", context))
    review_root = llm_exports.get("review_root")
    review_artifacts = llm_exports.get("review_artifacts") or {}
    if review_root:
        for level in ("low", "high"):
            if level in review_artifacts:
                context = workflow_editor_context(output)
                context["label_path"] = str(Path(review_root) / level)
                context["output_dir"] = context["label_path"]
                options.append((f"{level.upper()} LMM 검토 후보 라벨", context))
    return options


def default_editor_save_formats(source_format, report=None):
    if source_format in FORMAT_OPTIONS:
        return [source_format]
    formats = []
    for name in (report or {}).get("formats", {}):
        if name in FORMAT_OPTIONS and name not in formats:
            formats.append(name)
    return formats or ["vision_json"]


def find_images_recursive(image_dir):
    root = Path(image_dir)
    if not root.exists():
        return []
    image_exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    return [
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in image_exts
    ]


def generation_missing_info(
    workspace,
    image_dir,
    label_output,
    visualization_output,
    formats,
    prompt,
    approve_expensive,
    class_mapping,
    class_list_text,
):
    missing = []
    if not image_dir:
        missing.append(("이미지 디렉터리", "라벨을 생성할 이미지 폴더를 입력하세요."))
    else:
        resolved_image_dir = resolve_workspace_path(workspace, image_dir)
        if not Path(resolved_image_dir).exists():
            missing.append(("이미지 디렉터리", f"경로를 찾을 수 없습니다: {resolved_image_dir}"))
        elif not find_images_recursive(resolved_image_dir):
            missing.append(("이미지 파일", "이미지 디렉터리 아래에서 JPG/PNG/WEBP/BMP 이미지를 찾지 못했습니다."))
    if not label_output:
        missing.append(("라벨 출력", "생성된 라벨을 저장할 경로를 입력하세요."))
    if not visualization_output:
        missing.append(("시각화 출력", "오버레이/미리보기 결과를 저장할 경로를 입력하세요."))
    if not formats:
        missing.append(("출력 포맷", "YOLO, COCO, Pascal VOC, vision_json 중 하나 이상을 선택하세요."))
    if not prompt.strip() and not class_mapping and not class_list_text.strip():
        missing.append(("검출 지시", "프롬프트, 클래스 매핑 파일, 또는 검출 클래스 목록 중 하나를 입력하세요."))
    if not approve_expensive:
        missing.append(("모델 호출 승인", "라벨 생성을 실행하려면 모델 호출 승인이 필요합니다."))
    return missing


def build_generation_prompt(prompt, class_list_text):
    prompt = prompt.strip()
    classes = [line.strip() for line in class_list_text.replace(",", "\n").splitlines() if line.strip()]
    if not classes:
        return prompt
    class_instruction = "검출 대상 클래스는 다음 목록으로 제한하세요: " + ", ".join(classes)
    return f"{prompt}\n{class_instruction}".strip() if prompt else class_instruction


def class_mapping_candidates(workspace, input_path):
    root = Path(workspace).expanduser().resolve()
    try:
        label_path = Path(resolve_workspace_path(workspace, input_path))
    except (OSError, ValueError):
        label_path = root / str(input_path)
    base_dir = label_path if label_path.is_dir() else label_path.parent
    search_dirs = []
    current = base_dir.resolve()
    while True:
        try:
            current.relative_to(root)
        except ValueError:
            break
        search_dirs.append(current)
        if current == root or current.parent == current:
            break
        current = current.parent

    filenames = ("data.yaml", "data.yml", "dataset.yaml", "dataset.yml", "classes.txt")
    candidates = []
    for directory in search_dirs:
        for filename in filenames:
            path = directory / filename
            if path.is_file():
                candidates.append(_relative_workspace_path(root, path))
        for path in sorted(directory.glob("*.y*ml")):
            if path.name.lower() not in filenames:
                candidates.append(_relative_workspace_path(root, path))
    return list(dict.fromkeys(candidates))


def optional_class_mapping_path(workspace, value):
    if not value or str(value).strip() == "자동 탐색":
        return None
    return resolve_workspace_path(workspace, str(value).strip())


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
                    "BBox Self-Consistency",
                    "-" if mean_agreement is None else f"{mean_agreement * 100:.1f}%",
                )
            if specialist_consistency.get("llm_enabled_images"):
                st.markdown("**LMM 재생성 일관성 분석**")
                llm_columns = st.columns(4)
                llm_columns[0].metric("LMM 비교 이미지", specialist_consistency.get("llm_enabled_images", 0))
                llm_columns[1].metric("LMM 모드", specialist_consistency.get("llm_mode", "none"))
                llm_agreement = specialist_consistency.get("llm_mean_bbox_agreement")
                llm_columns[2].metric(
                    "Prediction Self-Consistency",
                    "-" if llm_agreement is None else f"{llm_agreement * 100:.1f}%",
                )
                llm_columns[3].metric("검토 필요", specialist_consistency.get("llm_review_required_images", 0))
                st.caption(
                    "1차 Vision Model 결과를 pseudo-reference로 두고, 선택한 LMM 재생성 결과와 bbox IoU 기반 self_consistency를 계산합니다."
                )
                llm_exports = output.get("llm_exports") or {}
                if llm_exports.get("enabled"):
                    st.caption(
                        f"LMM 재생성 라벨 저장 위치: `{llm_exports.get('output_root')}` / "
                        f"검토 필요 라벨 저장 위치: `{llm_exports.get('review_root')}`"
                    )
                    export_rows = []
                    for level, artifacts in (llm_exports.get("artifacts") or {}).items():
                        export_rows.append({
                            "LMM": level,
                            "저장 폴더": str(Path(llm_exports.get("output_root", "")) / level),
                            "산출물": ", ".join(sorted(artifacts.keys())) if isinstance(artifacts, dict) else "",
                        })
                    for level, artifacts in (llm_exports.get("review_artifacts") or {}).items():
                        export_rows.append({
                            "LMM": f"{level} review",
                            "저장 폴더": str(Path(llm_exports.get("review_root", "")) / level),
                            "산출물": ", ".join(sorted(artifacts.keys())) if isinstance(artifacts, dict) else "",
                        })
                    if export_rows:
                        st.dataframe(export_rows, width="stretch")
                llm_review_records = specialist_consistency.get("llm_review_records") or []
                if llm_review_records:
                    st.markdown("**LMM 기준 검토 대상 이미지**")
                    review_rows = []
                    for record in llm_review_records:
                        agreement = record.get("mean_bbox_agreement")
                        threshold = record.get("threshold")
                        review_rows.append({
                            "파일": record.get("image", ""),
                            "LMM 모드": record.get("mode", ""),
                            "Prediction Self-Consistency": "-" if agreement is None else f"{agreement * 100:.1f}%",
                            "임계치": "-" if threshold is None else f"{threshold * 100:.1f}%",
                        })
                    st.dataframe(review_rows, width="stretch")
                    review_files = [row["파일"] for row in review_rows if row.get("파일")]
                    context_options = generation_editor_context_options(output)
                    context_labels = [label for label, _ in context_options]
                    selected_context_label = st.selectbox(
                        "라벨 편집에 사용할 라벨 기준",
                        context_labels,
                        key=f"{key_prefix}-{index}-llm-review-editor-source",
                        help="같은 이미지 큐를 1차 Vision 라벨 또는 LMM 재생성 라벨 중 어떤 산출물 기준으로 편집할지 선택합니다.",
                    )
                    selected_context = dict(context_options[context_labels.index(selected_context_label)][1])
                    render_issue_editor_launcher(
                        review_files,
                        selected_context,
                        f"{key_prefix}-{index}-llm-review",
                        title="LMM 임계치 미달 이미지만 라벨 편집",
                        filter_to_issue_queue=True,
                    )
                elif specialist_consistency.get("llm_review_required_images", 0) == 0:
                    st.success("LMM 비교 기준으로 임계치 미달 이미지는 없습니다.")
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
                detailed_rows = [
                    {
                        "파일": record.get("image", ""),
                        "상태": record.get("status", ""),
                        "이슈 수": record.get("total_issues", 0),
                        "우선 조치": " / ".join(record.get("priority_actions", [])),
                    }
                    for record in detailed
                ]
                st.dataframe(detailed_rows, width="stretch")
                render_issue_editor_launcher(
                    [row["파일"] for row in detailed_rows],
                    workflow_editor_context(output),
                    f"{key_prefix}-{index}",
                )

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


def rows_from_table(value):
    if value is None:
        return []
    if hasattr(value, "to_dict"):
        return value.to_dict("records")
    return list(value)


def to_float(value, default=0.0):
    try:
        if value in ("", None):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def to_int(value, default=0):
    try:
        if value in ("", None):
            return default
        return int(float(value))
    except (TypeError, ValueError):
        return default


def clamp_unit(value):
    return max(0.0, min(1.0, to_float(value)))


def safe_state_key(value):
    return "".join(ch if ch.isalnum() else "_" for ch in str(value))[:80]


def normalize_segment_rows(rows):
    grouped = {}
    for row in rows_from_table(rows):
        label = str(row.get("label", "")).strip()
        if not label:
            continue
        segment_id = to_int(row.get("segment_id"), 0)
        grouped.setdefault(segment_id, []).append({
            "segment_id": segment_id,
            "point_id": to_int(row.get("point_id"), len(grouped.get(segment_id, []))),
            "label": label,
            "x": clamp_unit(row.get("x")),
            "y": clamp_unit(row.get("y")),
            "confidence": clamp_unit(row.get("confidence", 1.0)),
        })
    normalized = []
    for segment_id, points in sorted(grouped.items()):
        for point_id, row in enumerate(sorted(points, key=lambda item: item["point_id"])):
            row = dict(row)
            row["point_id"] = point_id
            normalized.append(row)
    return normalized


def merge_segment_rows(all_rows, segment_id, selected_rows):
    merged = [row for row in all_rows if to_int(row.get("segment_id"), 0) != segment_id]
    merged.extend(selected_rows)
    return normalize_segment_rows(merged)


def render_segmentation_point_editor(segment_rows, selected_image, image_path, image_exists):
    state_key = f"editor_segments_{safe_state_key(selected_image)}"
    source_key = f"{state_key}_source"
    source_signature = json.dumps(normalize_segment_rows(segment_rows), ensure_ascii=False, sort_keys=True)
    if st.session_state.get(source_key) != source_signature:
        st.session_state[state_key] = normalize_segment_rows(segment_rows)
        st.session_state[source_key] = source_signature

    rows = normalize_segment_rows(st.session_state.get(state_key, []))
    if not rows:
        st.info("편집할 polygon point가 없습니다. Canvas에서 polygon을 추가하거나 아래 표에 point를 직접 추가하세요.")
        edited = st.data_editor(rows, num_rows="dynamic", key=f"edit_segments_empty_{safe_state_key(selected_image)}")
        return normalize_segment_rows(edited)

    st.markdown("**Segmentation Point 정밀 편집**")
    st.caption("Canvas는 polygon 전체 이동/대략 편집에 사용하고, point 단위 좌표/순서 수정은 이 표에서 처리하세요.")
    segment_ids = sorted({to_int(row.get("segment_id"), 0) for row in rows})
    labels = {
        segment_id: next((row.get("label", "") for row in rows if to_int(row.get("segment_id"), 0) == segment_id), "")
        for segment_id in segment_ids
    }
    selected_segment = st.selectbox(
        "Polygon 선택",
        segment_ids,
        format_func=lambda value: f"segment_id={value} / label={labels.get(value, '')}",
        key=f"segment_select_{safe_state_key(selected_image)}",
    )
    width, height = image_size_or_default(image_path) if image_exists else (1, 1)
    coordinate_mode = st.radio(
        "좌표 단위",
        ["normalized", "pixel"],
        horizontal=True,
        key=f"segment_coord_mode_{safe_state_key(selected_image)}",
        help="pixel 좌표는 현재 이미지 크기를 기준으로 저장 시 normalized 좌표로 변환됩니다.",
    )
    selected_rows = normalize_segment_rows([
        row for row in rows if to_int(row.get("segment_id"), 0) == selected_segment
    ])
    point_options = [to_int(row.get("point_id"), 0) for row in selected_rows]
    selected_point = st.selectbox(
        "Point 선택",
        point_options,
        key=f"segment_point_select_{safe_state_key(selected_image)}",
    )
    point_index = point_options.index(selected_point) if selected_point in point_options else 0
    op_cols = st.columns(4)
    if op_cols[0].button("Point 추가", key=f"segment_point_add_{safe_state_key(selected_image)}"):
        current = selected_rows[point_index]
        next_row = selected_rows[(point_index + 1) % len(selected_rows)]
        new_row = {
            "segment_id": selected_segment,
            "point_id": point_index + 1,
            "label": current.get("label", "object"),
            "x": (to_float(current.get("x")) + to_float(next_row.get("x"))) / 2,
            "y": (to_float(current.get("y")) + to_float(next_row.get("y"))) / 2,
            "confidence": current.get("confidence", 1.0),
        }
        selected_rows.insert(point_index + 1, new_row)
        st.session_state[state_key] = merge_segment_rows(rows, selected_segment, selected_rows)
        st.rerun()
    if op_cols[1].button("Point 삭제", disabled=len(selected_rows) <= 3, key=f"segment_point_delete_{safe_state_key(selected_image)}"):
        selected_rows.pop(point_index)
        st.session_state[state_key] = merge_segment_rows(rows, selected_segment, selected_rows)
        st.rerun()
    if op_cols[2].button("순서 위로", disabled=point_index <= 0, key=f"segment_point_up_{safe_state_key(selected_image)}"):
        selected_rows[point_index - 1], selected_rows[point_index] = selected_rows[point_index], selected_rows[point_index - 1]
        st.session_state[state_key] = merge_segment_rows(rows, selected_segment, selected_rows)
        st.rerun()
    if op_cols[3].button("순서 아래로", disabled=point_index >= len(selected_rows) - 1, key=f"segment_point_down_{safe_state_key(selected_image)}"):
        selected_rows[point_index + 1], selected_rows[point_index] = selected_rows[point_index], selected_rows[point_index + 1]
        st.session_state[state_key] = merge_segment_rows(rows, selected_segment, selected_rows)
        st.rerun()

    if coordinate_mode == "pixel":
        display_rows = [
            {
                "segment_id": row["segment_id"],
                "point_id": row["point_id"],
                "label": row["label"],
                "x_px": round(row["x"] * width, 2),
                "y_px": round(row["y"] * height, 2),
                "confidence": row["confidence"],
            }
            for row in selected_rows
        ]
        edited_display = st.data_editor(
            display_rows,
            num_rows="dynamic",
            disabled=["segment_id", "point_id"],
            key=f"edit_segments_pixel_{safe_state_key(selected_image)}",
        )
        edited_selected = [
            {
                "segment_id": selected_segment,
                "point_id": to_int(row.get("point_id"), index),
                "label": str(row.get("label", "")).strip(),
                "x": clamp_unit(to_float(row.get("x_px")) / max(width, 1)),
                "y": clamp_unit(to_float(row.get("y_px")) / max(height, 1)),
                "confidence": clamp_unit(row.get("confidence", 1.0)),
            }
            for index, row in enumerate(rows_from_table(edited_display))
        ]
    else:
        edited_selected = st.data_editor(
            selected_rows,
            num_rows="dynamic",
            disabled=["segment_id", "point_id"],
            key=f"edit_segments_norm_{safe_state_key(selected_image)}",
        )

    return merge_segment_rows(rows, selected_segment, edited_selected)


def result_to_editor_tables(result):
    result = DetectionResult.model_validate(result)
    return {
        "classification": [{"label": item.label, "confidence": item.confidence} for item in result.classifications],
        "boxes": [
            {
                "label": box.label,
                "xmin": box.xmin,
                "ymin": box.ymin,
                "xmax": box.xmax,
                "ymax": box.ymax,
                "confidence": box.confidence,
            }
            for box in result.boxes
        ],
        "segments": [
            {
                "segment_id": segment_index,
                "point_id": point_index,
                "label": segment.label,
                "x": point.x,
                "y": point.y,
                "confidence": segment.confidence,
            }
            for segment_index, segment in enumerate(result.segments)
            for point_index, point in enumerate(segment.polygon)
        ],
        "poses": [
            {
                "pose_id": pose_index,
                "label": pose.label,
                "keypoint": point.name,
                "x": point.x,
                "y": point.y,
                "visible": point.visible,
                "confidence": point.confidence,
                "pose_confidence": pose.confidence,
            }
            for pose_index, pose in enumerate(result.poses)
            for point in pose.keypoints
        ],
        "texts": [
            {
                "text": item.text,
                "xmin": item.xmin,
                "ymin": item.ymin,
                "xmax": item.xmax,
                "ymax": item.ymax,
                "confidence": item.confidence,
            }
            for item in result.texts
        ],
        "tracks": [
            {
                "track_id": item.track_id,
                "frame_id": item.frame_id,
                "label": item.label,
                "xmin": item.xmin,
                "ymin": item.ymin,
                "xmax": item.xmax,
                "ymax": item.ymax,
                "confidence": item.confidence,
            }
            for item in result.tracks
        ],
    }


def editor_tables_to_result(task_type, tables):
    result = DetectionResult(task_type=task_type)
    for row in rows_from_table(tables.get("classification")):
        label = str(row.get("label", "")).strip()
        if label:
            result.classifications.append(ClassificationLabel(label=label, confidence=clamp_unit(row.get("confidence", 1.0))))
    for row in rows_from_table(tables.get("boxes")):
        label = str(row.get("label", "")).strip()
        if label:
            result.boxes.append(BoundingBox(
                label=label,
                xmin=clamp_unit(row.get("xmin")),
                ymin=clamp_unit(row.get("ymin")),
                xmax=clamp_unit(row.get("xmax")),
                ymax=clamp_unit(row.get("ymax")),
                confidence=clamp_unit(row.get("confidence", 1.0)),
            ))
    segments = {}
    for row in rows_from_table(tables.get("segments")):
        label = str(row.get("label", "")).strip()
        if not label:
            continue
        segment_id = to_int(row.get("segment_id"), 0)
        item = segments.setdefault(segment_id, {"label": label, "confidence": clamp_unit(row.get("confidence", 1.0)), "points": []})
        item["points"].append((to_int(row.get("point_id"), len(item["points"])), Point(x=clamp_unit(row.get("x")), y=clamp_unit(row.get("y")))))
    for item in segments.values():
        points = [point for _, point in sorted(item["points"], key=lambda pair: pair[0])]
        if len(points) >= 3:
            result.segments.append(PolygonSegment(label=item["label"], polygon=points, confidence=item["confidence"]))
    poses = {}
    for row in rows_from_table(tables.get("poses")):
        keypoint = str(row.get("keypoint", "")).strip()
        if not keypoint:
            continue
        pose_id = to_int(row.get("pose_id"), 0)
        item = poses.setdefault(pose_id, {
            "label": str(row.get("label", "person")).strip() or "person",
            "confidence": clamp_unit(row.get("pose_confidence", 1.0)),
            "keypoints": [],
        })
        item["keypoints"].append(Keypoint(
            name=keypoint,
            x=clamp_unit(row.get("x")),
            y=clamp_unit(row.get("y")),
            visible=bool(row.get("visible", True)),
            confidence=clamp_unit(row.get("confidence", 1.0)),
        ))
    for item in poses.values():
        if item["keypoints"]:
            result.poses.append(PoseInstance(label=item["label"], keypoints=item["keypoints"], confidence=item["confidence"]))
    for row in rows_from_table(tables.get("texts")):
        text = str(row.get("text", "")).strip()
        if text:
            result.texts.append(TextRegion(
                text=text,
                xmin=clamp_unit(row.get("xmin")),
                ymin=clamp_unit(row.get("ymin")),
                xmax=clamp_unit(row.get("xmax")),
                ymax=clamp_unit(row.get("ymax")),
                confidence=clamp_unit(row.get("confidence", 1.0)),
            ))
    for row in rows_from_table(tables.get("tracks")):
        track_id = str(row.get("track_id", "")).strip()
        label = str(row.get("label", "")).strip()
        if track_id and label:
            result.tracks.append(TrackInstance(
                track_id=track_id,
                frame_id=to_int(row.get("frame_id"), 0),
                label=label,
                xmin=clamp_unit(row.get("xmin")),
                ymin=clamp_unit(row.get("ymin")),
                xmax=clamp_unit(row.get("xmax")),
                ymax=clamp_unit(row.get("ymax")),
                confidence=clamp_unit(row.get("confidence", 1.0)),
            ))
    return result


def load_editor_records(label_path, image_dir, source_format, classes_path=None):
    batch = import_labels_with_report(label_path, image_dir, source_format=source_format, classes_path=classes_path)
    return [{"image": image, "result": result.model_dump()} for image, result in batch.records], batch.report


def image_size_or_default(image_path):
    try:
        with Image.open(image_path) as image:
            return image.size
    except Exception:
        return 1, 1


def fit_canvas_size(image_path, max_width=720):
    try:
        with Image.open(image_path) as image:
            width, height = image.size
    except Exception:
        return 1, 1, 1, 1
    if width <= max_width:
        return width, height, width, height
    ratio = max_width / width
    return width, height, int(width * ratio), int(height * ratio)


def rect_canvas_object(label, xmin, ymin, xmax, ymax, width, height, stroke="#ef4444", editor_type="bbox", extra=None):
    left = clamp_unit(xmin) * width
    top = clamp_unit(ymin) * height
    rect_width = max(1.0, (clamp_unit(xmax) - clamp_unit(xmin)) * width)
    rect_height = max(1.0, (clamp_unit(ymax) - clamp_unit(ymin)) * height)
    payload = {
        "type": "rect",
        "left": left,
        "top": top,
        "width": rect_width,
        "height": rect_height,
        "scaleX": 1,
        "scaleY": 1,
        "fill": "rgba(239, 68, 68, 0.12)",
        "stroke": stroke,
        "strokeWidth": 2,
        "label": label,
        "editor_type": editor_type,
    }
    if extra:
        payload.update(extra)
    return payload


def polygon_canvas_object(segment, width, height):
    points = [{"x": point.x * width, "y": point.y * height} for point in segment.polygon]
    return {
        "type": "polygon",
        "points": points,
        "left": 0,
        "top": 0,
        "fill": "rgba(37, 99, 235, 0.15)",
        "stroke": "#2563eb",
        "strokeWidth": 2,
        "label": segment.label,
        "confidence": segment.confidence,
        "editor_type": "segmentation",
    }


def canvas_objects_from_result(result, task_type, width, height):
    result = DetectionResult.model_validate(result)
    objects = []
    if task_type == "object_detection" or (task_type == "segmentation" and not result.segments):
        objects.extend(
            rect_canvas_object(box.label, box.xmin, box.ymin, box.xmax, box.ymax, width, height, extra={"confidence": box.confidence})
            for box in result.boxes
        )
    if task_type == "ocr":
        objects.extend(
            rect_canvas_object(item.text, item.xmin, item.ymin, item.xmax, item.ymax, width, height, stroke="#16a34a", editor_type="ocr", extra={"text": item.text, "confidence": item.confidence})
            for item in result.texts
        )
    if task_type == "tracking":
        objects.extend(
            rect_canvas_object(item.label, item.xmin, item.ymin, item.xmax, item.ymax, width, height, stroke="#f97316", editor_type="tracking", extra={"track_id": item.track_id, "frame_id": item.frame_id, "confidence": item.confidence})
            for item in result.tracks
        )
    if task_type == "segmentation":
        objects.extend(polygon_canvas_object(segment, width, height) for segment in result.segments)
    return {"version": "4.4.0", "objects": objects}


def object_rect_bounds(obj, width, height):
    left = to_float(obj.get("left"))
    top = to_float(obj.get("top"))
    rect_width = to_float(obj.get("width"), 1.0) * to_float(obj.get("scaleX"), 1.0)
    rect_height = to_float(obj.get("height"), 1.0) * to_float(obj.get("scaleY"), 1.0)
    xmin, xmax = sorted([left / max(width, 1), (left + rect_width) / max(width, 1)])
    ymin, ymax = sorted([top / max(height, 1), (top + rect_height) / max(height, 1)])
    return clamp_unit(xmin), clamp_unit(ymin), clamp_unit(xmax), clamp_unit(ymax)


def object_polygon_points(obj, width, height):
    left = to_float(obj.get("left"))
    top = to_float(obj.get("top"))
    scale_x = to_float(obj.get("scaleX"), 1.0)
    scale_y = to_float(obj.get("scaleY"), 1.0)
    path_offset = obj.get("pathOffset") or {}
    offset_x = to_float(path_offset.get("x"), 0.0) if isinstance(path_offset, dict) else 0.0
    offset_y = to_float(path_offset.get("y"), 0.0) if isinstance(path_offset, dict) else 0.0
    points = obj.get("points") or []
    if not points and obj.get("path"):
        points = []
        for command in obj.get("path") or []:
            if not isinstance(command, (list, tuple)) or len(command) < 3:
                continue
            command_type = str(command[0]).upper()
            if command_type not in {"M", "L"}:
                continue
            points.append({"x": command[1], "y": command[2], "_path": True})
    result = []
    for point in points:
        point_x = to_float(point.get("x"))
        point_y = to_float(point.get("y"))
        if point.get("_absolute"):
            x = point_x / max(width, 1)
            y = point_y / max(height, 1)
        elif point.get("_path"):
            x = (left + point_x * scale_x) / max(width, 1)
            y = (top + point_y * scale_y) / max(height, 1)
        elif 0 <= point_x <= 1 and 0 <= point_y <= 1:
            x = point_x
            y = point_y
        else:
            x = (left + (point_x - offset_x) * scale_x) / max(width, 1)
            y = (top + (point_y - offset_y) * scale_y) / max(height, 1)
        result.append(Point(x=clamp_unit(x), y=clamp_unit(y)))
    return result


def canvas_point_coordinate(obj, width, height):
    obj_type = str(obj.get("type", "")).lower()
    if obj_type not in {"circle", "point"}:
        return None
    left = to_float(obj.get("left"))
    top = to_float(obj.get("top"))
    radius = to_float(obj.get("radius"), 0.0)
    scale_x = to_float(obj.get("scaleX"), 1.0)
    scale_y = to_float(obj.get("scaleY"), 1.0)
    x = (left + radius * scale_x) / max(width, 1)
    y = (top + radius * scale_y) / max(height, 1)
    return Point(x=clamp_unit(x), y=clamp_unit(y))


def last_canvas_point(objects, width, height):
    for obj in reversed(objects or []):
        point = canvas_point_coordinate(obj, width, height)
        if point is not None:
            return point
    return None


def result_with_moved_segment_point(base_result, segment_index, point_index, point):
    result = DetectionResult.model_validate(base_result).model_copy(deep=True)
    if segment_index < 0 or segment_index >= len(result.segments):
        return result
    segment = result.segments[segment_index]
    if point_index < 0 or point_index >= len(segment.polygon):
        return result
    segment.polygon[point_index] = point
    return result


def segments_for_vertex_editor(result):
    result = DetectionResult.model_validate(result)
    return [
        {
            "label": segment.label,
            "confidence": segment.confidence,
            "polygon": [point.model_dump() for point in segment.polygon],
        }
        for segment in result.segments
    ]


def boxes_for_vertex_editor(result):
    result = DetectionResult.model_validate(result)
    return [
        {
            "label": box.label,
            "confidence": box.confidence,
            "xmin": box.xmin,
            "ymin": box.ymin,
            "xmax": box.xmax,
            "ymax": box.ymax,
        }
        for box in result.boxes
    ]


def result_with_vertex_editor_payload(base_result, segment_payload, box_payload):
    result = DetectionResult.model_validate(base_result).model_copy(deep=True)
    has_segment_payload = segment_payload is not None
    has_box_payload = box_payload is not None
    updated_segments = []
    if has_segment_payload:
        for item in segment_payload or []:
            label = str(item.get("label") or "object").strip()
            points = [
                Point(x=clamp_unit(point.get("x")), y=clamp_unit(point.get("y")))
                for point in item.get("polygon", [])
                if isinstance(point, dict)
            ]
            if len(points) >= 3:
                updated_segments.append(PolygonSegment(
                    label=label,
                    polygon=points,
                    confidence=clamp_unit(item.get("confidence", 1.0)),
                ))
        if updated_segments or not result.segments:
            result.segments = updated_segments
    updated_boxes = []
    if has_box_payload:
        for item in box_payload or []:
            label = str(item.get("label") or "object").strip()
            xmin = clamp_unit(item.get("xmin"))
            ymin = clamp_unit(item.get("ymin"))
            xmax = clamp_unit(item.get("xmax"))
            ymax = clamp_unit(item.get("ymax"))
            if xmin >= xmax or ymin >= ymax:
                continue
            updated_boxes.append(BoundingBox(
                label=label,
                xmin=xmin,
                ymin=ymin,
                xmax=xmax,
                ymax=ymax,
                confidence=clamp_unit(item.get("confidence", 1.0)),
            ))
        if updated_boxes or not result.boxes:
            result.boxes = updated_boxes
    return result


def result_with_canvas_objects(base_result, task_type, objects, default_label, default_text, default_track_id, width, height):
    result = DetectionResult.model_validate(base_result)
    if task_type == "object_detection":
        result.boxes = []
    elif task_type == "ocr":
        result.texts = []
    elif task_type == "tracking":
        result.tracks = []
    elif task_type == "segmentation":
        result.boxes = []
        result.segments = []

    for index, obj in enumerate(objects or []):
        obj_type = obj.get("type")
        label = str(obj.get("label") or default_label or "object").strip()
        confidence = clamp_unit(obj.get("confidence", 1.0))
        if obj_type == "rect" and task_type in {"object_detection", "segmentation"}:
            xmin, ymin, xmax, ymax = object_rect_bounds(obj, width, height)
            result.boxes.append(BoundingBox(label=label, xmin=xmin, ymin=ymin, xmax=xmax, ymax=ymax, confidence=confidence))
        elif obj_type == "rect" and task_type == "ocr":
            xmin, ymin, xmax, ymax = object_rect_bounds(obj, width, height)
            text = str(obj.get("text") or obj.get("label") or default_text or "text").strip()
            result.texts.append(TextRegion(text=text, xmin=xmin, ymin=ymin, xmax=xmax, ymax=ymax, confidence=confidence))
        elif obj_type == "rect" and task_type == "tracking":
            xmin, ymin, xmax, ymax = object_rect_bounds(obj, width, height)
            result.tracks.append(TrackInstance(
                track_id=str(obj.get("track_id") or default_track_id or f"track_{index + 1}"),
                frame_id=to_int(obj.get("frame_id"), 0),
                label=label,
                xmin=xmin,
                ymin=ymin,
                xmax=xmax,
                ymax=ymax,
                confidence=confidence,
            ))
        elif obj_type in {"polygon", "path"} and task_type == "segmentation":
            points = object_polygon_points(obj, width, height)
            if len(points) >= 3:
                result.segments.append(PolygonSegment(label=label, polygon=points, confidence=confidence))
    return result


def result_with_new_canvas_polygons(base_result, objects, default_label, width, height):
    result = DetectionResult.model_validate(base_result).model_copy(deep=True)
    polygon_objects = [
        obj for obj in objects or []
        if str(obj.get("type", "")).lower() in {"polygon", "path"}
    ]
    added = 0
    for obj in polygon_objects:
        points = object_polygon_points(obj, width, height)
        if len(points) < 3:
            continue
        label = str(obj.get("label") or default_label or "object").strip()
        confidence = clamp_unit(obj.get("confidence", 1.0))
        result.segments.append(PolygonSegment(label=label, polygon=points, confidence=confidence))
        added += 1
    return result, added


def move_editor_selection(image_names, current_image, offset):
    if not image_names:
        return None
    try:
        current_index = image_names.index(current_image)
    except ValueError:
        current_index = 0
    next_index = max(0, min(len(image_names) - 1, current_index + offset))
    return image_names[next_index]


def request_editor_selection(image_name):
    st.session_state.editor_pending_selected_image = image_name


def render_label_editor_tab(workspace):
    st.markdown('<div class="panel-title">라벨 편집</div>', unsafe_allow_html=True)
    st.caption("classification, detection, OCR, segmentation, pose-estimation, tracking 라벨을 불러와 편집하고 다시 저장합니다.")
    issue_context = st.session_state.get("editor_issue_context") or {}
    issue_queue = st.session_state.get("editor_issue_queue") or []
    if issue_context and issue_queue:
        st.info(
            f"문제 파일 {len(issue_queue)}개를 순차 편집하는 중입니다. "
            "필요하면 아래 경로와 포맷을 조정한 뒤 다시 불러오세요."
        )
    label_path_candidates = workspace_dir_candidates(workspace, {"labels", "labeled"}, WORKSPACE_DEFAULTS["labels"])
    image_dir_candidates = workspace_dir_candidates(workspace, {"raw", "images"}, WORKSPACE_DEFAULTS["images"])
    if "editor_label_path" not in st.session_state:
        st.session_state.editor_label_path = label_path_candidates[0]
    if "editor_image_dir" not in st.session_state:
        st.session_state.editor_image_dir = image_dir_candidates[0]

    with st.container(border=True):
        left, right = st.columns(2)
        with left:
            editor_label_path = editable_path_selectbox("편집할 라벨 경로", label_path_candidates, "editor_label_path", "라벨 파일 또는 라벨 폴더를 선택하세요.")
            editor_source_format = st.selectbox("라벨 형식", SOURCE_OPTIONS, key="editor_source_format")
            editor_classes = st.text_input("클래스 매핑 파일 (선택)", value="", key="editor_classes")
        with right:
            editor_image_dir = editable_path_selectbox("이미지 폴더 경로", image_dir_candidates, "editor_image_dir", "편집할 라벨과 연결되는 이미지 폴더입니다.")
            editor_task = st.selectbox("편집 태스크", ["object_detection", "classification", "ocr", "segmentation", "pose_estimation", "tracking"], key="editor_task")
            editor_output = st.text_input("저장 경로", value=WORKSPACE_DEFAULTS["labels"], key="editor_output")
        load_clicked = st.button("라벨 불러오기", type="primary", icon=":material/edit:")

    context = {
        "label_path": editor_label_path,
        "image_dir": editor_image_dir,
        "source_format": editor_source_format,
        "classes_path": editor_classes,
        "task_type": editor_task,
        "output_dir": editor_output,
    }
    context_signature = editor_context_signature(context)
    auto_load_issue_context = bool(issue_context and issue_queue) and (
        st.session_state.get("editor_loaded_context_signature") != context_signature
    )
    auto_load_editor_context = bool(st.session_state.pop("editor_auto_load", False)) and (
        st.session_state.get("editor_loaded_context_signature") != context_signature
    )
    if load_clicked or auto_load_issue_context or auto_load_editor_context:
        try:
            records, report = load_editor_records(
                resolve_workspace_path(workspace, editor_label_path),
                resolve_workspace_path(workspace, editor_image_dir),
                editor_source_format,
                classes_path=resolve_workspace_path(workspace, editor_classes) if editor_classes else None,
            )
            if issue_context.get("filter_to_issue_queue") and issue_queue:
                issue_set = set(issue_queue)
                records = [record for record in records if record.get("image") in issue_set]
            st.session_state.editor_records = records
            st.session_state.editor_report = report
            st.session_state.editor_loaded_context_signature = context_signature
            st.session_state.editor_selected_image = records[0]["image"] if records else ""
            if records:
                st.session_state.editor_selected_image_select = records[0]["image"]
            st.session_state.editor_save_formats = default_editor_save_formats(editor_source_format, report)
            if load_clicked:
                st.success(
                    f"{len(records)}개 라벨 데이터를 불러왔습니다. "
                    f"저장 포맷 기본값: {', '.join(st.session_state.editor_save_formats)}"
                )
            elif auto_load_editor_context:
                st.success(f"대화형 요청으로 {len(records)}개 라벨 데이터를 불러왔습니다.")
        except Exception as exc:
            message = f"라벨 불러오기 실패: {exc}"
            if auto_load_issue_context or auto_load_editor_context:
                st.warning(message)
            else:
                st.error(message)

    records = st.session_state.get("editor_records", [])
    if not records:
        st.info("라벨을 불러오면 이미지별 라벨을 편집할 수 있습니다.")
        return

    image_names = [record["image"] for record in records]
    if issue_queue:
        queue_index = min(st.session_state.get("editor_issue_index", 0), len(issue_queue) - 1)
        st.session_state.editor_issue_index = queue_index
        queue_columns = st.columns([1, 1, 1, 3])
        if queue_columns[0].button("이전 이슈", disabled=queue_index <= 0, key="editor_issue_prev"):
            st.session_state.editor_issue_index = max(queue_index - 1, 0)
            st.session_state.editor_selected_image_select = issue_queue[st.session_state.editor_issue_index]
            st.rerun()
        if queue_columns[1].button("다음 이슈", disabled=queue_index >= len(issue_queue) - 1, key="editor_issue_next"):
            st.session_state.editor_issue_index = min(queue_index + 1, len(issue_queue) - 1)
            st.session_state.editor_selected_image_select = issue_queue[st.session_state.editor_issue_index]
            st.rerun()
        if queue_columns[2].button("현재 파일 완료", key="editor_issue_done"):
            completed = issue_queue.pop(queue_index)
            st.session_state.editor_issue_queue = issue_queue
            if issue_queue:
                st.session_state.editor_issue_index = min(queue_index, len(issue_queue) - 1)
                st.session_state.editor_selected_image_select = issue_queue[st.session_state.editor_issue_index]
            else:
                st.session_state.editor_issue_index = 0
                st.session_state.editor_issue_context = {}
            st.success(f"{completed} 파일을 편집 큐에서 제거했습니다.")
            st.rerun()
        queue_columns[3].caption(f"{queue_index + 1} / {len(issue_queue)}: {issue_queue[queue_index]}")
        missing_from_records = [name for name in issue_queue if name not in image_names]
        if missing_from_records:
            st.warning(
                "현재 불러온 라벨에서 일부 문제 파일을 찾지 못했습니다: "
                + ", ".join(missing_from_records[:5])
                + (" ..." if len(missing_from_records) > 5 else "")
            )
        if issue_queue[queue_index] in image_names:
            st.session_state.editor_selected_image_select = issue_queue[queue_index]

    pending_selected = st.session_state.pop("editor_pending_selected_image", None)
    if pending_selected:
        st.session_state.editor_selected_image_select = pending_selected
    selected_default = st.session_state.get("editor_selected_image_select") or st.session_state.get("editor_selected_image")
    if selected_default not in image_names:
        selected_default = image_names[0]
        st.session_state.editor_selected_image_select = selected_default
    selected_image = st.selectbox(
        "편집 이미지",
        image_names,
        index=image_names.index(selected_default),
        key="editor_selected_image_select",
    )
    record_index = image_names.index(selected_image)
    nav_prev_col, nav_status_col, nav_next_col = st.columns([1, 2, 1])
    if nav_prev_col.button("이전 이미지", disabled=record_index <= 0, key="editor_prev_image"):
        request_editor_selection(move_editor_selection(image_names, selected_image, -1))
        st.rerun()
    nav_status_col.caption(f"{record_index + 1} / {len(image_names)}")
    if nav_next_col.button("다음 이미지", disabled=record_index >= len(image_names) - 1, key="editor_next_image"):
        request_editor_selection(move_editor_selection(image_names, selected_image, 1))
        st.rerun()
    record = records[record_index]
    result = DetectionResult.model_validate(record["result"])
    image_path = find_image_path(resolve_workspace_path(workspace, editor_image_dir), selected_image)
    image_exists = Path(image_path).is_file()

    preview_col, edit_col = st.columns([1, 1])
    with preview_col:
        st.markdown("**이미지 / 마우스 편집**")
        if image_exists:
            if editor_task == "segmentation":
                original_width, original_height, canvas_width, canvas_height = fit_canvas_size(image_path)
                st.caption("기본 editor에서는 polygon vertex와 bbox를 편집합니다. 새 polygon은 이전 Canvas 생성 방식으로 일시 전환해 만들 수 있습니다.")
                undo_key = f"polygon_vertex_editor_undo_{selected_image}"
                polygon_create_key = f"polygon_create_mode_{safe_state_key(selected_image)}"
                editor_version_key = f"polygon_vertex_editor_version_{safe_state_key(selected_image)}"
                undo_stack = st.session_state.get(undo_key, [])
                undo_col, create_col, info_col = st.columns([1, 1, 3])
                if undo_col.button("되돌리기", disabled=not undo_stack, key=f"undo_polygon_vertex_{safe_state_key(selected_image)}"):
                    previous = undo_stack.pop()
                    record["result"] = previous
                    records[record_index] = record
                    st.session_state.editor_records = records
                    st.session_state[undo_key] = undo_stack
                    st.rerun()
                if create_col.button("Polygon 생성", key=f"open_polygon_create_{safe_state_key(selected_image)}"):
                    st.session_state[polygon_create_key] = True
                    st.session_state[editor_version_key] = st.session_state.get(editor_version_key, 0) + 1
                    st.rerun()
                info_col.caption("되돌리기는 vertex 이동, polygon/bbox 생성 직전 상태로 복원합니다.")
                if st.session_state.get(polygon_create_key):
                    st.info("이전 Canvas polygon 생성 방식입니다. polygon을 그린 뒤 아래 적용 버튼을 누르면 통합 editor로 돌아갑니다.")
                    cancel_col, apply_col = st.columns(2)
                    if cancel_col.button("생성 취소", key=f"cancel_polygon_create_{safe_state_key(selected_image)}"):
                        st.session_state[polygon_create_key] = False
                        st.rerun()
                    if not st_canvas:
                        st.warning("이전 polygon 생성 방식을 사용하려면 `streamlit-drawable-canvas`가 필요합니다.")
                    else:
                        canvas_label = st.text_input("새 Polygon 라벨", value="object", key=f"polygon_canvas_label_{safe_state_key(selected_image)}")
                        st.caption("polygon을 모두 찍은 뒤 더블클릭으로 마무리하거나 Canvas 상단의 전송 버튼을 누른 다음 `Polygon 생성 적용`을 누르세요.")
                        with Image.open(image_path) as bg_image:
                            bg_image = bg_image.convert("RGB").resize((canvas_width, canvas_height))
                            canvas_result = st_canvas(
                                fill_color="rgba(37, 99, 235, 0.15)",
                                stroke_width=2,
                                stroke_color="#2563eb",
                                background_image=bg_image,
                                update_streamlit=True,
                                height=canvas_height,
                                width=canvas_width,
                                drawing_mode="polygon",
                                initial_drawing={"version": "4.4.0", "objects": []},
                                key=f"polygon_create_canvas_{selected_image}_{st.session_state.get(editor_version_key, 0)}",
                            )
                        if apply_col.button("Polygon 생성 적용", key=f"apply_polygon_create_{safe_state_key(selected_image)}"):
                            objects = []
                            if canvas_result and canvas_result.json_data:
                                objects = canvas_result.json_data.get("objects", [])
                            updated, added_count = result_with_new_canvas_polygons(
                                result,
                                objects,
                                canvas_label,
                                canvas_width,
                                canvas_height,
                            )
                            if added_count < 1:
                                st.warning("생성된 polygon을 찾지 못했습니다. Canvas에서 polygon을 그린 뒤 다시 적용하세요.")
                                st.stop()
                            st.session_state.setdefault(undo_key, []).append(result.model_dump())
                            record["result"] = updated.model_dump()
                            records[record_index] = record
                            st.session_state.editor_records = records
                            st.session_state[polygon_create_key] = False
                            st.session_state[editor_version_key] = st.session_state.get(editor_version_key, 0) + 1
                            st.success("Polygon을 추가하고 통합 editor로 돌아갑니다.")
                            st.rerun()
                else:
                    editor_value = polygon_vertex_editor(
                        image_path=image_path,
                        segments=segments_for_vertex_editor(result),
                        boxes=boxes_for_vertex_editor(result),
                        width=canvas_width,
                        height=canvas_height,
                        key=f"polygon_vertex_editor_{selected_image}_{len(result.segments)}_{st.session_state.get(editor_version_key, 0)}",
                    )
                    if editor_value and (editor_value.get("segments") is not None or editor_value.get("boxes") is not None):
                        update_key = f"polygon_vertex_editor_update_{selected_image}"
                        updated_at = editor_value.get("updatedAt")
                        action = editor_value.get("action")
                        if action in {"move_vertex", "add_bbox", "add_polygon"} and updated_at and st.session_state.get(update_key) != updated_at:
                            st.session_state.setdefault(undo_key, []).append(result.model_dump())
                            updated = result_with_vertex_editor_payload(
                                result,
                                editor_value.get("segments", []),
                                editor_value.get("boxes", []),
                            )
                            record["result"] = updated.model_dump()
                            records[record_index] = record
                            st.session_state.editor_records = records
                            st.session_state[update_key] = updated_at
                            st.rerun()
            elif st_canvas:
                original_width, original_height, canvas_width, canvas_height = fit_canvas_size(image_path)
                canvas_label = st.text_input("새 도형 기본 라벨", value="object", key="editor_canvas_label")
                canvas_text = st.text_input("새 OCR 텍스트", value="text", key="editor_canvas_text")
                canvas_track = st.text_input("새 Track ID", value="track_1", key="editor_canvas_track")
                if editor_task in {"object_detection", "ocr", "tracking"}:
                    mode_options = ["transform", "rect"]
                    mode_help = "rect로 새 bbox를 그리고, transform으로 기존 bbox를 통째로 이동/리사이즈하세요."
                elif editor_task == "pose_estimation":
                    mode_options = ["transform", "point"]
                    mode_help = "pose는 표 편집이 기본이며, point 도형은 참고용으로 사용할 수 있습니다."
                else:
                    mode_options = ["transform"]
                    mode_help = "classification은 이미지 단위 라벨이므로 표에서 수정하세요."
                drawing_mode = st.selectbox("Canvas 모드", mode_options, help=mode_help, key="editor_canvas_mode")
                with Image.open(image_path) as bg_image:
                    bg_image = bg_image.convert("RGB").resize((canvas_width, canvas_height))
                    canvas_result = st_canvas(
                        fill_color="rgba(239, 68, 68, 0.12)",
                        stroke_width=2,
                        stroke_color="#ef4444",
                        background_image=bg_image,
                        update_streamlit=True,
                        height=canvas_height,
                        width=canvas_width,
                        drawing_mode=drawing_mode,
                        initial_drawing=canvas_objects_from_result(result, editor_task, canvas_width, canvas_height),
                        key=f"canvas_{selected_image}_{editor_task}",
                    )
                st.caption(
                    "도형을 선택해 통째로 이동하거나 크기를 조절할 수 있습니다. "
                    "새 bbox를 그린 뒤 아래 버튼을 눌러 라벨에 반영하세요."
                )
                if st.button("Canvas 도형을 현재 라벨에 적용", icon=":material/select_check_box:"):
                    objects = []
                    if canvas_result and canvas_result.json_data:
                        objects = canvas_result.json_data.get("objects", [])
                    updated = result_with_canvas_objects(
                        result,
                        editor_task,
                        objects,
                        canvas_label,
                        canvas_text,
                        canvas_track,
                        canvas_width,
                        canvas_height,
                    )
                    record["result"] = updated.model_dump()
                    records[record_index] = record
                    st.session_state.editor_records = records
                    st.success("Canvas 도형을 현재 이미지 라벨에 반영했습니다.")
                    st.rerun()
            else:
                st.image(image_path, caption=selected_image, use_container_width=True)
                st.warning("마우스 도형 편집을 사용하려면 `pip install streamlit-drawable-canvas`가 필요합니다. 현재는 표 편집만 사용할 수 있습니다.")
        else:
            st.warning("연결된 이미지 파일을 찾지 못했습니다. 표 편집은 계속 사용할 수 있습니다.")

    with edit_col:
        st.markdown("**라벨 테이블 편집**")
        tables = result_to_editor_tables(result)
        edited_tables = {}
        if editor_task == "classification":
            edited_tables["classification"] = st.data_editor(tables["classification"], num_rows="dynamic", key=f"edit_cls_{selected_image}")
        elif editor_task == "ocr":
            edited_tables["texts"] = st.data_editor(tables["texts"], num_rows="dynamic", key=f"edit_texts_{selected_image}")
        elif editor_task == "segmentation":
            st.caption("segmentation은 Canvas에서 polygon을 직접 추가/이동하고, 좌표를 수치로 미세 조정해야 할 때만 아래 정밀 편집을 열어 사용하세요.")
            with st.expander("Polygon point 정밀 편집 열기", expanded=False):
                edited_tables["segments"] = render_segmentation_point_editor(
                    tables["segments"],
                    selected_image,
                    image_path,
                    image_exists,
                )
            if "segments" not in edited_tables:
                edited_tables["segments"] = tables["segments"]
        elif editor_task == "pose_estimation":
            edited_tables["poses"] = st.data_editor(tables["poses"], num_rows="dynamic", key=f"edit_poses_{selected_image}")
        elif editor_task == "tracking":
            edited_tables["tracks"] = st.data_editor(tables["tracks"], num_rows="dynamic", key=f"edit_tracks_{selected_image}")
        else:
            edited_tables["boxes"] = st.data_editor(tables["boxes"], num_rows="dynamic", key=f"edit_boxes_{selected_image}")
        apply_col, apply_next_col = st.columns(2)
        apply_current = apply_col.button("현재 이미지 라벨 적용", icon=":material/save:")
        apply_and_next = apply_next_col.button(
            "적용 후 다음 이미지",
            icon=":material/arrow_forward:",
            disabled=record_index >= len(image_names) - 1,
        )
        if apply_current or apply_and_next:
            merged_tables = result_to_editor_tables(result)
            merged_tables.update(edited_tables)
            record["result"] = editor_tables_to_result(editor_task, merged_tables).model_dump()
            records[record_index] = record
            st.session_state.editor_records = records
            if apply_and_next:
                request_editor_selection(move_editor_selection(image_names, selected_image, 1))
                st.success("현재 이미지 라벨을 반영하고 다음 이미지로 이동합니다.")
                st.rerun()
            st.success("현재 이미지 라벨을 편집 상태에 반영했습니다.")

    with st.container(border=True):
        st.markdown("**전체 라벨 파일 저장**")
        if "editor_save_formats" not in st.session_state:
            st.session_state.editor_save_formats = default_editor_save_formats(editor_source_format, st.session_state.get("editor_report", {}))
        st.caption("현재 편집 세션에 불러온 모든 이미지 라벨을 파일로 저장합니다. 이미지 이동은 위 `이전/다음 이미지` 버튼을 사용하세요.")
        save_formats = st.multiselect("저장 포맷", FORMAT_OPTIONS, key="editor_save_formats")
        if st.button("전체 편집 라벨 저장", type="primary", icon=":material/save_as:"):
            if not save_formats:
                st.error("저장 포맷을 하나 이상 선택하세요.")
            else:
                try:
                    output_dir = resolve_workspace_path(workspace, editor_output)
                    writer = LabelExportWriter(output_dir, formats=save_formats)
                    image_dir_abs = resolve_workspace_path(workspace, editor_image_dir)
                    for item in records:
                        image_path = find_image_path(image_dir_abs, item["image"])
                        writer.save(DetectionResult.model_validate(item["result"]), image_path, formats=save_formats)
                    artifacts = writer.finalize()
                    if "manual_results" not in st.session_state:
                        st.session_state.manual_results = {}
                    st.session_state.manual_results["label_editor"] = {
                        "action": "label_editor",
                        "images": len(records),
                        "formats": save_formats,
                        "artifacts": artifacts,
                        "output_dir": output_dir,
                    }
                    st.success(f"불러온 전체 {len(records)}개 이미지의 편집 라벨을 저장했습니다: {output_dir}")
                except Exception as exc:
                    st.error(f"라벨 저장 실패: {exc}")


def build_convert_preflight_preview(
    input_path,
    image_dir,
    source_format,
    target_formats,
    classes_path=None,
    duplicate_iou=0.85,
    custom_mapping_spec=None,
):
    batch = import_labels_with_report(
        input_path,
        image_dir,
        source_format=source_format,
        classes_path=classes_path,
        duplicate_iou=duplicate_iou,
        custom_mapping_spec=custom_mapping_spec,
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
        "editor_context": {
            "label_path": input_path,
            "image_dir": image_dir,
            "source_format": source_format,
            "classes_path": classes_path or "",
            "task_type": "object_detection",
            "output_dir": WORKSPACE_DEFAULTS["labels"],
            "custom_label_mapping": custom_mapping_spec or "",
        },
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
        render_issue_editor_launcher(
            [row["파일"] for row in detailed],
            preview.get("editor_context", {}),
            "convert-preflight",
        )


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
        custom_mapping_spec=operation.get("custom_label_mapping"),
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


def chat_rerun_plan(first_pass_plan, llm_mode):
    plan = copy.deepcopy(first_pass_plan)
    for operation in plan.get("operations", []):
        if operation.get("action") == "generate":
            operation["specialist_consistency_runs"] = 0
            operation["specialist_advisor_mode"] = "none"
            operation["llm_consistency_mode"] = llm_mode
    return plan


def render_selective_rerun_controls(first_pass_plan, key_prefix, on_complete):
    if not first_pass_plan or not has_generate_operation(first_pass_plan):
        return
    st.divider()
    st.markdown("#### 1차 추론 이후 LMM 재생성 비교")
    st.caption("1차 Vision Model 결과를 pseudo-reference로 두고, 선택한 LMM 재생성 결과와 bbox IoU 기반 self_consistency를 계산합니다.")
    llm_mode = st.selectbox(
        "재생성 LMM",
        ["low", "high", "both"],
        index=0,
        key=f"{key_prefix}_llm_consistency_mode",
        help="선택한 Low/High LMM이 같은 이미지에 대해 라벨을 다시 생성하고, 1차 Vision 결과와 prediction-to-prediction self_consistency를 계산합니다.",
    )
    if st.button("LMM 재생성 비교 실행", type="secondary", key=f"{key_prefix}_llm_consistency_rerun"):
        try:
            result = execute_plan(chat_rerun_plan(first_pass_plan, llm_mode), auto_approve=True)
            on_complete(result)
        except Exception as exc:
            st.error(f"LMM 재생성 비교 실행 중 오류가 발생했습니다: {exc}")
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
    if "pending_model_dataset_request" not in st.session_state:
        st.session_state.pending_model_dataset_request = None

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
        raw_chat_request = chat_request
        if st.session_state.get("pending_model_dataset_request") and not st.session_state.pending_proposal:
            chat_request = st.session_state.pending_model_dataset_request["request"] + "\n" + chat_request
        st.session_state.chat_messages.append({"role": "user", "content": raw_chat_request})
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
                    st.session_state.pending_model_dataset_request = None
                    proposal = first_pass_chat_proposal(routed["proposal"])
                    response = describe_plan(proposal, workspace)
                    st.session_state.pending_proposal = proposal
                    st.session_state.pending_preflight_preview = render_chat_preflight_for_proposal(proposal)
                elif routed["kind"] == "open_editor":
                    st.session_state.pending_model_dataset_request = None
                    response = routed["response"]
                    start_label_editor_context(workspace, routed.get("editor_context") or {})
                else:
                    response = routed["response"]
                    if routed.get("diagnosis"):
                        st.session_state.pending_model_dataset_request = {
                            "request": chat_request,
                            "diagnosis": routed["diagnosis"],
                        }
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
chat_tab, convert_tab, generate_tab, editor_tab, result_tab, settings_tab = st.tabs(
    ["대화형 작업", "형식 변환", "라벨 생성", "라벨 편집", "결과 리포트", "설정"]
)

if st.session_state.pop("open_result_report", False):
    st.html(
        """
        <script>
        const labels = Array.from(window.parent.document.querySelectorAll('[role="tab"]'));
        const target = labels.find((el) => el.textContent.includes('결과 리포트'));
        if (target) target.click();
        </script>
        """,
        unsafe_allow_javascript=True,
    )

if st.session_state.pop("open_label_editor", False):
    st.html(
        """
        <script>
        const labels = Array.from(window.parent.document.querySelectorAll('[role="tab"]'));
        const target = labels.find((el) => el.textContent.includes('라벨 편집'));
        if (target) target.click();
        </script>
        """,
        unsafe_allow_javascript=True,
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
    if "convert_input_path" not in st.session_state:
        st.session_state.convert_input_path = label_path_candidates[0]
    if "convert_image_dir" not in st.session_state:
        st.session_state.convert_image_dir = image_dir_candidates[0]
    if "convert_custom_mapping_text" not in st.session_state:
        st.session_state.convert_custom_mapping_text = ""
    pending_custom_mapping_text = st.session_state.pop("convert_custom_mapping_text_pending", None)
    if pending_custom_mapping_text is not None:
        st.session_state.convert_custom_mapping_text = pending_custom_mapping_text
    with st.container(border=True):
        st.markdown('<span class="path-panel-marker"></span>', unsafe_allow_html=True)
        st.markdown('<div class="form-section">데이터 경로</div>', unsafe_allow_html=True)
        path_left, path_right = st.columns(2)
        with path_left:
            input_path = editable_path_selectbox(
                "라벨 폴더 경로",
                label_path_candidates,
                "convert_input_path",
                (
                    "추천 경로를 선택하거나 같은 칸에 직접 입력하세요. "
                    "라벨 파일이 들어 있는 폴더만 지정하세요. "
                    "결과 폴더(converted/reports/visualized)가 섞이면 불필요한 파일도 검사될 수 있습니다."
                ),
            )
        with path_right:
            image_dir = editable_path_selectbox(
                "이미지 폴더 경로",
                image_dir_candidates,
                "convert_image_dir",
                "추천 경로를 선택하거나 같은 칸에 직접 입력하세요. 라벨 파일명과 같은 원본 이미지가 들어 있는 폴더를 지정하세요.",
            )
    with st.form("convert_form"):
        left, right = st.columns(2)
        with left:
            st.markdown('<div class="form-section">출력 설정</div>', unsafe_allow_html=True)
            output_dir = st.text_input("출력 디렉터리", value=WORKSPACE_DEFAULTS["converted"])
            mapping_options = ["자동 탐색"] + class_mapping_candidates(workspace, input_path)
            convert_classes = st.selectbox(
                "클래스 매핑 파일 (선택)",
                mapping_options,
                index=0,
                accept_new_options=True,
                help=(
                    "비워두면 입력 라벨 폴더와 상위 폴더에서 data.yaml, dataset.yaml, classes.txt를 자동으로 찾습니다. "
                    "다른 파일을 쓰려면 이 칸에 직접 경로를 입력하세요."
                ),
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
        custom_mapping_text = ""
        if source_format == CUSTOM_MAPPING_FORMAT:
            st.markdown("**커스텀 라벨 매핑**")
            st.caption(
                "샘플 JSON 라벨 1개를 분석해 안전한 매핑 스펙을 만들고, 내부 파서가 그 스펙만 사용해 변환합니다. "
                "LLM이 생성한 코드는 실행하지 않습니다."
            )
            custom_mapping_text = st.text_area(
                "커스텀 매핑 스펙(JSON)",
                key="convert_custom_mapping_text",
                height=240,
                help="커스텀 JSON 라벨의 image/object/label/bbox 경로를 정의합니다. 비어 있으면 변환 실행 전에 먼저 분석이 필요합니다.",
            )
        analyze_custom_submit = st.form_submit_button("커스텀 포맷 분석", icon=":material/schema:")
        preflight_submit = st.form_submit_button("변환 사전 점검", icon=":material/rule:")
        convert_submit = st.form_submit_button("변환 실행", type="primary", icon=":material/sync_alt:")
    if analyze_custom_submit:
        if source_format != CUSTOM_MAPPING_FORMAT:
            st.info("입력 파일 형식을 custom_mapping으로 선택한 뒤 커스텀 포맷 분석을 실행하세요.")
        elif not input_path:
            st.error("입력 라벨 경로를 지정하세요.")
        else:
            try:
                with st.status("커스텀 라벨 샘플 분석 중", expanded=True) as status:
                    sample_path = sample_custom_label_file(resolve_workspace_path(workspace, input_path))
                    st.write(f"샘플 파일: {sample_path}")
                    model_name = os.getenv("CUSTOM_LABEL_MAPPER_MODEL") or resolve_planner_model()
                    spec = infer_custom_mapping_spec_from_sample(sample_path, model_name=model_name)
                    st.session_state.convert_custom_mapping_text_pending = json.dumps(spec, ensure_ascii=False, indent=2)
                    status.update(label="커스텀 매핑 스펙 생성 완료", state="complete", expanded=False)
                st.rerun()
            except Exception as exc:
                st.error(f"커스텀 포맷 분석 실패: {exc}")
    if preflight_submit:
        if not input_path or not image_dir or not target_formats:
            st.error("입력 라벨 경로, 이미지 디렉터리, 출력 포맷을 지정하세요.")
        elif source_format == CUSTOM_MAPPING_FORMAT and not custom_mapping_text.strip():
            st.error("custom_mapping 변환에는 커스텀 매핑 스펙(JSON)이 필요합니다. 먼저 커스텀 포맷 분석을 실행하세요.")
        else:
            try:
                st.session_state.convert_preflight_preview = build_convert_preflight_preview(
                    resolve_workspace_path(workspace, input_path),
                    resolve_workspace_path(workspace, image_dir),
                    source_format,
                    target_formats,
                    classes_path=optional_class_mapping_path(workspace, convert_classes),
                    duplicate_iou=duplicate_iou,
                    custom_mapping_spec=custom_mapping_text if source_format == CUSTOM_MAPPING_FORMAT else None,
                )
            except Exception as exc:
                st.error(f"사전 점검 실패: {exc}")
    if st.session_state.get("convert_preflight_preview"):
        render_convert_preflight_preview(st.session_state.convert_preflight_preview)
    if convert_submit:
        if not input_path or not image_dir or not output_dir or not target_formats:
            st.error("입력, 이미지, 출력 경로와 출력 포맷을 모두 지정하세요.")
        elif source_format == CUSTOM_MAPPING_FORMAT and not custom_mapping_text.strip():
            st.error("custom_mapping 변환에는 커스텀 매핑 스펙(JSON)이 필요합니다. 먼저 커스텀 포맷 분석을 실행하세요.")
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
                    "classes_path": optional_class_mapping_path(workspace, convert_classes),
                    "custom_label_mapping": custom_mapping_text if source_format == CUSTOM_MAPPING_FORMAT else None,
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
            generation_class_list = st.text_area(
                "검출 클래스 목록 (선택)",
                value="",
                placeholder="person\ncar\ncat 또는 person, car, cat",
                help="클래스 매핑 파일이 없거나 프롬프트에 대상 클래스가 명확하지 않을 때 입력하세요.",
            )
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
        missing_info = generation_missing_info(
            workspace,
            generation_images,
            generation_output,
            visualization_output,
            generation_formats,
            prompt,
            approve_expensive,
            generation_classes,
            generation_class_list,
        )
        if missing_info:
            st.error("라벨 생성을 실행하기 전에 추가 정보가 필요합니다.")
            st.dataframe(
                [{"필요 정보": item, "입력 안내": message} for item, message in missing_info],
                width="stretch",
            )
        else:
            effective_prompt = build_generation_prompt(prompt, generation_class_list)
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
                    "prompt": effective_prompt,
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
with editor_tab:
    render_label_editor_tab(workspace)
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
