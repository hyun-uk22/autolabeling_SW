import json
import os
from pathlib import Path
from typing import Dict, List

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QAction, QFontDatabase
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QDialog,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QStackedWidget,
    QStyle,
    QVBoxLayout,
    QWidget,
)

from .settings import ENV_FIELDS, SECRET_FIELDS, read_user_settings, save_user_settings, user_env_path
from .worker import WorkflowWorker
from ..core.workspace import (
    WORKSPACE_DEFAULTS,
    load_workspace,
    normalize_workspace,
    relative_to_workspace,
    resolve_workspace_path,
    save_workspace,
)


APP_STYLE = """
QWidget { color: #20242a; font-family: "Malgun Gothic", "Segoe UI"; font-size: 13px; }
QMainWindow, QWidget#root { background: #f4f5f7; }
QWidget#sidebar { background: #20252b; }
QLabel#brand { color: #ffffff; font-size: 20px; font-weight: 700; padding: 20px 16px 14px 16px; }
QListWidget { background: transparent; border: 0; outline: 0; color: #dfe3e8; padding: 4px 8px; }
QListWidget::item { min-height: 42px; padding: 0 12px; border-radius: 4px; }
QListWidget::item:selected { background: #3b82f6; color: white; }
QLabel#pageTitle { font-size: 22px; font-weight: 700; color: #171a1f; }
QLabel#sectionTitle { font-size: 14px; font-weight: 700; color: #343a43; padding-top: 10px; }
QLineEdit, QComboBox, QDoubleSpinBox, QSpinBox, QPlainTextEdit {
    background: white; border: 1px solid #cbd0d7; border-radius: 4px; padding: 7px;
}
QLineEdit:focus, QComboBox:focus, QPlainTextEdit:focus { border: 1px solid #3b82f6; }
QPushButton { min-height: 32px; border: 1px solid #b8bec7; border-radius: 4px; background: #ffffff; padding: 0 12px; }
QPushButton:hover { background: #eef1f4; }
QPushButton#primary { background: #2563eb; border-color: #2563eb; color: white; font-weight: 600; }
QPushButton#primary:hover { background: #1d4ed8; }
QPushButton:disabled { background: #e6e8eb; color: #9298a1; border-color: #d3d6da; }
QProgressBar { min-height: 6px; max-height: 6px; border: 0; background: #dfe3e8; }
QProgressBar::chunk { background: #3b82f6; }
QCheckBox { spacing: 7px; }
QTabWidget::pane { border: 0; }
QTabBar::tab { padding: 9px 14px; border-bottom: 2px solid transparent; }
QTabBar::tab:selected { color: #2563eb; border-bottom-color: #2563eb; }
"""


def _register_korean_font():
    korean_font = os.path.join(os.environ.get("WINDIR", "C:\\Windows"), "Fonts", "malgun.ttf")
    if os.path.exists(korean_font):
        QFontDatabase.addApplicationFont(korean_font)


class PathField(QWidget):
    def __init__(self, mode="directory", file_filter="모든 파일 (*.*)", base_dir=None, relative=False, parent=None):
        super().__init__(parent)
        self.mode = mode
        self.file_filter = file_filter
        self.base_dir = str(base_dir) if base_dir else None
        self.relative = relative
        self.edit = QLineEdit()
        self.button = QPushButton()
        self.button.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_DirOpenIcon))
        self.button.setToolTip("경로 선택")
        self.button.setFixedWidth(38)
        self.button.clicked.connect(self._browse)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)
        layout.addWidget(self.edit)
        layout.addWidget(self.button)

    def _browse(self):
        current = self.edit.text().strip() or os.getcwd()
        if self.base_dir and not os.path.isabs(current):
            current = os.path.join(self.base_dir, current)
        if self.mode == "file":
            value, _ = QFileDialog.getOpenFileName(self, "파일 선택", current, self.file_filter)
        else:
            value = QFileDialog.getExistingDirectory(self, "디렉터리 선택", current)
        if value:
            if self.base_dir and self.relative:
                value = relative_to_workspace(self.base_dir, value)
            self.edit.setText(value)

    def text(self):
        return self.edit.text().strip()

    def setText(self, value):
        self.edit.setText(value)


class FormatSelector(QWidget):
    def __init__(self, include_custom=False, parent=None):
        super().__init__(parent)
        names = ["yolo", "pascal_voc", "coco", "vision_json"]
        if include_custom:
            names.append("custom")
        self.checks = {name: QCheckBox(name) for name in names}
        self.checks["yolo"].setChecked(True)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        for check in self.checks.values():
            layout.addWidget(check)
        layout.addStretch()

    def values(self) -> List[str]:
        return [name for name, check in self.checks.items() if check.isChecked()]


class WorkspaceDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        _register_korean_font()
        self.setStyleSheet(APP_STYLE)
        self.workspace = None
        self.setWindowTitle("Workspace 선택")
        self.setMinimumWidth(560)
        layout = QVBoxLayout(self)
        title = QLabel("Workspace 선택")
        title.setObjectName("pageTitle")
        layout.addWidget(title)
        self.path_field = PathField()
        saved = load_workspace()
        self.path_field.setText(str(saved or (Path.home() / "Documents" / "AutoLabelWorkspace")))
        layout.addWidget(self.path_field)
        self.create_layout_check = QCheckBox("표준 폴더 구조 생성")
        self.create_layout_check.setChecked(not bool(saved))
        self.create_layout_check.setToolTip("이미 작업 중인 디렉터리를 그대로 사용하려면 체크 해제하세요.")
        layout.addWidget(self.create_layout_check)
        self.layout_hint = QLabel(
            "체크 시 workspace 아래에 다음 폴더가 자동 생성됩니다:\n"
            "  data/raw — 원본 이미지\n"
            "  data/labeled — 생성된 라벨\n"
            "  data/visualized — 라벨 시각화 결과\n"
            "  data/converted — 형식 변환 라벨\n"
            "  data/ground_truth — 평가용 정답 라벨\n"
            "  data/reports — 평가 리포트\n"
            "  configs/plugins.json — 플러그인 설정 파일"
        )
        self.layout_hint.setStyleSheet("color: #6b7280; font-size: 11px; padding-left: 22px;")
        self.layout_hint.setVisible(not bool(saved))
        layout.addWidget(self.layout_hint)
        self.create_layout_check.toggled.connect(self.layout_hint.setVisible)
        actions = QHBoxLayout()
        actions.addStretch()
        cancel = QPushButton("취소")
        cancel.clicked.connect(self.reject)
        apply_button = QPushButton("적용")
        apply_button.setObjectName("primary")
        apply_button.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_DialogApplyButton))
        apply_button.clicked.connect(self._apply)
        actions.addWidget(cancel)
        actions.addWidget(apply_button)
        layout.addLayout(actions)

    def _apply(self):
        value = self.path_field.text()
        if not value:
            QMessageBox.warning(self, "입력 확인", "Workspace 경로를 선택하세요.")
            return
        try:
            self.workspace = str(save_workspace(value, create_layout=self.create_layout_check.isChecked()))
        except OSError as exc:
            QMessageBox.critical(self, "Workspace 생성 실패", str(exc))
            return
        self.accept()


class OperationPage(QWidget):
    run_requested = Signal(object, bool)

    def __init__(self, title, parent=None):
        super().__init__(parent)
        self.root_layout = QVBoxLayout(self)
        self.root_layout.setContentsMargins(26, 22, 26, 22)
        self.root_layout.setSpacing(14)
        title_label = QLabel(title)
        title_label.setObjectName("pageTitle")
        self.root_layout.addWidget(title_label)
        self.form_widget = QWidget()
        self.form = QFormLayout(self.form_widget)
        self.form.setLabelAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        self.form.setHorizontalSpacing(18)
        self.form.setVerticalSpacing(10)
        self.form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
        self.root_layout.addWidget(self.form_widget)
        controls = QHBoxLayout()
        controls.addStretch()
        self.run_button = QPushButton("실행")
        self.run_button.setObjectName("primary")
        self.run_button.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_MediaPlay))
        self.run_button.clicked.connect(self._emit_plan)
        controls.addWidget(self.run_button)
        self.root_layout.addLayout(controls)
        self.progress = QProgressBar()
        self.progress.setRange(0, 0)
        self.progress.hide()
        self.root_layout.addWidget(self.progress)
        result_title = QLabel("실행 결과")
        result_title.setObjectName("sectionTitle")
        self.root_layout.addWidget(result_title)
        self.result = QPlainTextEdit()
        self.result.setReadOnly(True)
        self.result.setMinimumHeight(180)
        self.root_layout.addWidget(self.result, 1)

    def _emit_plan(self):
        try:
            plan, expensive = self.build_plan()
        except ValueError as exc:
            QMessageBox.warning(self, "입력 확인", str(exc))
            return
        self.run_requested.emit(plan, expensive)

    def build_plan(self):
        raise NotImplementedError

    def set_busy(self, busy):
        self.run_button.setDisabled(busy)
        self.progress.setVisible(busy)

    def show_result(self, value):
        self.result.setPlainText(json.dumps(value, ensure_ascii=False, indent=2, default=str))

    def show_error(self, message):
        self.result.setPlainText(message)


def _require_path(field: PathField, label: str) -> str:
    value = field.text()
    if not value:
        raise ValueError(f"{label} 경로를 선택하세요.")
    return value


class ConvertPage(OperationPage):
    def __init__(self, workspace, parent=None):
        super().__init__("라벨 형식 변환", parent)
        self.workspace = workspace
        self.input_path = PathField(base_dir=workspace, relative=True)
        self.image_dir = PathField(base_dir=workspace, relative=True)
        self.output_dir = PathField(base_dir=workspace, relative=True)
        self.input_path.setText(WORKSPACE_DEFAULTS["labels"])
        self.image_dir.setText(WORKSPACE_DEFAULTS["images"])
        self.output_dir.setText(WORKSPACE_DEFAULTS["converted"])
        self.source_format = QComboBox()
        self.source_format.addItems(["auto", "yolo", "pascal_voc", "coco", "vision_json", "csv", "generic_json"])
        self.formats = FormatSelector()
        self.duplicate_iou = QDoubleSpinBox()
        self.duplicate_iou.setRange(0.01, 1.0)
        self.duplicate_iou.setSingleStep(0.05)
        self.duplicate_iou.setValue(0.85)
        self.strict = QCheckBox("검증 이슈가 있는 레코드 제외")
        self.form.addRow("입력 라벨", self.input_path)
        self.form.addRow("이미지 디렉터리", self.image_dir)
        self.form.addRow("출력 디렉터리", self.output_dir)
        self.form.addRow("입력 포맷", self.source_format)
        self.form.addRow("출력 포맷", self.formats)
        self.form.addRow("중복 IoU", self.duplicate_iou)
        self.form.addRow("검증", self.strict)

    def build_plan(self):
        formats = self.formats.values()
        if not formats:
            raise ValueError("출력 포맷을 하나 이상 선택하세요.")
        operation = {
            "action": "convert",
            "input_path": resolve_workspace_path(self.workspace, _require_path(self.input_path, "입력 라벨")),
            "img_dir": resolve_workspace_path(self.workspace, _require_path(self.image_dir, "이미지 디렉터리")),
            "out_dir": resolve_workspace_path(self.workspace, _require_path(self.output_dir, "출력 디렉터리")),
            "source_format": self.source_format.currentText(),
            "formats": formats,
            "duplicate_iou": self.duplicate_iou.value(),
            "strict": self.strict.isChecked(),
        }
        return {"request_summary": "Desktop label conversion", "operations": [operation]}, False


class GeneratePage(OperationPage):
    def __init__(self, workspace, parent=None):
        super().__init__("자동 라벨 생성", parent)
        self.workspace = workspace
        self.image_dir = PathField(base_dir=workspace, relative=True)
        self.output_dir = PathField(base_dir=workspace, relative=True)
        self.visual_dir = PathField(base_dir=workspace, relative=True)
        self.image_dir.setText(WORKSPACE_DEFAULTS["images"])
        self.output_dir.setText(WORKSPACE_DEFAULTS["labels"])
        self.visual_dir.setText(WORKSPACE_DEFAULTS["visualized"])
        self.task_type = QComboBox()
        self.task_type.addItems(["object_detection", "classification", "segmentation", "pose_estimation", "ocr", "tracking", "all"])
        self.formats = FormatSelector()
        self.threshold = QDoubleSpinBox()
        self.threshold.setRange(0.0, 1.0)
        self.threshold.setSingleStep(0.05)
        self.threshold.setValue(0.75)
        self.inference_count = QSpinBox()
        self.inference_count.setRange(1, 10)
        self.inference_count.setValue(3)
        self.plugin_config = PathField("file", "JSON (*.json)", base_dir=workspace, relative=True)
        self.plugin_config.setText(WORKSPACE_DEFAULTS["plugin_config"])
        self.prompt = QPlainTextEdit()
        self.prompt.setMinimumHeight(90)
        self.prompt.setPlainText("Detect and classify all prominent objects in this image. Output strictly as JSON.")
        self.form.addRow("이미지 디렉터리", self.image_dir)
        self.form.addRow("라벨 출력", self.output_dir)
        self.form.addRow("시각화 출력", self.visual_dir)
        self.form.addRow("태스크", self.task_type)
        self.form.addRow("출력 포맷", self.formats)
        self.form.addRow("신뢰도 기준", self.threshold)
        self.form.addRow("초안 추론 횟수", self.inference_count)
        self.form.addRow("Plugin 설정", self.plugin_config)
        self.form.addRow("프롬프트", self.prompt)

    def build_plan(self):
        formats = self.formats.values()
        if not formats:
            raise ValueError("출력 포맷을 하나 이상 선택하세요.")
        operation = {
            "action": "generate",
            "task_type": self.task_type.currentText(),
            "img_dir": resolve_workspace_path(self.workspace, _require_path(self.image_dir, "이미지 디렉터리")),
            "out_dir": resolve_workspace_path(self.workspace, _require_path(self.output_dir, "라벨 출력")),
            "vis_dir": resolve_workspace_path(self.workspace, _require_path(self.visual_dir, "시각화 출력")),
            "formats": formats,
            "threshold": self.threshold.value(),
            "inference_count": self.inference_count.value(),
            "prompt": self.prompt.toPlainText().strip(),
            "plugin_config": resolve_workspace_path(self.workspace, self.plugin_config.text()) if self.plugin_config.text() else None,
            "require_approval": True,
        }
        return {"request_summary": "Desktop automatic label generation", "operations": [operation]}, True


class EvaluatePage(OperationPage):
    def __init__(self, workspace, parent=None):
        super().__init__("실험 결과 평가", parent)
        self.workspace = workspace
        self.ground_truth = PathField(base_dir=workspace, relative=True)
        self.output_dir = PathField(base_dir=workspace, relative=True)
        self.ground_truth.setText(WORKSPACE_DEFAULTS["ground_truth"])
        self.output_dir.setText(WORKSPACE_DEFAULTS["reports"])
        self.runs = QPlainTextEdit()
        self.runs.setPlaceholderText("baseline=D:/runs/baseline\ncascade=D:/runs/cascade")
        self.runs.setMinimumHeight(100)
        self.form.addRow("Ground truth", self.ground_truth)
        self.form.addRow("리포트 출력", self.output_dir)
        self.form.addRow("실험 경로", self.runs)

    def build_plan(self):
        runs: Dict[str, str] = {}
        for line in self.runs.toPlainText().splitlines():
            line = line.strip()
            if not line:
                continue
            if "=" not in line:
                raise ValueError("실험 경로는 이름=경로 형식으로 입력하세요.")
            name, path = line.split("=", 1)
            runs[name.strip()] = resolve_workspace_path(self.workspace, path.strip())
        if not runs:
            raise ValueError("평가할 실험 경로를 하나 이상 입력하세요.")
        operation = {
            "action": "evaluate",
            "gt_dir": resolve_workspace_path(self.workspace, self.ground_truth.text()) if self.ground_truth.text() else None,
            "out_dir": resolve_workspace_path(self.workspace, _require_path(self.output_dir, "리포트 출력")),
            "runs": runs,
        }
        return {"request_summary": "Desktop experiment evaluation", "operations": [operation]}, False


class SettingsPage(QWidget):
    saved = Signal(str)

    LABELS = {
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

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(26, 22, 26, 22)
        layout.setSpacing(14)
        title = QLabel("사용자 설정")
        title.setObjectName("pageTitle")
        layout.addWidget(title)
        path_label = QLabel(str(user_env_path()))
        path_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        layout.addWidget(path_label)
        form_widget = QWidget()
        form = QFormLayout(form_widget)
        form.setVerticalSpacing(10)
        form.setHorizontalSpacing(18)
        self.fields: Dict[str, QLineEdit] = {}
        for key in ENV_FIELDS:
            edit = QLineEdit()
            if key in SECRET_FIELDS:
                edit.setEchoMode(QLineEdit.EchoMode.Password)
            self.fields[key] = edit
            form.addRow(self.LABELS[key], edit)
        layout.addWidget(form_widget)
        actions = QHBoxLayout()
        self.reveal = QCheckBox("비밀값 표시")
        self.reveal.toggled.connect(self._toggle_secrets)
        actions.addWidget(self.reveal)
        actions.addStretch()
        save_button = QPushButton("저장")
        save_button.setObjectName("primary")
        save_button.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_DialogSaveButton))
        save_button.clicked.connect(self._save)
        actions.addWidget(save_button)
        layout.addLayout(actions)
        layout.addStretch()
        self.reload()

    def reload(self):
        values = read_user_settings()
        for key, edit in self.fields.items():
            edit.setText(values.get(key, ""))

    def _toggle_secrets(self, visible):
        mode = QLineEdit.EchoMode.Normal if visible else QLineEdit.EchoMode.Password
        for key in SECRET_FIELDS:
            self.fields[key].setEchoMode(mode)

    def _save(self):
        values = {key: edit.text() for key, edit in self.fields.items()}
        try:
            path = save_user_settings(values)
        except OSError as exc:
            QMessageBox.critical(self, "저장 실패", str(exc))
            return
        self.saved.emit(str(path))
        QMessageBox.information(self, "설정 저장", "사용자 설정을 저장했습니다.")


class MainWindow(QMainWindow):
    def __init__(self, workspace=None):
        super().__init__()
        self.workspace = str(normalize_workspace(workspace or load_workspace() or os.getcwd()))
        _register_korean_font()
        self.setWindowTitle("AutoLabel")
        self.resize(1120, 760)
        self.setMinimumSize(900, 620)
        self.setStyleSheet(APP_STYLE)
        self.worker = None
        self.active_page = None

        root = QWidget()
        root.setObjectName("root")
        self.setCentralWidget(root)
        layout = QHBoxLayout(root)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        sidebar = QWidget()
        sidebar.setObjectName("sidebar")
        sidebar.setFixedWidth(210)
        side_layout = QVBoxLayout(sidebar)
        side_layout.setContentsMargins(0, 0, 0, 12)
        brand = QLabel("AutoLabel")
        brand.setObjectName("brand")
        side_layout.addWidget(brand)
        self.navigation = QListWidget()
        side_layout.addWidget(self.navigation, 1)
        layout.addWidget(sidebar)

        self.pages = QStackedWidget()
        layout.addWidget(self.pages, 1)

        self.convert_page = ConvertPage(self.workspace)
        self.generate_page = GeneratePage(self.workspace)
        self.evaluate_page = EvaluatePage(self.workspace)
        self.settings_page = SettingsPage()
        page_data = [
            ("형식 변환", self.convert_page, QStyle.StandardPixmap.SP_FileDialogDetailedView),
            ("라벨 생성", self.generate_page, QStyle.StandardPixmap.SP_ComputerIcon),
            ("평가", self.evaluate_page, QStyle.StandardPixmap.SP_FileDialogInfoView),
            ("설정", self.settings_page, QStyle.StandardPixmap.SP_FileDialogContentsView),
        ]
        for label, page, icon_type in page_data:
            self.pages.addWidget(page)
            item = QListWidgetItem(self.style().standardIcon(icon_type), label)
            self.navigation.addItem(item)
        self.navigation.currentRowChanged.connect(self.pages.setCurrentIndex)
        self.navigation.setCurrentRow(0)

        for page in (self.convert_page, self.generate_page, self.evaluate_page):
            page.run_requested.connect(self._run_workflow)
        self.settings_page.saved.connect(lambda path: self.statusBar().showMessage(f"설정 저장: {path}", 5000))

        exit_action = QAction("종료", self)
        exit_action.triggered.connect(self.close)
        self.addAction(exit_action)
        self.statusBar().showMessage(f"Workspace: {self.workspace}")

    def _run_workflow(self, plan, expensive):
        if self.worker and self.worker.isRunning():
            QMessageBox.warning(self, "실행 중", "현재 작업이 끝난 뒤 다시 실행하세요.")
            return
        if expensive:
            answer = QMessageBox.question(
                self,
                "모델 호출 승인",
                "라벨 생성 중 고비용 모델 API가 호출될 수 있습니다. 실행할까요?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if answer != QMessageBox.StandardButton.Yes:
                return

        self.active_page = self.sender()
        self.active_page.set_busy(True)
        self.active_page.result.clear()
        self.statusBar().showMessage("작업 실행 중")
        self.worker = WorkflowWorker(plan, auto_approve=expensive, parent=self)
        self.worker.completed.connect(self._workflow_completed)
        self.worker.failed.connect(self._workflow_failed)
        self.worker.finished.connect(self._worker_finished)
        self.worker.start()

    def _workflow_completed(self, result):
        if self.active_page:
            self.active_page.show_result({
                "status": result.get("status"),
                "outputs": result.get("operation_outputs", []),
                "errors": result.get("errors", []),
                "history_path": result.get("history_path", ""),
            })
        self.statusBar().showMessage(f"작업 상태: {result.get('status', 'unknown')}", 8000)

    def _workflow_failed(self, message):
        if self.active_page:
            self.active_page.show_error(message)
        self.statusBar().showMessage("작업 실패", 8000)

    def _worker_finished(self):
        if self.active_page:
            self.active_page.set_busy(False)
        self.worker = None
        self.active_page = None

    def closeEvent(self, event):
        if self.worker and self.worker.isRunning():
            QMessageBox.warning(self, "작업 실행 중", "작업이 끝난 뒤 프로그램을 종료하세요.")
            event.ignore()
            return
        super().closeEvent(event)
