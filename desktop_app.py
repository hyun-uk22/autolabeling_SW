import sys

from src.ui.settings import load_user_environment


def main():
    load_user_environment()

    if "--smoke-test" in sys.argv:
        from src.workflow.graph import build_workflow_graph
        from src.workflow.runtime import WorkflowRuntime

        build_workflow_graph(WorkflowRuntime())
        return 0

    from PySide6.QtWidgets import QApplication

    from src.ui.main_window import MainWindow, WorkspaceDialog

    app = QApplication(sys.argv)
    app.setApplicationName("AutoLabel")
    app.setOrganizationName("AutoLabel")
    workspace_dialog = WorkspaceDialog()
    if workspace_dialog.exec() != WorkspaceDialog.DialogCode.Accepted:
        return 0
    window = MainWindow(workspace_dialog.workspace)
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
