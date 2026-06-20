import traceback
from typing import Any, Dict

from PySide6.QtCore import QThread, Signal


class WorkflowWorker(QThread):
    completed = Signal(dict)
    failed = Signal(str)

    def __init__(self, plan: Dict[str, Any], auto_approve: bool = False, parent=None):
        super().__init__(parent)
        self.plan = plan
        self.auto_approve = auto_approve

    def run(self):
        try:
            from src.workflow.service import execute_workflow_plan

            result = execute_workflow_plan(
                self.plan,
                auto_approve=self.auto_approve,
                thread_id="desktop",
            )
            self.completed.emit(result)
        except Exception:
            self.failed.emit(traceback.format_exc())
