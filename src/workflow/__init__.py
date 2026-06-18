from .models import OperationPlan, WorkflowPlan, WorkflowState


def build_workflow_graph(*args, **kwargs):
    from .graph import build_workflow_graph as _build_workflow_graph

    return _build_workflow_graph(*args, **kwargs)


__all__ = ["OperationPlan", "WorkflowPlan", "WorkflowState", "build_workflow_graph"]
