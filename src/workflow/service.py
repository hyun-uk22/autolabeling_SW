from typing import Any, Dict

from .graph import build_workflow_graph
from .runtime import WorkflowRuntime


def execute_workflow_plan(
    plan: Dict[str, Any],
    auto_approve: bool = False,
    thread_id: str = "local-ui",
) -> Dict[str, Any]:
    graph = build_workflow_graph(WorkflowRuntime())
    result = graph.invoke(
        {
            "request": "",
            "supplied_plan": plan,
            "auto_approve": auto_approve,
            "thread_id": thread_id,
            "history": [],
            "errors": [],
        },
        config={"recursion_limit": 100000},
    )
    return dict(result)

