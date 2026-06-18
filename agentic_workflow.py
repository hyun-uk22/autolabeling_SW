import argparse
import json
import os
import sqlite3
import uuid


def load_plan(path):
    if not path:
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser(description="LangGraph orchestrator for natural-language vision data workflows.")
    parser.add_argument("--request", default="", help="Natural-language workflow request or WorkflowPlan JSON")
    parser.add_argument("--plan", default=None, help="Optional WorkflowPlan JSON file")
    parser.add_argument("--thread-id", default=None, help="Persistent workflow thread id")
    parser.add_argument("--checkpoint-db", default="data/workflow/checkpoints.sqlite", help="SQLite checkpoint database")
    parser.add_argument("--planner-model", default=None, help="Optional model used to convert natural language into WorkflowPlan")
    parser.add_argument("--auto-approve", action="store_true", help="Automatically approve expensive high-model calls")
    parser.add_argument("--allow-same-model", action="store_true", help="Allow identical low/high VLMs")
    parser.add_argument("--resume", choices=["approve", "reject", "continue"], default=None, help="Resume an interrupted thread")
    parser.add_argument("--status", action="store_true", help="Show the latest checkpoint state without executing")
    args = parser.parse_args()

    try:
        from langgraph.checkpoint.sqlite import SqliteSaver
        from langgraph.types import Command
        from src.workflow.graph import build_workflow_graph
        from src.workflow.runtime import WorkflowRuntime
    except ImportError as exc:
        raise SystemExit(
            "LangGraph dependencies are missing. Run: pip install -r requirements.txt"
        ) from exc

    thread_id = args.thread_id or str(uuid.uuid4())
    os.makedirs(os.path.dirname(args.checkpoint_db) or ".", exist_ok=True)
    connection = sqlite3.connect(args.checkpoint_db, check_same_thread=False)
    checkpointer = SqliteSaver(connection)
    runtime = WorkflowRuntime(args.planner_model, allow_same_model=args.allow_same_model)
    graph = build_workflow_graph(runtime, checkpointer=checkpointer)
    config = {
        "configurable": {"thread_id": thread_id},
        "recursion_limit": 100000,
    }

    print(f"Workflow thread id: {thread_id}")
    try:
        if args.status:
            snapshot = graph.get_state(config)
            print(json.dumps(snapshot.values if snapshot else {}, ensure_ascii=False, indent=2, default=str))
            return

        if args.resume:
            if args.resume == "approve":
                result = graph.invoke(Command(resume={"approved": True}), config=config)
            elif args.resume == "reject":
                result = graph.invoke(Command(resume={"approved": False}), config=config)
            else:
                result = graph.invoke(None, config=config)
        else:
            if not args.request and not args.plan:
                parser.error("--request or --plan is required for a new workflow")
            result = graph.invoke(
                {
                    "request": args.request,
                    "supplied_plan": load_plan(args.plan),
                    "auto_approve": args.auto_approve,
                    "thread_id": thread_id,
                    "history": [],
                    "errors": [],
                },
                config=config,
            )

        interrupts = result.get("__interrupt__", []) if isinstance(result, dict) else []
        if interrupts:
            print("Workflow interrupted for user input:")
            for item in interrupts:
                print(json.dumps(getattr(item, "value", item), ensure_ascii=False, indent=2, default=str))
            print(f"Resume with: python agentic_workflow.py --thread-id {thread_id} --resume approve")
            return

        print(f"Workflow status: {result.get('status', 'unknown')}")
        print(json.dumps(result.get("operation_outputs", []), ensure_ascii=False, indent=2, default=str))
        if result.get("history_path"):
            print(f"History: {os.path.abspath(result['history_path'])}")
    finally:
        connection.close()


if __name__ == "__main__":
    main()
