from typing import Any, Dict


def _operation_summary(operation: Dict[str, Any]) -> str:
    action = operation.get("action", "unknown")
    input_path = operation.get("input_path") or operation.get("img_dir") or "-"
    out_dir = operation.get("out_dir") or "-"
    formats = ", ".join(operation.get("formats") or [])
    return f"- {action}: `{input_path}` -> `{out_dir}` ({formats or 'default'})"


def describe_plan(proposal: Dict[str, Any], workspace: str = "") -> str:
    plan = proposal.get("plan", proposal)
    operations = plan.get("operations", [])
    lines = ["실행 전 확인이 필요합니다.", ""]
    if workspace:
        lines.append(f"Workspace: `{workspace}`")
    lines.append(f"요약: {plan.get('request_summary', '작업 계획')}")
    lines.append("")
    lines.extend(_operation_summary(operation) for operation in operations)
    lines.append("")
    lines.append("문제가 있는 입력 데이터가 있으면 실행 전에 표로 보여주고 제외 여부를 확인합니다.")
    return "\n".join(lines)


def describe_result(result: Dict[str, Any], workspace: str = "") -> str:
    status = result.get("status", "unknown")
    outputs = result.get("operation_outputs", [])
    lines = [f"작업 상태: `{status}`"]
    for output in outputs:
        action = output.get("action", "unknown")
        if action == "convert":
            lines.append(
                "변환 결과: "
                f"{output.get('records_converted', 0)}/{output.get('records_read', 0)}개 완료"
            )
            if output.get("user_action_report_path"):
                lines.append(f"사용자 리포트: `{output['user_action_report_path']}`")
        elif action == "generate":
            lines.append(
                "라벨링 결과: "
                f"{output.get('images', 0)}개 이미지, {output.get('total_labels', 0)}개 라벨"
            )
        elif action == "evaluate":
            lines.append(f"평가 결과: {len(output.get('rows', []))}개 실행 비교")
    if result.get("errors"):
        lines.append("오류: " + " / ".join(result["errors"][:3]))
    return "\n".join(lines)
