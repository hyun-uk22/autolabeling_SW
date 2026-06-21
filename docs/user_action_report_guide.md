# 오류 데이터 사용자 수정 제안 리포트

변환이나 검증 과정에서 문제가 발견되면 `user_action_report`가 생성된다. 기존 필드와 호환성을 유지하면서, 각 이슈에는 사용자가 바로 확인할 수 있는 수정 정보가 추가된다.

## 주요 필드

- `message`: 오류를 사람이 읽기 쉬운 한국어 문장으로 설명한다.
- `user_action`: 기존 리포트와 호환되는 기본 조치 문구다.
- `suggestions`: 같은 오류를 해결할 때 확인할 항목 목록이다.
- `affected_path`: 오류 문자열에서 추출한 관련 파일 경로다. 경로 정보가 없으면 빈 문자열이다.
- `fix_instruction`: `affected_path`를 포함한 구체적인 수정 지침이다.
- `priority_actions`: 이미지별 핵심 수정 지침을 최대 3개까지 모아 보여준다.

## 사용 흐름

1. Streamlit 또는 CLI 변환 결과에서 `preflight`를 먼저 확인한다.
2. `user_action_report.status`가 `needs_review` 또는 `partial_success`라면 `recommended_actions`를 확인한다.
3. `detailed_records[].priority_actions`를 우선 수정한다.
4. 특정 오류의 원인이 필요하면 `detailed_records[].detailed_issues[]`의 `affected_path`, `suggestions`, `fix_instruction`을 확인한다.
5. 수정 후 같은 입력으로 변환을 다시 실행한다.

## 예시

YOLO 라벨 행이 비어 있는 경우:

```json
{
  "code": "no_label_rows",
  "message": "YOLO 라벨 행이 없습니다",
  "affected_path": "D:/output/sample.txt",
  "fix_instruction": "YOLO 출력 파일(D:/output/sample.txt)에 class_id와 bbox 좌표 4개로 구성된 라벨 행이 생성되도록 입력 bbox를 확인하세요.",
  "suggestions": [
    "YOLO txt에 class_id와 bbox 좌표 행이 있는지 확인하세요.",
    "bbox가 없는 task라면 Vision JSON 등 표현 가능한 출력 포맷을 선택하세요."
  ]
}
```
