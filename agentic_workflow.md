# LangGraph 상위 Orchestrator 명세

## 1. 목적

`agentic_workflow.py`는 사용자의 자연어 또는 구조화 JSON 요청을 실행 가능한 `WorkflowPlan`으로 변환하고, 자동 라벨 생성, 기존 라벨 변환, 정량 평가를 하나의 상태 그래프에서 실행한다.

LangGraph는 실제 라벨 추론 모델을 대체하지 않는다. 상태, 조건 분기, 반복, 승인, checkpoint, 재개를 담당하고 실제 처리는 기존 service와 plugin이 담당한다.

```text
LangGraph
  - plan/routing/retry/approval/checkpoint
        |
        +-- WorkflowRuntime
              +-- VLM clients
              +-- specialist plugins
              +-- label importer/validator/exporter
              +-- evaluation/report
```

## 2. 실행 파일과 모듈

| 파일 | 책임 |
| --- | --- |
| `agentic_workflow.py` | CLI, SQLite checkpointer, invoke/resume/status |
| `src/workflow/models.py` | `OperationPlan`, `WorkflowPlan`, `WorkflowState` |
| `src/workflow/planner.py` | 자연어 또는 JSON을 typed plan으로 변환 |
| `src/workflow/runtime.py` | 비직렬화 모델 객체와 실제 task service |
| `src/workflow/schema_repair.py` | 입력 포맷 재분석 후보와 라벨 repair |
| `src/workflow/graph.py` | LangGraph node/edge/interrupt 정의 |
| `configs/workflow.example.json` | 복합 workflow plan 예시 |

## 3. WorkflowPlan

상위 plan은 operation 목록을 요청 순서대로 보유한다.

```json
{
  "request_summary": "Generate, convert, and evaluate labels.",
  "operations": [
    {
      "action": "generate",
      "task_type": "segmentation",
      "img_dir": "data/raw",
      "out_dir": "data/runs/cascade",
      "formats": ["coco", "vision_json"],
      "plugin_config": "configs/plugins.example.json",
      "threshold": 0.75,
      "require_approval": true,
      "max_retries": 2
    },
    {
      "action": "convert",
      "input_path": "data/runs/cascade/vision_annotations.jsonl",
      "img_dir": "data/raw",
      "out_dir": "data/converted",
      "formats": ["coco"],
      "source_format": "auto",
      "duplicate_iou": 0.85
    },
    {
      "action": "evaluate",
      "out_dir": "data/reports",
      "gt_dir": "data/ground_truth",
      "runs": {"cascade": "data/runs/cascade"}
    }
  ]
}
```

`convert`는 `input_path`, `evaluate`는 `runs`가 필수다. Pydantic validation과 graph의 plan validation node가 이를 검사한다.

`convert`의 디렉터리 `input_path`와 `source_format: "auto"`를 함께 사용하면 지원되는 표준 라벨 포맷을 파일별로 판별하고 이미지별로 병합한다. `duplicate_iou`는 같은 클래스 공간 라벨의 중복 기준이며 입력 탐색·병합 상세 결과는 conversion report의 `input_summary`에 저장된다.

## 4. 자연어 Planner

planner 우선순위:

1. `--plan` JSON 파일
2. JSON 문자열 형태의 `--request`
3. `--planner-model`, `PLANNER_MODEL`, `LOW_MODEL`을 사용한 LLM planning
4. keyword 기반 deterministic fallback

LLM planner는 JSON만 반환하도록 지시되고 결과는 `WorkflowPlan`으로 검증된다.

fallback은 `생성/generate`, `변환/convert`, `평가/evaluate` 키워드를 찾아 operation을 구성한다. 경로가 생략된 복잡한 요청에서는 fallback이 기본 경로를 사용하므로, 재현 가능한 실험에는 JSON plan이 권장된다.

## 5. Graph 구조

```text
START
  -> parse_request
  -> validate_plan
  -> begin_operation
       |
       +-- generate
       |    -> prepare_generate
       |    -> select_image
       |    -> run_specialists
       |         +-- 결과 없음 또는 vlm_first -> generate_draft
       |    -> decide_high
       |         +-- approval_gate -> high_verify
       |         +-- validate_generated
       |                  +-- repair_generated -> validate_generated
       |                  +-- save_generated -> select_image
       |    -> finalize_generate
       |
       +-- convert
       |    -> prepare_convert
       |    -> load_conversion
       |         +-- reanalyze_schema -> load_conversion
       |    -> validate_conversion
       |         +-- repair_conversion -> validate_conversion
       |    -> export_conversion
       |
       +-- evaluate
            -> execute_evaluate
  -> advance_operation
       +-- begin_operation
       +-- finalize_workflow
  -> END
```

## 6. Generate Workflow

1. 이미지 목록을 정렬한다.
2. `classes_path`, 프롬프트의 YOLO `names:` 블록, plugin 설정에서 후보 클래스를 구성한다.
3. 기본 `specialist_first` 전략에서는 task를 지원하는 plugin을 설정 순서대로 먼저 실행한다.
4. specialist 결과가 있으면 low VLM 호출 없이 validation/export 후보로 사용한다.
5. specialist 결과가 비어 있거나 operation이 `generation_strategy=vlm_first`이면 low VLM을 반복 호출해 draft와 self-consistency를 만든다.
6. plugin agreement와 VLM consistency를 함께 검사한다.
7. threshold 미만 또는 validation issue가 있으면 high verification 대상으로 지정한다.
8. high VLM 호출 전 approval interrupt를 발생시킨다.
9. 승인 시 high 결과와 전문 모델 결과를 병합한다.
10. validation 실패 시 repair node가 malformed 항목을 제거한다.
11. 이미지 결과를 checkpoint state와 run record에 추가한다.
12. 모든 이미지가 끝나면 포맷 export와 metric 저장을 수행한다.

이 경로는 직접 CLI와 같은 generation strategy를 사용한다. 기본값은 `specialist_first`이고, 기존 VLM 우선 실험은 `vlm_first`로 되돌릴 수 있다.

## 7. Conversion Schema Retry

변환 importer가 예외를 발생시키거나 레코드를 하나도 만들지 못하면 `schema_error` 상태가 된다.

`reanalyze_schema`는 파일 확장자, 디렉터리 내 파일 종류, COCO 필수 JSON key를 사용해 source format 후보를 다시 구성한다.

```text
requested coco -> parse failure -> directory contains txt -> retry yolo
```

재시도는 operation의 `max_retries`를 넘지 않는다. 현재 schema 재분석은 deterministic heuristic이며 임의 field mapping을 LLM이 새로 생성하지는 않는다.

## 8. Validation Repair

생성과 변환 모두 `validate_result()`를 사용한다.

repair node는 다음 malformed 항목을 제거한다.

- label 없는 classification/bbox/segment
- 범위를 벗어나거나 순서가 잘못된 bbox
- point가 부족하거나 좌표가 잘못된 polygon
- keypoint가 없거나 잘못된 pose
- text가 없거나 bbox가 잘못된 OCR
- track id/label/bbox가 잘못된 tracking

repair 후 validation node로 돌아간다. `max_retries` 이후에도 issue가 남으면 결과와 issue를 기록하고 다음 단계로 진행한다.

## 9. High Model 승인

high verification이 필요하면 LangGraph `interrupt()`를 호출한다.

```bash
python agentic_workflow.py --plan configs/workflow.example.json --thread-id exp-001
```

승인:

```bash
python agentic_workflow.py --thread-id exp-001 --resume approve
```

거절:

```bash
python agentic_workflow.py --thread-id exp-001 --resume reject
```

프로세스 종료 등 사용자 interrupt가 아닌 checkpoint에서 계속 실행:

```bash
python agentic_workflow.py --thread-id exp-001 --resume continue
```

거절하면 high VLM을 호출하지 않고 현재 low/plugin 결과를 validation/repair 단계로 전달한다.

`--auto-approve` 또는 operation의 `require_approval=false`이면 interrupt 없이 high VLM을 호출한다.

## 10. Checkpoint와 Resume

기본 checkpoint DB:

```text
data/workflow/checkpoints.sqlite
```

각 workflow는 `thread_id`로 구분된다. 새 실행에서 `--thread-id`를 생략하면 UUID를 생성해 출력한다.

상태 확인:

```bash
python agentic_workflow.py --thread-id exp-001 --status
```

checkpoint에는 plan, operation index, image index, 현재 결과, API attempt, validation issue, output 목록, history가 저장된다. VLM client와 model weight 같은 비직렬화 객체는 저장하지 않고 resume 시 `WorkflowRuntime`이 operation 설정으로 다시 초기화한다.

LangGraph checkpoint는 node 완료 시점의 상태를 보존한다. 외부 API 호출 도중 프로세스가 종료되면 해당 node가 재실행되어 같은 API 호출이 반복될 수 있으므로 완전한 exactly-once 보장은 아니다.

## 11. 이미지별 실행 이력

state의 `history`에 다음 event가 누적된다.

- plan 생성/검증
- operation 시작/종료
- 이미지 선택
- draft 생성
- specialist 실행
- high 결정과 승인
- high verification
- validation/repair
- 이미지 완료
- export/evaluation
- workflow 완료

최종 파일:

```text
data/workflow/<thread_id>/workflow_history.json
```

## 12. 동적 Plugin 선택

operation의 `plugin_config`가 지정되면 registry가 JSON을 읽는다.

- `enabled=false` plugin 제외
- 현재 `task_type`을 지원하는 plugin만 실행
- `class=module:ClassName` custom plugin 동적 import
- 설정 순서대로 chain 실행

resume 후 runtime이 재생성되어도 같은 plugin config를 다시 읽어 chain을 복원한다.

## 13. 복합 Operation

하나의 plan에 generate, convert, evaluate를 원하는 순서로 넣을 수 있다. 각 operation의 output은 `operation_outputs`에 누적되고 `advance_operation`이 다음 operation으로 이동한다.

```bash
python agentic_workflow.py --plan configs/workflow.example.json --thread-id full-pipeline
```

앞 operation의 출력 경로와 다음 operation의 입력 경로는 plan에서 명시적으로 연결해야 한다. 현재 graph가 이전 output을 보고 다음 path를 자동 치환하지는 않는다.

## 14. 유지보수 경계

| 계층 | 변경 이유 |
| --- | --- |
| `core/model_config.py` | provider/model 선택 규칙 변경 |
| `utils/result_metrics.py` | confidence/count/uncertainty 공식 변경 |
| `workflow/planner.py` | 자연어 plan schema/prompt 변경 |
| `workflow/graph.py` | 노드 순서와 조건 분기 변경 |
| `workflow/runtime.py` | 실제 generate/convert/evaluate 동작 변경 |
| `plugins/` | 전문 모델 추가/교체 |
| importer/validator/exporter | 라벨 포맷 및 검증 규칙 변경 |

Graph state에는 JSON 직렬화 가능한 값만 넣고 client/model 객체는 runtime에 둔다. 이 경계를 유지해야 SQLite checkpoint가 안정적으로 동작한다.

## 15. 기존 CLI와의 관계

- `main.py`: 자동 라벨 생성 직접 실행 및 기존 호환
- `convert_labels.py`: 라벨 변환 직접 실행
- `evaluate_experiments.py`: 평가 직접 실행
- `agentic_workflow.py`: 자연어/복합 실행, 승인, retry, checkpoint가 필요한 상위 경로

기존 CLI는 제거하지 않았으며 단일 기능 디버깅과 ablation에 사용할 수 있다. 공통 모델 설정과 result metric은 별도 모듈로 추출해 두 경로가 공유한다.

## 16. 현재 제한 사항

- schema 재분석은 heuristic이며 LLM field mapping 생성은 미지원
- repair는 malformed 항목 제거 중심이며 의미적 라벨 수정은 하지 않음
- 자연어 fallback은 복잡한 경로를 안정적으로 추출하지 못함
- approval은 불확실 이미지마다 발생할 수 있음
- 외부 API node의 exactly-once 실행은 보장하지 않음
- 복합 operation 사이 path 자동 binding 미지원
- SQLite 단일 파일의 다중 프로세스 병렬 접근은 별도 운영 검증 필요
- 실제 전문 모델 실행에는 추가 dependency와 checkpoint 다운로드 필요
