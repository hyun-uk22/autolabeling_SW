# 자동 라벨 생성 기능 명세

## 1. 목적

이 문서는 이미지로부터 새 라벨을 생성하는 자동 라벨링 기능의 상세 명세, 실행 조건, 데이터 흐름, 계층적 검증 방식, 태스크별 처리, 전문 모델 plugin 연동, 출력 및 평가 과정을 설명한다.

자동 라벨 생성의 직접 실행 진입점은 `main.py`다. 자연어 계획, 승인 interrupt, checkpoint/resume, 복합 operation이 필요하면 `agentic_workflow.py`를 사용한다.

이 기능은 기존 라벨 파일을 다른 포맷으로 바꾸는 `convert_labels.py`와 구분된다.

| 기능 | 입력 | 출력 | 실행 파일 |
| --- | --- | --- | --- |
| 자동 라벨 생성 | 이미지 + 자연어 prompt | 새 라벨과 검증 지표 | `main.py` |
| Agentic 복합 workflow | 자연어 또는 WorkflowPlan | 생성/변환/평가 결과 | `agentic_workflow.py` |
| 기존 라벨 변환 | 기존 라벨 + 원본 이미지 | 변환된 라벨 | `convert_labels.py` |

## 2. 설계 목표

현재 자동 라벨 생성 기능은 다음 목표를 가진다.

1. 자연어 prompt를 사용해 라벨링 대상과 태스크를 지정한다.
2. 저비용 VLM의 반복 추론으로 초안 라벨을 생성한다.
3. 반복 결과의 self-consistency를 측정한다.
4. consistency가 낮은 이미지만 고성능 VLM으로 에스컬레이션한다.
5. 선택적으로 태스크별 전문 모델 plugin으로 결과를 보강한다.
6. 결과를 내부 공통 표현으로 정규화한다.
7. YOLO, Pascal VOC, COCO, Vision JSONL, custom 포맷으로 저장한다.
8. 처리 시간, API 호출, uncertainty, GT 평가 결과를 기록한다.

## 3. 주요 구성 요소

| 구성 요소 | 파일 | 책임 |
| --- | --- | --- |
| 실행 orchestration | `main.py` | 옵션 처리, 모델 초기화, 이미지 반복, 저장, metric |
| VLM client | `src/core/llm_client.py` | provider 호출, JSON 추출, 좌표/신뢰도 정규화 |
| Labeling Agent | `src/agents/labeling_agent.py` | low VLM 반복 추론 |
| Verification Agent | `src/agents/verification_agent.py` | consistency 계산 및 high VLM escalation |
| Dataset Insight Agent | `src/agents/insight_agent.py` | 클래스 분포와 불균형 요약 |
| 내부 결과 모델 | `src/core/models.py` | 태스크별 canonical schema |
| consistency | `src/utils/geometry.py` | IoU 및 Jaccard 기반 일관성 계산 |
| plugin registry | `src/plugins/registry.py` | built-in/custom plugin 로드 |
| plugin orchestrator | `src/plugins/orchestrator.py` | 전문 모델 실행, 결과 병합, 점수 갱신 |
| built-in plugins | `src/plugins/builtin.py` | CLIP, DINO, SAM, pose, OCR, tracking adapter |
| exporter | `src/utils/format_converter.py` | 라벨 포맷 저장 |
| visualization | `src/utils/visualize.py` | 이미지 위 라벨 시각화 |
| evaluation | `src/utils/evaluation.py` | YOLO GT 평가와 실험 리포트 |

## 4. 지원 태스크

`--task_type`으로 태스크를 선택한다.

| task type | VLM 출력 필드 | 전문 plugin 예시 | 대표 출력 |
| --- | --- | --- | --- |
| `classification` | `classifications` | CLIP | Vision JSONL/custom |
| `object_detection` | `boxes` | Grounding DINO | YOLO/VOC/COCO/Vision JSONL |
| `segmentation` | `segments` | Grounding DINO + SAM | COCO/Vision JSONL |
| `pose_estimation` | `poses` | Ultralytics pose | Vision JSONL/custom |
| `ocr` | `texts` | EasyOCR | Vision JSONL/custom |
| `tracking` | `tracks` | Grounding DINO + ByteTrack | Vision JSONL/custom |
| `all` | 전체 필드 | 설정된 모든 plugin | Vision JSONL/COCO/custom |

기본 태스크는 `object_detection`이다.

태스크 지원은 두 층으로 구성된다.

- VLM layer: 태스크별 JSON schema를 prompt로 제공해 결과를 생성한다.
- Specialist layer: plugin 설정이 있을 때 태스크 전용 모델로 결과를 보강한다.

전문 모델 plugin을 지정하지 않으면 모든 태스크는 VLM 응답만으로 생성된다.

## 5. 전체 워크플로우

```text
CLI 옵션 / .env
      |
      v
low/high 모델명 해석 및 cascade 검증
      |
      v
출력 포맷 설정 + optional plugin 설정 로드
      |
      v
API key 확인 및 VLM client 초기화
      |
      v
입력 이미지 목록 정렬
      |
      +------------------------------------------+
      | 이미지별 반복                            |
      |                                          |
      |  Low VLM N회 추론                        |
      |          |                               |
      |          v                               |
      |  태스크별 consistency 계산               |
      |          |                               |
      |          +-- threshold 이상 --> 첫 결과  |
      |          |                               |
      |          +-- threshold 미만 --> High VLM |
      |                                          |
      |  optional specialist plugin chain        |
      |          |                               |
      |          v                               |
      |  결과 병합 + plugin score + uncertainty  |
      |          |                               |
      |          v                               |
      |  포맷 저장 + 시각화 + metric 누적        |
      +------------------------------------------+
      |
      v
dataset 단위 exporter finalize
      |
      v
run_metrics / run_summary / optional GT evaluation
      |
      v
Dataset Insight Report
```

## 6. 실행 전 설정

### 6.1 기본 의존성

```powershell
pip install -r requirements.txt
```

기본 의존성은 OpenAI, Anthropic, AWS Bedrock, Pydantic, Pillow, NumPy 등을 포함한다.

### 6.2 전문 모델 선택 의존성

```powershell
pip install -r requirements-specialists.txt
```

전문 모델 plugin을 쓰지 않으면 이 파일의 패키지는 필요하지 않다.

### 6.3 환경 변수

권장 Bedrock cascade:

```env
AWS_REGION=us-east-1
LOW_MODEL=bedrock:us.anthropic.claude-haiku-4-5-20251001-v1:0
HIGH_MODEL=bedrock:us.anthropic.claude-sonnet-4-5-20250929-v1:0
```

직접 provider를 사용할 경우:

```env
OPENAI_API_KEY=...
ANTHROPIC_API_KEY=...
```

AWS Bedrock은 boto3가 인식할 수 있는 AWS credential 설정이 별도로 필요하다.

## 7. CLI 명세

기본 실행:

```powershell
python main.py
```

주요 옵션:

| 옵션 | 기본값 | 설명 |
| --- | --- | --- |
| `--img_dir` | `data/raw` | 입력 이미지 디렉터리 |
| `--out_dir` | `data/labeled` | 라벨과 metric 출력 디렉터리 |
| `--vis_dir` | `data/visualized` | 시각화 출력 디렉터리 |
| `--prompt` | 객체 탐지 기본 prompt | VLM에 전달할 사용자 지시문 |
| `--task_type` | `object_detection` | 라벨링 태스크 |
| `--threshold` | `0.75` | low 결과 consistency 에스컬레이션 기준 |
| `--low_model` | 환경 변수 또는 `gpt-4o-mini` | 초안 생성 VLM |
| `--high_model` | 환경 변수 또는 `gpt-4o` | 불확실 샘플 검증 VLM |
| `--inference_count` | `3` | low VLM 반복 추론 횟수 |
| `--draft_temperature` | `0.7` | low 반복 추론 temperature |
| `--allow_same_model` | `False` | low/high 동일 모델 허용 |
| `--label_formats` | `yolo` | 출력 포맷 목록 |
| `--custom_label_template` | 없음 | custom 출력 템플릿 |
| `--custom_label_extension` | `.json` | custom 출력 확장자 |
| `--plugin_config` | 없음 | 전문 모델 plugin 설정 JSON |
| `--plugin_fail_fast` | `False` | plugin 오류 시 즉시 중단 |
| `--gt_dir` | 없음 | YOLO ground truth 디렉터리 |
| `--eval_iou` | `0.5` | GT bbox 매칭 IoU 기준 |

비 detection 태스크에서 `--label_formats`를 생략하면 코드가 기본 `yolo` 대신 `vision_json`을 사용한다.

## 8. 초기화 과정

### 8.1 출력 디렉터리 생성

다음 디렉터리를 없으면 생성한다.

- `img_dir`
- `out_dir`
- `vis_dir`

입력 디렉터리도 자동 생성하므로 경로 오타가 있어도 빈 디렉터리가 생길 수 있다.

### 8.2 모델명 해석

low model 우선순위:

1. `--low_model`
2. `LOW_MODEL`
3. `AWS_BEDROCK_LOW_MODEL_ID`
4. `gpt-4o-mini`

high model 우선순위:

1. `--high_model`
2. `HIGH_MODEL`
3. `AWS_BEDROCK_HIGH_MODEL_ID`
4. legacy `AWS_BEDROCK_MODEL_ID`
5. `gpt-4o`

Bedrock 한쪽 모델만 설정된 경우 다른 쪽에 프로젝트 기본 Haiku/Sonnet ID를 보완한다.

### 8.3 Cascade 유효성 검사

low와 high model 이름이 같으면 기본적으로 실행을 중단한다.

```text
LOW_MODEL == HIGH_MODEL -> ERROR
```

`--allow_same_model`을 지정하면 디버깅 목적으로 계속 실행한다.

모델명에 Haiku/Sonnet/Opus가 포함된 경우 capacity 순서를 단순 추정한다.

```text
Haiku < Sonnet < Opus
```

low rank가 high rank보다 높거나 같으면 warning만 출력하고 실행은 계속한다.

### 8.4 출력 포맷 초기화

`LabelExportWriter`가 출력 포맷을 검증한다.

지원 값:

- `yolo`
- `pascal_voc`
- `coco`
- `vision_json`
- `custom`
- `all`

custom을 선택했는데 템플릿이 없으면 실행 전에 중단한다.

### 8.5 Plugin 초기화

`--plugin_config`가 있으면 다음 순서로 처리한다.

1. built-in registry 생성
2. JSON 설정 읽기
3. `enabled=false` plugin 제외
4. custom `module:ClassName` 동적 등록
5. plugin 인스턴스 생성
6. orchestrator 생성

전문 모델 weight는 이 단계에서 바로 로드하지 않는다. 각 plugin의 첫 `refine()` 호출에서 lazy loading한다.

### 8.6 API key 검사

- 모델명에 `gpt`가 포함되면 `OPENAI_API_KEY` 확인
- direct Anthropic Claude이면 `ANTHROPIC_API_KEY` 확인
- Bedrock Claude이면 위 두 API key는 검사하지 않음

필요한 key가 없으면 VLM client를 만들기 전에 실행을 종료한다.

## 9. 입력 이미지 처리

`main.py`가 직접 처리하는 이미지 확장자는 다음 세 가지다.

- `.png`
- `.jpg`
- `.jpeg`

파일명은 정렬한 뒤 처리한다. 이 정렬은 tracking plugin에서 프레임 순서를 결정하는 데 사용된다.

입력 이미지가 하나도 없으면 `src/utils/setup_samples.py`의 샘플 다운로드를 실행하고 목록을 다시 읽는다.

현재 `VisionLLMClient` 자체는 WebP media type을 인식하지만 `main.py` 이미지 목록 필터에는 `.webp`가 포함되어 있지 않다.

## 10. 태스크별 VLM Prompt

VLM에는 사용자 prompt와 시스템 prompt가 함께 전달된다.

공통 시스템 지시:

- vision labeling AI 역할
- 태스크별 JSON schema 준수
- 좌표는 `0.0~1.0`
- bbox는 `xmin < xmax`, `ymin < ymax`
- 설명 없이 JSON만 출력

### 10.1 Classification

```json
{
  "classifications": [
    {"label": "class_name", "confidence": 0.9}
  ]
}
```

### 10.2 Object Detection

```json
{
  "boxes": [
    {
      "label": "class_name",
      "xmin": 0.1,
      "ymin": 0.2,
      "xmax": 0.6,
      "ymax": 0.9,
      "confidence": 0.9
    }
  ]
}
```

### 10.3 Segmentation

```json
{
  "segments": [
    {
      "label": "class_name",
      "polygon": [
        {"x": 0.1, "y": 0.2},
        {"x": 0.4, "y": 0.2},
        {"x": 0.3, "y": 0.7}
      ],
      "confidence": 0.9
    }
  ]
}
```

### 10.4 Pose Estimation

```json
{
  "poses": [
    {
      "label": "person",
      "confidence": 0.9,
      "keypoints": [
        {
          "name": "nose",
          "x": 0.5,
          "y": 0.2,
          "visible": true,
          "confidence": 0.9
        }
      ]
    }
  ]
}
```

### 10.5 OCR

```json
{
  "texts": [
    {
      "text": "recognized text",
      "xmin": 0.1,
      "ymin": 0.2,
      "xmax": 0.7,
      "ymax": 0.3,
      "confidence": 0.9
    }
  ]
}
```

### 10.6 Tracking

```json
{
  "tracks": [
    {
      "track_id": "1",
      "frame_id": 0,
      "label": "person",
      "xmin": 0.1,
      "ymin": 0.2,
      "xmax": 0.6,
      "ymax": 0.9,
      "confidence": 0.9
    }
  ]
}
```

### 10.7 All

`all`은 다음 key를 한 응답에서 선택적으로 반환하도록 지시한다.

```json
{
  "classifications": [],
  "boxes": [],
  "segments": [],
  "poses": [],
  "texts": [],
  "tracks": []
}
```

## 11. VLM Provider 호출

### 11.1 OpenAI

- system message + user text + base64 image
- JSON object response format 요청
- 사용자 지정 temperature 적용

### 11.2 Direct Anthropic

- base64 image content
- `max_tokens=1024`
- system message
- 사용자 지정 temperature

### 11.3 AWS Bedrock Claude

- `bedrock-runtime.invoke_model()`
- `anthropic_version=bedrock-2023-05-31`
- `max_tokens=1024`
- base64 image + user prompt

현재 API timeout은 명시적으로 설정되어 있지 않다.

## 12. VLM 호출 재시도

각 `predict()`는 최대 3회 시도한다.

```text
1차 실패 -> 1초 대기
2차 실패 -> 2초 대기
3차 실패 -> 종료
```

코드는 `2 ** attempt` exponential backoff를 사용한다.

모든 시도가 실패하면 예외를 다시 발생시키지 않고 해당 태스크의 empty `DetectionResult`를 반환한다.

`api_attempts`, `successful_predictions`, `failed_predictions`가 client 단위로 누적된다.

## 13. JSON 추출 및 정규화

### 13.1 JSON 추출

응답이 fenced JSON이면 내부 본문을 추출한다.

````text
```json
{...}
```
````

JSON parsing에 실패하면 warning을 출력하고 `{"boxes": []}`를 반환한다. 비 detection 태스크에서도 결과적으로 empty result가 된다.

### 13.2 좌표 정규화

좌표 값이 `1.0`보다 크면 `0~1000` 좌표로 간주해 `1000`으로 나눈다.

```text
coord > 1.0 -> coord / 1000.0
```

그 후 `0.0~1.0`으로 clamp한다.

### 13.3 Confidence 정규화

confidence가 `1.0`보다 크면 percentage로 간주해 `100`으로 나눈다.

```text
confidence > 1.0 -> confidence / 100.0
```

그 후 `0.0~1.0`으로 clamp한다.

### 13.4 Invalid 항목 제거

- label 없는 classification 제외
- label 없는 bbox 제외
- 잘못된 bbox 순서 제외
- polygon point 3개 미만 제외
- keypoint name 없는 point 제외
- keypoint가 없는 pose 제외
- text 없는 OCR 항목 제외
- track id 또는 label 없는 tracking 항목 제외

## 14. Low VLM 반복 추론

`LabelingAgent`는 같은 이미지를 low VLM에 `inference_count`회 전달한다.

기본값:

```text
inference_count = 3
temperature = 0.7
```

temperature를 0보다 높게 두는 이유는 반복 결과의 다양성을 발생시켜 모델이 안정적으로 같은 라벨을 생성하는지 측정하기 위해서다.

각 결과의 `source_model`에는 low model 이름을 기록한다.

## 15. Self-Consistency 계산

`get_consistency_score()`는 모든 low 결과 쌍을 비교한다.

3회 추론이면 다음 3쌍을 사용한다.

```text
result 0 vs result 1
result 0 vs result 2
result 1 vs result 2
```

최종 consistency는 pairwise score 평균이다.

### 15.1 Detection

같은 label의 bbox끼리 greedy matching하고 IoU가 `0.5` 이상인 match만 합산한다.

```text
pair score = matched IoU sum / max(box count A, box count B)
```

따라서 누락 또는 추가 bbox도 score를 낮춘다.

### 15.2 Segmentation

polygon 자체의 mask IoU가 아니라 polygon의 bounding box를 만든 뒤 bbox IoU를 계산한다.

### 15.3 Box와 Segment 교차 비교

한 결과는 bbox, 다른 결과는 segment인 경우 segment bounding box와 bbox를 비교한다.

### 15.4 Classification, Pose, OCR, Tracking

다음 문자열 집합의 Jaccard similarity를 사용한다.

- classification label
- pose label
- visible keypoint name
- OCR text
- `track_id:label`

```text
Jaccard = intersection size / union size
```

위 방식은 태스크별 전문 metric의 근사치이며 pose keypoint 거리, OCR edit distance, tracking ID switch 등을 측정하지 않는다.

## 16. Uncertainty 계산

초안 평균 confidence는 첫 번째 low 결과만 사용한다.

```text
draft_confidence = mean confidence of results[0]
```

pose는 pose confidence와 각 keypoint confidence를 모두 평균에 포함한다.

신뢰도와 uncertainty:

```text
reliability = (consistency + confidence) / 2
uncertainty = 1 - reliability
```

현재 두 항목은 동일한 가중치 `0.5`를 사용한다.

## 17. High VLM 에스컬레이션

결정 규칙:

```text
consistency < threshold -> High VLM
consistency >= threshold -> Low 결과 사용
```

### 17.1 Consistent 경로

- 첫 번째 low 결과를 최종 결과로 선택
- status: `Consistent`
- source model: low model

반복 결과들을 ensemble하거나 평균 bbox를 계산하지 않는다.

### 17.2 Escalated 경로

- high VLM을 temperature `0.0`으로 1회 호출
- status: `Escalated`
- source model: high model

high 결과의 `consistency_score`에는 low 반복 결과의 consistency가 들어간다.

high 결과의 `mean_confidence`는 high 결과로 다시 계산한다.

현재 `uncertainty_score`는 high 결과 confidence로 다시 계산하지 않고, low consistency와 첫 low 결과 confidence로 계산한 값을 유지한다.

## 18. 전문 모델 Plugin 단계

plugin 설정이 없으면 이 단계는 생략된다.

설정 예시:

```powershell
python main.py `
  --task_type segmentation `
  --label_formats coco,vision_json `
  --plugin_config configs/plugins.example.json
```

### 18.1 실행 조건

orchestrator는 설정 순서대로 plugin을 순회하고 현재 `task_type`을 지원하는 plugin만 실행한다.

`task_type=all`이면 설정된 모든 plugin이 실행 대상이다.

### 18.2 Built-in Plugin

| plugin | 기본 모델/엔진 | 역할 |
| --- | --- | --- |
| `classification` | `openai/clip-vit-base-patch32` | zero-shot classification |
| `grounding_dino` | `IDEA-Research/grounding-dino-tiny` | text-conditioned bbox |
| `sam` | `sam2_b.pt` | seed bbox 기반 polygon mask |
| `pose` | `yolo11n-pose.pt` | pose keypoint |
| `ocr` | EasyOCR | text detection/recognition |
| `tracking` | `yolo11n.pt` + ByteTrack | frame sequence tracking |

모델 파일은 repository에 포함되지 않으며 최초 plugin 실행 시 library가 다운로드할 수 있다.

### 18.3 Plugin 입력

모든 plugin은 다음 값을 받는다.

```text
image_path
user prompt
현재까지의 seed DetectionResult
```

Grounding DINO는 설정의 `labels`와 VLM 결과의 classification/box/segment/pose label을 후보 text prompt로 사용한다.

SAM은 seed bbox가 필요하다. 따라서 segmentation chain에서 Grounding DINO가 SAM보다 먼저 있어야 한다.

### 18.4 Plugin 출력

plugin은 `PluginOutput`을 반환한다.

```text
result: DetectionResult
score: optional confidence
metadata: model/provenance
```

score가 없으면 seed와 plugin 결과의 agreement를 score로 사용한다.

### 18.5 결과 병합

- classification: label 기준, confidence가 높은 값 사용
- bbox: 같은 label이고 IoU `0.5` 이상이면 plugin 좌표로 교체하고 confidence 평균
- segment: label과 근사 centroid 기준
- pose: label과 근사 keypoint centroid 기준
- OCR: text와 근사 위치 기준
- tracking: `(frame_id, track_id)` 기준

plugin마다 다음 값이 기록된다.

- `plugin_scores`
- `plugin_metadata`
- `plugin_records`

### 18.6 Plugin 이후 신뢰도 갱신

plugin score를 설정의 weight로 가중 평균한다.

```text
plugin_score = weighted mean(plugin scores)
new_consistency = (previous_consistency + plugin_score) / 2
new_uncertainty = 1 - ((new_consistency + merged_mean_confidence) / 2)
```

`source_model`에는 성공한 plugin 이름이 추가된다.

```text
bedrock:...+grounding_dino+sam
```

### 18.7 Plugin 오류 처리

기본 동작:

- 오류를 `plugin_records`에 기록
- 다음 plugin 및 다음 이미지 처리 계속

`--plugin_fail_fast` 사용 시 plugin 오류를 다시 발생시켜 현재 이미지의 처리 흐름을 중단한다. 바깥 이미지별 예외 처리에서 오류를 출력하고 다음 이미지로 이동한다.

### 18.8 현재 Plugin 에스컬레이션 제한

high VLM 에스컬레이션 결정은 plugin 실행 전에 low VLM consistency만으로 완료된다.

따라서 plugin이 VLM과 크게 불일치해 최종 uncertainty가 높아져도 high VLM을 두 번째로 호출하지 않는다.

현재 순서:

```text
Low VLM 반복 -> High 여부 결정 -> Optional Plugins
```

향후 권장 순서:

```text
Low VLM 반복 -> Specialist agreement -> 통합 uncertainty -> High VLM 결정
```

## 19. Dataset Insight Agent

최종 병합 결과에서 label을 누적한다.

- classification label
- bbox label
- segment label
- pose label
- OCR은 `text`라는 공통 label
- tracking label

클래스별 count와 percentage를 출력한다.

가장 많은 클래스와 가장 적은 클래스 비율이 `3`보다 크면 imbalance warning과 희소 클래스 데이터 수집/증강 제안을 출력한다.

이 분석은 단순 frequency 기반이며 맥락적 불균형이나 이미지별 co-occurrence는 분석하지 않는다.

## 20. 라벨 저장

이미지별 final result를 `LabelExportWriter`에 전달한다.

### 20.1 YOLO

- object detection bbox만 표현
- 이미지별 `.txt`
- dataset 단위 `classes.txt`

### 20.2 Pascal VOC

- object detection bbox만 표현
- 이미지별 `.xml`

### 20.3 COCO

- bbox annotation
- polygon segmentation
- dataset 단위 `coco_annotations.json`

### 20.4 Vision JSONL

다음 전체 데이터를 보존한다.

- task type
- VLM/plugin source
- consistency/confidence/uncertainty
- plugin score/metadata
- classification
- bbox
- segmentation
- pose
- OCR
- tracking

### 20.5 Custom

사용자 템플릿의 `{result_json}` 등을 치환해 저장한다.

상세 포맷 명세는 `transform_label_format.md`를 참조한다.

## 21. 시각화

`visualize_boxes()`가 원본 이미지에 다음 요소를 그린다.

- bbox와 label/confidence
- segmentation polygon
- pose keypoint와 name
- OCR text region과 문자열
- tracking bbox와 `track_id:label`
- source model과 uncertainty header

결과 파일명:

```text
vis_<원본 이미지 파일명>
```

## 22. 이미지별 Metric

`run_metrics.csv`와 `run_metrics.jsonl`에 다음 정보를 기록한다.

| 필드 | 설명 |
| --- | --- |
| `image` | 이미지 파일명 |
| `status` | `Consistent` 또는 `Escalated` |
| `source_model` | VLM 및 성공 plugin |
| `low_model` | low VLM |
| `high_model` | high VLM |
| `task_type` | 태스크 |
| `objects` | 전체 라벨 수 |
| `boxes` | bbox 개수 |
| `segments` | segment 개수 |
| `poses` | pose 개수 |
| `texts` | OCR text 개수 |
| `tracks` | track 개수 |
| `classifications` | classification 개수 |
| `consistency_score` | VLM/plugin 일관성 |
| `mean_confidence` | 결과 평균 confidence |
| `uncertainty_score` | 최종 uncertainty |
| `plugin_scores` | plugin별 score JSON |
| `plugin_records` | plugin 성공/실패 JSON |
| `low_api_attempts` | 이미지별 low API 시도 수 |
| `high_api_attempts` | 이미지별 high API 시도 수 |
| `elapsed_sec` | 처리 시간 |
| `label_path` | 대표 출력 경로 |
| `label_paths` | 포맷별 출력 경로 JSON |
| `visualization_path` | 시각화 경로 |

## 23. 실행 Summary

`run_summary.json`은 다음 실행 단위 정보를 저장한다.

- 이미지 수
- 태스크와 출력 포맷
- 전체 라벨 수
- 전체/평균 처리 시간
- 추정 수동 라벨링 시간
- 시간 절감률
- low/high model
- low/high API attempts
- escalation count/rate
- 단순 cost reduction percentage
- plugin 목록
- optional GT evaluation

## 24. 시간 및 비용 지표

수동 라벨링 기준은 코드에 이미지당 `45초`로 고정되어 있다.

```text
manual time = image count * 45 seconds
time saved % = (manual time - pipeline time) / manual time * 100
```

단순 API cost saving:

```text
cost reduction % = non-escalated image count / total image count * 100
```

이 값은 high VLM 호출을 피한 이미지 비율이며 실제 화폐 비용이 아니다. low model 반복 호출 비용, token 수, provider별 단가, 전문 모델 GPU 비용을 반영하지 않는다.

실제 상대 비용 비교는 `evaluate_experiments.py`의 `low_unit_cost`, `high_unit_cost`를 사용한다.

## 25. Ground Truth 평가

조건:

- `--gt_dir`가 존재
- 출력 포맷에 `yolo` 포함

평가 지표:

- true positive
- false positive
- false negative
- precision
- recall
- F1
- mean matched IoU

class id가 같고 IoU가 `--eval_iou` 이상인 bbox를 greedy matching한다.

현재 GT 평가는 object detection YOLO에만 적용된다. classification accuracy, segmentation mask IoU, pose OKS/PCK, OCR CER/WER, tracking MOTA/IDF1은 구현되어 있지 않다.

## 26. 호출량

기본 성공 경로:

```text
low VLM 3회
consistency 미달 시 high VLM 1회
```

각 VLM 호출은 최대 3회 retry할 수 있다.

기본 최악 API attempt:

```text
(low 3회 + high 1회) * retry 3회 = 이미지당 최대 12 attempts
```

전문 모델 plugin은 API attempts에 포함되지 않으며 별도 로컬 추론 비용이 발생할 수 있다.

## 27. 오류 처리

### 27.1 실행 전 중단

- low/high 모델이 같고 `--allow_same_model`이 없음
- 출력 포맷 설정 오류
- custom 템플릿 누락
- plugin config 오류
- 필요한 API key 누락
- VLM client 초기화 실패

### 27.2 이미지별 오류

이미지 처리 전체가 `try/except`로 감싸져 있다.

한 이미지에서 오류가 발생하면:

1. 이미지 이름과 오류 출력
2. 해당 이미지 metric은 기록하지 않음
3. 다음 이미지 처리 계속

### 27.3 Empty VLM 결과

모든 VLM retry가 실패하거나 JSON parsing이 실패하면 empty result가 반환될 수 있다.

직접 실행하는 `main.py` 경로에는 `label_validator.py`가 연결되어 있지 않으므로 empty result도 빈 라벨 파일로 저장될 수 있다. LangGraph의 `agentic_workflow.py` 경로는 validation 및 repair node를 거친다.

## 28. 태스크별 실행 예시

### 28.1 VLM-only Object Detection

```powershell
python main.py `
  --task_type object_detection `
  --label_formats yolo,coco `
  --prompt "Detect people, vehicles, and animals."
```

### 28.2 Grounding DINO Detection

```powershell
python main.py `
  --task_type object_detection `
  --label_formats yolo,coco,vision_json `
  --plugin_config configs/plugins.example.json
```

### 28.3 DINO + SAM Segmentation

```powershell
python main.py `
  --task_type segmentation `
  --label_formats coco,vision_json `
  --plugin_config configs/plugins.example.json
```

### 28.4 Pose Estimation

```powershell
python main.py `
  --task_type pose_estimation `
  --label_formats vision_json `
  --plugin_config configs/plugins.example.json
```

### 28.5 OCR

```powershell
python main.py `
  --task_type ocr `
  --label_formats vision_json `
  --prompt "Read all visible Korean and English text." `
  --plugin_config configs/plugins.example.json
```

### 28.6 Tracking

```powershell
python main.py `
  --img_dir data/frames `
  --task_type tracking `
  --label_formats vision_json `
  --plugin_config configs/plugins.example.json
```

tracking은 정렬된 이미지 파일을 frame sequence로 처리한다. 비디오 파일 직접 입력은 지원하지 않는다.

## 29. Ablation 실험

### 29.1 Low-only

```powershell
python main.py `
  --out_dir data/runs/low `
  --vis_dir data/runs/low_vis `
  --threshold -1 `
  --gt_dir data/ground_truth
```

consistency가 `-1`보다 작을 수 없으므로 high escalation이 발생하지 않는다.

### 29.2 High-only 근사

```powershell
python main.py `
  --out_dir data/runs/high `
  --vis_dir data/runs/high_vis `
  --low_model <HIGH_MODEL> `
  --high_model <HIGH_MODEL> `
  --allow_same_model `
  --inference_count 1 `
  --threshold -1 `
  --gt_dir data/ground_truth
```

### 29.3 Cascade

```powershell
python main.py `
  --out_dir data/runs/cascade `
  --vis_dir data/runs/cascade_vis `
  --threshold 0.75 `
  --gt_dir data/ground_truth
```

### 29.4 Plugin Cascade

```powershell
python main.py `
  --out_dir data/runs/plugin_cascade `
  --vis_dir data/runs/plugin_vis `
  --threshold 0.75 `
  --plugin_config configs/plugins.example.json `
  --gt_dir data/ground_truth
```

비교 리포트:

```powershell
python evaluate_experiments.py `
  --runs low=data/runs/low high=data/runs/high cascade=data/runs/cascade plugin=data/runs/plugin_cascade `
  --gt_dir data/ground_truth `
  --out_dir data/reports
```

## 30. 현재 제한 사항

- 전문 모델 weight는 repository에 포함되지 않음
- 전문 모델의 실제 추론은 설치 환경과 checkpoint availability에 의존
- plugin disagreement가 high VLM 재호출을 유발하지 않음
- low 반복 결과 중 첫 번째 결과만 최종 초안으로 사용
- high escalation 후 uncertainty가 high confidence 기준으로 재계산되지 않음
- segmentation consistency는 true mask IoU가 아니라 polygon bounding box IoU
- pose consistency는 OKS/PCK가 아니라 label/keypoint name Jaccard
- OCR consistency는 CER/WER가 아니라 exact text set Jaccard
- tracking consistency는 MOT metric이 아니라 `track_id:label` set Jaccard
- `task_type=all`은 모든 plugin을 실행하므로 비용과 시간이 크게 증가할 수 있음
- WebP는 client가 인코딩 가능하지만 main image 목록에서 제외됨
- 비디오 파일 직접 tracking 미지원
- 자동 라벨 생성 경로에 구조 validation/strict skip 미연결
- API timeout 미설정
- 처리 병렬화 없음
- prompt 결과의 클래스 taxonomy 고정/매핑 기능 없음
- 실제 비용 계산에 token/GPU 비용 미반영

## 31. 권장 개선 방향

1. specialist agreement를 high VLM escalation 전에 계산한다.
2. plugin/VLM 결과를 GT로 calibration해 태스크별 weight를 학습한다.
3. detection은 mAP, segmentation은 mask IoU/Dice를 추가한다.
4. pose는 OKS/PCK, OCR은 CER/WER, tracking은 MOTA/IDF1을 추가한다.
5. empty result와 invalid geometry를 `label_validator.py`로 검사한다.
6. low 반복 결과를 consensus ensemble로 통합한다.
7. high VLM 결과 이후 uncertainty를 재계산한다.
8. 비디오 decoder와 frame batching을 추가한다.
9. API timeout/rate limit/concurrency를 명시적으로 관리한다.
10. 모델별 token/API/GPU 비용을 실제 단가 기반으로 기록한다.

## 32. 관련 문서

| 문서 | 설명 |
| --- | --- |
| `README.md` | 프로젝트 설치 및 일반 사용법 |
| `transform_label_format.md` | 기존 라벨 포맷 변환 상세 명세 |
| `agentic_workflow.md` | LangGraph 상위 orchestrator 상세 명세 |
| `proposal.md` | 연구 제안과 평가 계획 |
| `abstract.md` | 논문 초록 |
| `changelog.md` | 변경 이력 |
