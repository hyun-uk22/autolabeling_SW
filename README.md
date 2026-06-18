# Agentic Auto-Labeling System
- 소프트웨어 중심대학 프로젝트
- 이미지 폴더를 입력으로 받아 AWS Bedrock Claude Vision 모델로 객체를 탐지하고, 결과를 YOLO, Pascal VOC, COCO, 사용자 정의 라벨 형식과 시각화 이미지로 저장하는 자동 라벨링 실험용 프로젝트입니다. 저용량 Claude 모델로 여러 번 초안을 생성한 뒤 결과 일관성이 낮으면 고용량 Claude 모델로 에스컬레이션하는 계층형 검증 구조를 사용합니다.

## 주요 기능

- classification, object detection, segmentation, pose estimation, OCR, tracking 라벨링
- LLM 출력 JSON을 내부 `DetectionResult` 구조로 정규화
- YOLO, Pascal VOC XML, COCO JSON, 범용 Vision JSONL, 사용자 정의 템플릿 라벨 파일 생성
- 바운딩 박스가 그려진 시각화 이미지 생성
- 저용량 모델 반복 추론 기반 consistency score 계산
- consistency가 임계값보다 낮을 때 고용량 모델로 재추론
- 실행별 처리 시간, 객체 수, confidence, uncertainty 기록
- 선택적으로 ground truth YOLO 라벨과 precision/recall 평가
- 전체 라벨 분포를 요약하고 클래스 불균형을 간단히 리포트

## 프로젝트 구조

```text
.
├── main.py                         # 실행 엔트리포인트
├── requirements.txt                # Python 의존성
├── data/
│   ├── raw/                        # 입력 이미지
│   ├── labeled/                    # 출력 라벨 및 메트릭
│   └── visualized/                 # 출력 시각화 이미지
└── src/
    ├── agents/
    │   ├── labeling_agent.py       # 저용량 모델 반복 추론
    │   ├── verification_agent.py   # consistency 검증 및 에스컬레이션
    │   └── insight_agent.py        # 라벨 분포 리포트
    ├── core/
    │   ├── llm_client.py           # Bedrock Claude, OpenAI, Anthropic Vision LLM 호출
    │   └── models.py               # DetectionResult, BoundingBox 모델
    └── utils/
        ├── format_converter.py     # YOLO/Pascal VOC/COCO/custom 포맷 저장
        ├── visualize.py            # 바운딩 박스 시각화
        ├── geometry.py             # IoU 및 consistency 계산
        ├── evaluation.py           # YOLO ground truth 평가
        └── setup_samples.py        # 샘플 이미지 다운로드
```

## 환경 구성

Python 가상환경을 만든 뒤 의존성을 설치합니다.

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

macOS/Linux에서는 활성화 명령만 다릅니다.

```bash
source .venv/bin/activate
```

## 빠른 시작

AWS Bedrock Claude 기반 실험을 기준으로 하면, 먼저 `.env.example`을 참고해 프로젝트 루트의 `.env`를 아래처럼 설정합니다.

```env
AWS_REGION=us-east-1
LOW_MODEL=bedrock:us.anthropic.claude-haiku-4-5-20251001-v1:0
HIGH_MODEL=bedrock:us.anthropic.claude-sonnet-4-5-20250929-v1:0
```

그 다음 실행합니다.

```bash
python main.py
```

현재 `.env`에서 `LOW_MODEL`과 `HIGH_MODEL`이 같은 값이면 실행 전에 중단됩니다. 논문 주장인 heterogeneous capacity cascade를 만족하려면 두 모델이 달라야 합니다.

## 환경 변수

프로젝트 루트에 `.env` 파일을 만들고 사용할 모델을 설정합니다. AWS Bedrock Claude 사용 시 별도의 `OPENAI_API_KEY`나 `ANTHROPIC_API_KEY`는 필요하지 않습니다. 대신 AWS credential은 로컬 AWS CLI/환경 변수/프로파일 등 boto3가 인식할 수 있는 방식으로 설정되어 있어야 합니다.

권장 설정:

```env
AWS_REGION=us-east-1
LOW_MODEL=bedrock:us.anthropic.claude-haiku-4-5-20251001-v1:0
HIGH_MODEL=bedrock:us.anthropic.claude-sonnet-4-5-20250929-v1:0
```

모델 선택 규칙:

- `LOW_MODEL`: 초안 라벨 생성을 담당하는 저용량 모델입니다. 논문 실험에서는 Claude Haiku 4.5를 권장합니다.
- `HIGH_MODEL`: consistency가 낮을 때 사용하는 고용량 검증 모델입니다. 논문 실험에서는 Claude Sonnet 4.5를 권장합니다.
- AWS Bedrock Claude 기준 권장 조합은 `Claude Haiku 4.5 -> Claude Sonnet 4.5`입니다.
- `AWS_BEDROCK_LOW_MODEL_ID`, `AWS_BEDROCK_HIGH_MODEL_ID`를 raw Bedrock model id로 지정해도 됩니다. 이 경우 `bedrock:` prefix는 코드가 자동으로 붙입니다.
- `AWS_BEDROCK_MODEL_ID`는 legacy 호환용 high model fallback으로 사용됩니다.
- `LOW_MODEL`과 `HIGH_MODEL`이 같으면 기본적으로 실행을 중단합니다. 논문 주장인 heterogeneous capacity cascade를 만족하지 않기 때문입니다.
- 디버깅 목적으로 같은 모델을 허용하려면 `--allow_same_model`을 사용합니다.
- 모델명이 `gpt`를 포함하면 `OPENAI_API_KEY`가 필요합니다. 현재 논문 실험용 기본 경로는 아닙니다.
- 모델명이 `claude`를 포함하고 Bedrock 모델이 아니면 `ANTHROPIC_API_KEY`가 필요합니다. 현재 논문 실험용 기본 경로는 아닙니다.

## 실행 방법

기본 실행:

```bash
python main.py
```

커스텀 경로와 옵션을 지정하는 예시:

```bash
python main.py ^
  --img_dir data/raw ^
  --out_dir data/labeled ^
  --vis_dir data/visualized ^
  --threshold 0.75 ^
  --low_model bedrock:us.anthropic.claude-haiku-4-5-20251001-v1:0 ^
  --high_model bedrock:us.anthropic.claude-sonnet-4-5-20250929-v1:0 ^
  --prompt "Detect and classify all prominent objects in this image. Output strictly as JSON."
```

ground truth 라벨이 있을 때 평가까지 실행하는 예시:

```bash
python main.py --gt_dir data/ground_truth --eval_iou 0.5
```

같은 모델로 디버깅만 하고 싶을 때:

```bash
python main.py --allow_same_model
```

이 옵션은 논문 실험 결과 산출용으로는 권장하지 않습니다.

## CLI 옵션

| 옵션 | 기본값 | 설명 |
| --- | --- | --- |
| `--img_dir` | `data/raw` | 입력 이미지 폴더 |
| `--out_dir` | `data/labeled` | 선택한 라벨 포맷과 실행 메트릭 저장 폴더 |
| `--vis_dir` | `data/visualized` | 바운딩 박스 시각화 이미지 저장 폴더 |
| `--prompt` | `Detect and classify all prominent objects in this image. Output strictly as JSON.` | Vision LLM에 전달할 라벨링 지시문 |
| `--task_type` | `object_detection` | 라벨링 태스크. `classification`, `object_detection`, `segmentation`, `pose_estimation`, `ocr`, `tracking`, `all` |
| `--threshold` | `0.75` | consistency가 이 값보다 낮으면 고용량 모델로 에스컬레이션 |
| `--low_model` | 환경 변수 또는 `gpt-4o-mini` | 초안 라벨 생성을 담당하는 저용량 모델. AWS 실험에서는 Claude Haiku 4.5 권장 |
| `--high_model` | 환경 변수 또는 `gpt-4o` | 불확실 샘플을 재검증하는 고용량 모델. AWS 실험에서는 Claude Sonnet 4.5 권장 |
| `--inference_count` | `3` | consistency 계산을 위한 저용량 모델 반복 추론 횟수 |
| `--draft_temperature` | `0.7` | 반복 추론 다양성을 위한 초안 모델 temperature |
| `--allow_same_model` | `False` | 같은 low/high 모델을 디버깅용으로 허용 |
| `--gt_dir` | `None` | 선택 입력. ground truth YOLO 라벨 폴더 |
| `--eval_iou` | `0.5` | ground truth 평가 시 IoU 매칭 기준 |
| `--label_formats` | `yolo` | 출력 라벨 형식. `yolo`, `pascal_voc`, `coco`, `vision_json`, `custom`, `all`을 쉼표로 조합 |
| `--custom_label_template` | `None` | custom 라벨 출력에 사용할 템플릿 파일 |
| `--custom_label_extension` | `.json` | custom 라벨 파일 확장자 |

## 입력 형식

### 이미지 입력

`--img_dir` 폴더에 이미지 파일을 넣습니다.

지원 확장자:

- `.jpg`
- `.jpeg`
- `.png`

예시:

```text
data/raw/
├── 01_dog_bike_car.jpg
├── 02_eagle.jpg
└── 03_person_horse.jpg
```

입력 이미지가 없으면 `src/utils/setup_samples.py`의 샘플 이미지 다운로드 함수가 자동으로 실행됩니다.

### LLM 응답 JSON 형식

LLM은 `--task_type`에 따라 아래 JSON key를 반환하도록 프롬프트됩니다.

| task_type | 기대 JSON key | 설명 |
| --- | --- | --- |
| `classification` | `classifications` | 이미지 단위 label/confidence |
| `object_detection` | `boxes` | 객체별 bbox label/confidence |
| `segmentation` | `segments` | 객체별 polygon label/confidence |
| `pose_estimation` | `poses` | 인스턴스별 keypoints |
| `ocr` | `texts` | 텍스트 영역 bbox와 인식 문자열 |
| `tracking` | `tracks` | frame_id, track_id, bbox |
| `all` | 위 key 전체 | 이미지에서 가능한 라벨을 복합 추출 |

기본 `object_detection`은 아래 JSON 형태를 반환하도록 프롬프트됩니다.

```json
{
  "boxes": [
    {
      "label": "person",
      "xmin": 0.12,
      "ymin": 0.18,
      "xmax": 0.44,
      "ymax": 0.92,
      "confidence": 0.87
    }
  ]
}
```

좌표와 confidence는 내부에서 보정됩니다.

- 좌표가 `1.0`보다 크면 `0~1000` 스케일로 보고 `1000`으로 나눕니다.
- confidence가 `1.0`보다 크면 `0~100` 스케일로 보고 `100`으로 나눕니다.
- 최종 값은 `0.0~1.0` 범위로 clamp됩니다.
- `xmin < xmax`, `ymin < ymax` 조건을 만족하지 않는 박스는 제외됩니다.

### Ground Truth 입력

`--gt_dir`를 사용할 경우, ground truth 라벨은 YOLO 형식이어야 합니다.

```text
<class_id> <x_center> <y_center> <width> <height>
```

예시:

```text
0 0.512000 0.433000 0.320000 0.510000
1 0.245000 0.670000 0.180000 0.220000
```

파일명은 예측 라벨과 같아야 합니다.

```text
data/ground_truth/
├── 01_dog_bike_car.txt
├── 02_eagle.txt
└── 03_person_horse.txt
```

## 출력 형식

`--label_formats`로 여러 포맷을 동시에 내보낼 수 있습니다.

```bash
python main.py --label_formats all
python main.py --label_formats yolo,pascal_voc,coco
python main.py --task_type segmentation --label_formats vision_json,coco
python main.py --task_type pose_estimation --label_formats vision_json
python main.py --label_formats custom --custom_label_template templates/my_label.json --custom_label_extension .json
```

`all`은 기본 제공 포맷인 YOLO, Pascal VOC, COCO, Vision JSONL을 생성합니다. custom은 템플릿을 명시했을 때만 추가됩니다. `--task_type`이 `object_detection`이 아닌데 `--label_formats`를 지정하지 않으면 기본 `yolo` 대신 `vision_json`으로 저장합니다.

### YOLO 라벨

각 이미지마다 `--out_dir`에 같은 basename의 `.txt` 파일이 생성됩니다.

```text
data/labeled/
├── 01_dog_bike_car.txt
├── 02_eagle.txt
├── classes.txt
├── run_metrics.csv
└── run_metrics.jsonl
```

YOLO 라벨 형식:

```text
<class_id> <x_center> <y_center> <width> <height>
```

예시:

```text
0 0.512000 0.433000 0.320000 0.510000
1 0.245000 0.670000 0.180000 0.220000
```

`classes.txt`에는 실행 중 발견된 클래스명이 class id 순서대로 저장됩니다.

```text
person
dog
car
```

### Pascal VOC XML

`--label_formats pascal_voc` 또는 `all`을 사용하면 각 이미지마다 같은 basename의 `.xml` 파일이 생성됩니다.

```text
data/labeled/
├── 01_dog_bike_car.xml
└── 02_eagle.xml
```

좌표는 이미지 크기를 기준으로 복원된 pixel 단위 `xmin`, `ymin`, `xmax`, `ymax`입니다.

### COCO JSON

`--label_formats coco` 또는 `all`을 사용하면 전체 데이터셋 단위의 `coco_annotations.json`이 생성됩니다.

```text
data/labeled/
└── coco_annotations.json
```

COCO 구조는 `images`, `annotations`, `categories`를 포함합니다. bbox는 pixel 단위 `[x, y, width, height]`입니다. segmentation 태스크의 polygon은 COCO `segmentation` 필드에도 저장됩니다.

### Vision JSONL

`--label_formats vision_json` 또는 `all`을 사용하면 모든 태스크를 손실 없이 담는 `vision_annotations.jsonl`이 생성됩니다.

```text
data/labeled/
└── vision_annotations.jsonl
```

이 포맷은 classification, detection, segmentation, pose, OCR, tracking 결과를 모두 보존하기 위한 프로젝트 공통 포맷입니다. YOLO/Pascal VOC처럼 특정 태스크에 제한된 포맷으로 표현하기 어려운 라벨은 이 포맷을 사용합니다.

### 사용자 정의 라벨

`--label_formats custom`은 템플릿 파일을 읽어 이미지별 라벨 파일을 생성합니다. 템플릿에는 Python `str.format` placeholder를 사용할 수 있습니다. JSON/XML처럼 중괄호를 리터럴로 써야 하는 포맷은 `{{`와 `}}`로 이스케이프합니다.

사용 가능한 주요 placeholder:

| placeholder | 설명 |
| --- | --- |
| `{image_name}` | 이미지 파일명 |
| `{image_path}` | 이미지 경로 |
| `{image_width}` | 이미지 너비 |
| `{image_height}` | 이미지 높이 |
| `{source_model}` | 최종 라벨 생성 모델 |
| `{object_count}` | 객체 수 |
| `{objects_json}` | label, class_id, confidence, normalized/pixel/yolo/coco 좌표를 담은 JSON 배열 |
| `{boxes_json}` | normalized bbox 배열 |
| `{labels_json}` | label 배열 |
| `{result_json}` | classification, boxes, segments, poses, texts, tracks를 모두 포함한 전체 결과 JSON |

예시 템플릿:

```json
{{
  "image": "{image_name}",
  "width": {image_width},
  "height": {image_height},
  "objects": {objects_json}
}}
```

### 시각화 이미지

`--vis_dir`에 `vis_<원본파일명>` 형태로 저장됩니다.

```text
data/visualized/
├── vis_01_dog_bike_car.jpg
└── vis_02_eagle.jpg
```

이미지에는 다음 정보가 포함됩니다.

- 바운딩 박스
- segmentation polygon
- pose keypoint
- OCR/text 영역
- track id
- 클래스명
- confidence
- 사용 모델명
- uncertainty score

### 실행 메트릭

`run_metrics.csv`와 `run_metrics.jsonl`에는 이미지별 실행 결과가 저장됩니다.

필드:

| 필드 | 설명 |
| --- | --- |
| `image` | 처리한 이미지 파일명 |
| `status` | `Consistent` 또는 `Escalated` |
| `source_model` | 최종 라벨 생성에 사용된 모델 |
| `low_model` | 초안 라벨 생성 모델 |
| `high_model` | 에스컬레이션 검증 모델 |
| `task_type` | 실행한 라벨링 태스크 |
| `objects` | 태스크별 전체 라벨 수 |
| `boxes` | bbox 라벨 수 |
| `segments` | segmentation 라벨 수 |
| `poses` | pose 인스턴스 수 |
| `texts` | OCR/text 라벨 수 |
| `tracks` | tracking 라벨 수 |
| `classifications` | classification 라벨 수 |
| `consistency_score` | 반복 추론 결과 간 IoU 기반 일관성 |
| `mean_confidence` | 최종 또는 초안 결과의 평균 confidence |
| `uncertainty_score` | `1 - ((consistency + confidence) / 2)` |
| `low_api_attempts` | 해당 이미지에서 발생한 low model API 요청 시도 수 |
| `high_api_attempts` | 해당 이미지에서 발생한 high model API 요청 시도 수 |
| `elapsed_sec` | 이미지 처리 시간 |
| `label_path` | 대표 라벨 파일 경로. YOLO가 있으면 YOLO 경로 |
| `label_paths` | 포맷별 라벨 파일 경로 JSON |
| `visualization_path` | 시각화 이미지 파일 경로 |

JSONL 예시:

```json
{"image": "01_dog_bike_car.jpg", "status": "Consistent", "source_model": "bedrock:us.anthropic.claude-haiku-4-5-20251001-v1:0", "low_model": "bedrock:us.anthropic.claude-haiku-4-5-20251001-v1:0", "high_model": "bedrock:us.anthropic.claude-sonnet-4-5-20250929-v1:0", "objects": 3, "consistency_score": 0.82, "mean_confidence": 0.88, "uncertainty_score": 0.15, "low_api_attempts": 3, "high_api_attempts": 0, "elapsed_sec": 4.21, "label_path": "data/labeled/01_dog_bike_car.txt", "label_paths": "{\"yolo\": \"data/labeled/01_dog_bike_car.txt\", \"pascal_voc\": \"data/labeled/01_dog_bike_car.xml\", \"coco\": \"data/labeled/coco_annotations.json\"}", "visualization_path": "data/visualized/vis_01_dog_bike_car.jpg"}
```

## 처리 흐름

1. `main.py`가 CLI 옵션을 읽고 `--img_dir`, `--out_dir`, `--vis_dir` 폴더를 준비합니다.
2. `LOW_MODEL`, `HIGH_MODEL`, Bedrock fallback 환경 변수를 해석하고, 두 모델이 같은 경우 기본적으로 실행을 중단합니다.
3. `--task_type`과 `--label_formats`를 검증하고 `LabelExportWriter`를 초기화합니다. custom 포맷은 `--custom_label_template`이 없으면 실행 전에 중단됩니다.
4. 필요한 API key를 확인한 뒤 `VisionLLMClient`를 low/high 모델용으로 각각 생성합니다.
5. 입력 이미지 목록을 읽습니다. 이미지가 없으면 샘플 다운로드 함수를 실행한 뒤 다시 목록을 읽습니다.
6. 각 이미지마다 `HierarchicalVerificationAgent`가 처리합니다.
7. `LabelingAgent`가 저용량 모델로 같은 이미지를 `--inference_count`회 추론합니다. 이때 `--task_type`에 맞는 JSON 스키마를 LLM system prompt에 넣습니다.
8. `geometry.py`가 반복 결과의 consistency를 계산하고, 첫 번째 초안의 평균 confidence와 결합해 uncertainty를 계산합니다. bbox는 IoU, segmentation은 polygon bounding box IoU, 그 외 태스크는 label/text/keypoint/track id의 Jaccard 유사도를 사용합니다.
9. consistency가 `--threshold`보다 낮으면 고용량 모델을 temperature `0.0`으로 한 번 더 호출하고, 아니면 첫 번째 저용량 초안을 최종 결과로 사용합니다.
10. 최종 결과를 `DatasetInsightAgent`에 누적하고, `LabelExportWriter`가 선택한 라벨 포맷으로 저장합니다.
11. 최종 결과를 원본 이미지 위에 그려 시각화 이미지를 저장하고, 이미지별 메트릭을 메모리에 누적합니다.
12. 모든 이미지 처리가 끝나면 `classes.txt`와 COCO dataset JSON처럼 전체 실행 단위 산출물을 finalize합니다.
13. `run_metrics.csv`와 `run_metrics.jsonl`을 저장합니다.
14. `--gt_dir`가 있고 YOLO 출력이 포함된 경우에만 ground truth와 비교해 precision, recall, mean IoU를 출력합니다.
15. 전체 클래스 분포와 불균형 여부를 콘솔에 출력합니다.

이미지 하나를 처리하다가 예외가 발생하면 해당 이미지 오류를 출력하고 다음 이미지 처리를 계속합니다.

## API 호출량과 비용 주의

기본 설정에서는 이미지 1장당 저용량 모델을 3회 호출합니다. consistency가 낮으면 고용량 모델을 1회 추가 호출합니다.

```text
이미지당 정상 호출 수 = 3회 또는 4회
```

각 `predict()` 호출은 실패 시 최대 3회까지 재시도합니다.

```text
이미지당 최악 요청 수 = (3회 저용량 + 1회 고용량) * 3회 재시도 = 최대 12회
```

현재 구현은 입력 이미지 목록을 한 번만 순회하므로 무한 루프 구조는 아닙니다. 다만 입력 이미지가 많거나 threshold가 높아 에스컬레이션이 자주 발생하면 비용이 증가합니다.

논문 주장에 맞는 비용 절감률을 해석하려면 `LOW_MODEL`이 실제로 `HIGH_MODEL`보다 저렴하고 빠른 모델이어야 합니다. 예를 들어 AWS Bedrock Claude에서는 Haiku 계열을 low model, Sonnet 또는 Opus 계열을 high model로 두는 구성이 적합합니다.

## 논문 주장과의 대응

현재 구현은 proposal/abstract의 핵심 구조에 맞춰 다음 항목을 지원합니다.

| 논문 주장 | 구현 |
| --- | --- |
| 이종 capacity LMM cascade | `LOW_MODEL`과 `HIGH_MODEL`을 분리하고, 같은 모델이면 기본 차단 |
| 불확실성 기반 에스컬레이션 | 반복 추론 consistency가 `--threshold` 미만이면 high model 호출 |
| Ground Truth 없는 self-consistency | 같은 이미지에 대한 반복 예측을 태스크별 consistency로 계산 |
| confidence + consistency 결합 | `uncertainty_score = 1 - ((consistency + confidence) / 2)` |
| 비용 제어 | high model 호출 횟수와 low/high API attempts를 메트릭에 기록 |
| 다중 포맷 변환 | 최종 라벨 결과를 YOLO txt, Pascal VOC XML, COCO JSON, Vision JSONL, custom 템플릿으로 저장 |
| 데이터셋 인사이트 | 클래스 분포와 단순 불균형 경고 출력 |

## 현재 구현상 참고 사항

- OpenAI 호출에는 별도 `max_tokens` 제한이 지정되어 있지 않습니다.
- Anthropic 및 Bedrock 호출은 `max_tokens=1024`로 제한됩니다.
- API timeout은 명시적으로 설정되어 있지 않습니다.
- `LabelingAgent`의 반복 추론 횟수는 `--inference_count`로 조정할 수 있습니다.
- `opencv-python`과 `matplotlib`은 의존성에 포함되어 있지만 현재 주요 실행 경로에서는 Pillow 기반 시각화를 사용합니다.
