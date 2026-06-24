# Agentic Auto-Labeling System

비전 AI 데이터셋의 라벨 생성, 라벨 형식 변환, 사전 검증, 결과 리포트, 라벨 편집을 하나의 Streamlit UI에서 수행하는 자동 라벨링 도구입니다.

기본 사용 방식은 `web_app.py`를 실행한 뒤 브라우저에서 작업하는 흐름입니다. CLI 스크립트는 개발·실험·자동화용 보조 경로로 유지합니다.

## 주요 기능

- 자연어 기반 대화형 작업 계획 생성
- classification, object detection, segmentation, pose estimation, OCR, tracking 라벨 생성
- YOLO, Pascal VOC, COCO, Vision JSON 등 라벨 포맷 변환
- 포맷별 클래스 정보 수집: YOLO mapping 파일, COCO categories, Pascal VOC object name, Vision JSON/CSV label 필드
- 혼합 라벨 포맷 입력의 파일별 감지, 병합, 중복 제거
- 변환 전 사전 점검과 부족 정보 안내
- 결과 리포트, 입력 데이터 문제, 결과 파일 문제 표시
- 1차 Vision Specialist 결과와 선택적 Low/High LMM 재생성 결과 간 IoU 기반 self_consistency 계산
- self_consistency 임계치 미달 이미지만 라벨 편집 큐로 전달
- bbox, polygon, OCR, tracking, pose, classification 라벨 편집
- 결과 파일과 리포트 다운로드

## 빠른 시작

### 1. 가상환경 생성

```powershell
python -m venv .venv
.\.venv\Scripts\activate
```

### 2. 의존성 설치

기본 실행과 Streamlit UI:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip install -r requirements-web.txt
```

태스크별 특정 Vision Model을 사용할 경우:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-specialists.txt
```

### 3. Streamlit 실행

```powershell
.\.venv\Scripts\python.exe -m streamlit run web_app.py
```

브라우저에서 `http://localhost:8501`을 엽니다.

## Workspace

앱을 시작하면 작업 기준 디렉터리인 workspace를 선택합니다. 화면에 표시되는 대부분의 경로는 workspace 기준 상대 경로입니다.

권장 폴더 구조:

```text
data/raw                 입력 이미지
data/labeled             생성 라벨 출력
data/visualized          시각화 출력
data/converted           변환 라벨 출력
data/ground_truth        평가용 정답 라벨
data/reports             리포트 출력
configs/plugins.json     모델 plugin 설정
```

`표준 폴더 구조 생성`을 선택하면 위 폴더가 자동으로 생성됩니다. 기존 데이터셋 폴더를 그대로 사용할 수도 있습니다.

주의:
- workspace를 지정해도 기존 파일을 자동 삭제하지 않습니다.
- 결과는 사용자가 지정한 출력 경로에 저장됩니다.
- 모델 가중치와 cache 파일은 일반적으로 각 라이브러리 cache 또는 프로젝트 설정 경로에 저장됩니다.

## Streamlit 탭 구성

현재 Streamlit UI는 다음 6개 탭으로 구성됩니다.

1. 대화형 작업
2. 형식 변환
3. 라벨 생성
4. 라벨 편집
5. 결과 리포트
6. 설정

## 1. 대화형 작업

자연어로 작업을 요청하면 workspace를 탐색한 뒤 실행 계획을 먼저 보여줍니다. 사용자가 승인해야 실제 파일 생성 또는 변환이 실행됩니다.

예시:

```text
현재 데이터셋의 라벨링 형식을 MS COCO 형식으로 바꿔줘
```

```text
names:
  0: person
  1: car
path: D:\project\autolabel\test_images

위 클래스 객체만 검출해줘
```

대화형 작업 특징:
- 라벨 생성, 형식 변환, 실험 평가 요청을 해석합니다.
- 규칙 기반 parser가 먼저 동작하고, 실패 시 LMM Intent Router를 사용할 수 있습니다.
- 실행 계획이 뜬 뒤 사용자가 추가 프롬프트로 경로, 출력 포맷, 태스크 등을 수정할 수 있습니다.
- 실행 전 확인 사항과 검증 이슈를 함께 보여줍니다.
- 실행 완료 후 결과 리포트 탭에서 상세 결과를 확인합니다.

## 2. 형식 변환

기존 라벨 파일을 다른 라벨 포맷으로 변환합니다.

지원 입력:
- YOLO txt
- Pascal VOC XML
- COCO JSON
- Vision JSON / Vision JSONL
- CSV bbox
- generic JSON bbox
- 사용자 정의 mapping 기반 custom label

지원 출력:
- YOLO
- Pascal VOC
- COCO
- Vision JSON
- custom template

주요 입력 항목:

| 항목 | 설명 |
| --- | --- |
| 입력 라벨 경로 | 변환할 라벨 파일 또는 라벨 폴더 |
| 이미지 디렉터리 | 라벨과 연결되는 원본 이미지 폴더 |
| 출력 디렉터리 | 변환 결과 저장 위치 |
| 클래스 매핑 파일 | YOLO class id를 클래스명으로 해석할 `data.yaml`, `dataset.yaml`, `classes.txt` |
| 입력 포맷 | `auto` 또는 명시 포맷 |
| 출력 포맷 | 생성할 라벨 포맷 |

`변환 사전 점검`을 누르면 파일 생성 없이 다음을 먼저 확인합니다.
- 라벨 파일 탐색 결과
- 입력 포맷 판별 결과
- YOLO class mapping 누락 여부
- 이미지 연결 여부
- 좌표 오류, 빈 라벨, 지원하지 않는 파일 형식

이미지가 없는 경우:
- `missing_image`는 기본적으로 warning/report 대상으로 처리합니다.
- 가능한 출력은 계속 생성합니다.
- 다만 이미지 크기가 반드시 필요한 출력은 결과 파일 문제로 기록될 수 있습니다.

## 3. 라벨 생성

이미지에서 새 라벨을 생성합니다. 기본 구조는 Vision Specialist Model 우선입니다.

태스크별 기본 모델 흐름:

| 태스크 | 기본 모델 흐름 |
| --- | --- |
| classification | SigLIP |
| object detection | Grounding DINO |
| segmentation | Grounded-SAM2 pipeline, Grounding DINO + SAM2 |
| pose estimation | Ultralytics YOLO26 pose |
| OCR | PaddleOCR PP-OCRv5, Korean mobile recognition |
| tracking | YOLO26n + ByteTrack |

입력 항목:

| 항목 | 설명 |
| --- | --- |
| 이미지 디렉터리 | 라벨을 생성할 이미지 폴더 |
| 라벨 출력 | 생성 라벨 저장 위치 |
| 시각화 출력 | bbox/polygon 시각화 저장 위치 |
| 태스크 | detection, segmentation, OCR 등 |
| 출력 포맷 | YOLO, COCO, Vision JSON 등 |
| 클래스 매핑 파일 | 탐지할 클래스 후보 |
| 프롬프트 | 탐지 대상과 조건 |

라벨 생성 흐름:

1. Vision Specialist Model로 1차 추론
2. 결과 리포트 표시
3. 사용자가 선택할 경우 specialist 재추론 또는 LMM 재생성 비교 실행
4. IoU 기반 self_consistency 계산
5. 임계치 미달 이미지만 라벨 편집 큐로 전달

LMM 재생성 비교는 최종 라벨을 자동 교체하지 않습니다. 1차 Vision Model 결과와 Low/High LMM 결과를 비교해 검토 우선순위를 정하는 용도입니다.

## 4. 라벨 편집

라벨 파일을 불러와 이미지 위에서 직접 수정합니다.

지원 편집:
- bbox 이동 및 크기 조정
- polygon vertex drag/drop
- polygon 생성
- OCR 영역 수정
- tracking bbox 수정
- pose/classification 레코드 편집
- 원본 포맷 기반 저장

결과 리포트나 검증 이슈 상세에서 특정 파일을 선택해 라벨 편집 탭으로 넘길 수 있습니다.

LMM self_consistency 임계치 미달 이미지의 경우:
- 미달 이미지만 라벨 편집 큐에 들어갑니다.
- 임계치를 통과한 이미지는 해당 큐에 표시하지 않습니다.
- 이전/다음 이슈 버튼으로 순차 편집할 수 있습니다.

## 5. 결과 리포트

라벨 생성, 형식 변환, 대화형 작업 실행 결과를 확인합니다.

표시 항목:
- 처리 이미지 수
- 생성 라벨 수
- 평균 처리 시간
- 완료율
- 입력 데이터 문제
- 결과 파일 문제
- 검토 필요 파일
- Dataset Insight
- 클래스 분포
- LMM 재생성 self_consistency 요약
- 임계치 미달 이미지 목록
- 결과 파일 다운로드

용어:

| 용어 | 의미 |
| --- | --- |
| 입력 데이터 문제 | 입력/변환 전후 검증 단계에서 발견된 문제 |
| 결과 파일 문제 | 저장된 결과물에서 발견된 문제 |
| 검토 필요 | 사용자가 확인해야 하는 실제 데이터 파일 수 |

## 6. 설정

Streamlit 설정 탭에서 모델과 API credential을 저장할 수 있습니다.

저장 위치:

```text
%APPDATA%\AutoLabel\.env
```

주요 설정:
- AWS Region, Profile, Access Key, Secret Key, Session Token
- OpenAI API Key
- Anthropic API Key
- Low Model
- High Model
- Planner Model

권장 Bedrock 예시:

```env
AWS_REGION=us-east-1
LOW_MODEL=bedrock:us.anthropic.claude-haiku-4-5-20251001-v1:0
HIGH_MODEL=bedrock:us.anthropic.claude-sonnet-4-5-20250929-v1:0
```

`LOW_MODEL`과 `HIGH_MODEL`이 같으면 기본적으로 실행을 중단합니다. LMM 재생성 비교와 fallback 검증에서 서로 다른 capacity의 모델을 구분하기 위한 설정입니다.

## 출력 파일

라벨 생성 기본 출력:

```text
data/labeled/
├── *.txt 또는 *.xml
├── coco_annotations.json
├── vision_annotations.jsonl
├── classes.txt
├── data.yaml
├── run_metrics.csv
├── run_metrics.jsonl
├── run_summary.json
└── user_action_report.json
```

형식 변환 기본 출력:

```text
data/converted/
├── 변환된 라벨 파일
├── conversion_report.json
└── user_action_report.json
```

시각화 출력:

```text
data/visualized/
└── vis_<원본파일명>
```

## Specialist 모델 의존성

기본 Streamlit UI와 라벨 변환은 `requirements.txt`, `requirements-web.txt`로 실행할 수 있습니다. 실제 Vision Specialist Model 추론을 사용하려면 `requirements-specialists.txt`가 필요합니다.

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-specialists.txt
```

모델 가중치는 repository에 포함하지 않습니다. 각 plugin은 실행 또는 prepare 단계에서 필요한 모델을 로드하거나 다운로드합니다.

기본 모델 예:
- SigLIP: Hugging Face loader
- Grounding DINO: Hugging Face loader
- SAM2: `sam2_b.pt`
- PaddleOCR: PP-OCRv5 Korean mobile recognition
- YOLO26n / YOLO26 pose: Ultralytics loader

## Windows Setup 배포

Windows 설치형 앱은 `packaging/build.ps1`로 생성합니다.

```powershell
.\packaging\build.ps1
```

산출물:

```text
dist-installer\AutoLabel-Setup.exe
```

기본 installer는 용량과 환경 의존성을 줄이기 위해 무거운 specialist 의존성과 모델 가중치를 포함하지 않는 구성을 권장합니다. specialist 포함 배포판은 대상 CPU/GPU, CUDA, PyTorch 조합을 고정해 별도로 빌드하는 방식이 안전합니다.

## 참고 문서

| 문서 | 내용 |
| --- | --- |
| [streamlit_workflow.md](streamlit_workflow.md) | Streamlit 탭별 세부 동작 |
| [generate_label.md](generate_label.md) | 자동 라벨 생성과 specialist plugin 상세 |
| [transform_label_format.md](transform_label_format.md) | 라벨 포맷 변환 상세 |
| [agentic_workflow.md](agentic_workflow.md) | LangGraph workflow와 checkpoint |
| [desktop_setup.md](desktop_setup.md) | Windows installer 빌드와 배포 |
| [docs/user_action_report_guide.md](docs/user_action_report_guide.md) | 오류 데이터 사용자 수정 제안 리포트 |
| [CONTRIBUTING.md](CONTRIBUTING.md) | 개발 및 테스트 기준 |
| [SECURITY.md](SECURITY.md) | credential 및 데이터 보안 정책 |

## 보안 및 운영 주의

- Streamlit은 로컬 파일 경로와 credential을 다루므로 외부 네트워크에 공개하지 않는 것을 기본 운영 방식으로 합니다.
- `%APPDATA%\AutoLabel\.env`에는 API key가 평문으로 저장될 수 있습니다.
- 실제 배포 시 installer 코드 서명과 clean VM 검증을 권장합니다.
- 대용량 모델 가중치, cache, dataset, metrics 출력물은 Git에 포함하지 않는 것이 원칙입니다.
