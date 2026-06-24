# Streamlit Workflow

## 1. 문서 목적

이 문서는 `web_app.py`를 기준으로 AutoLabel Streamlit 인터페이스의 사용자 흐름과 내부 workflow 연결 방식을 설명한다. Streamlit은 별도의 처리 엔진이 아니라 `src/workflow`의 공통 LangGraph workflow를 호출하는 UI 계층이다.

전체 흐름은 다음과 같다.

```text
사용자 입력
  -> Streamlit UI 검증
  -> WorkflowPlan 생성
  -> execute_workflow_plan()
  -> LangGraph plan 검증 및 operation 실행
  -> 라벨·리포트 파일 저장
  -> Streamlit에 상태, 요약, 오류, 결과 경로 표시
```

## 2. 실행 방법

루트 의존성과 Streamlit 의존성을 설치한 뒤 프로젝트 루트에서 실행한다.

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip install -r requirements-web.txt
.\.venv\Scripts\python.exe -m streamlit run web_app.py
```

기본 접속 주소는 `http://localhost:8501`이다. 로컬 파일 경로와 사용자 credential을 사용하므로 외부 네트워크에 공개하지 않는 것을 기본 운영 방식으로 한다.

## 3. Workspace 선택

앱 최초 실행 시 작업 기준 디렉터리인 workspace를 선택한다. `표준 폴더 구조 생성`을 선택하면 다음 구조가 생성된다.

| 용도 | 기본 상대 경로 |
| --- | --- |
| 원본 이미지 | `data/raw` |
| 생성 라벨 | `data/labeled` |
| 라벨 시각화 | `data/visualized` |
| 변환 라벨 | `data/converted` |
| 평가 ground truth | `data/ground_truth` |
| 평가 리포트 | `data/reports` |
| plugin 설정 | `configs/plugins.json` |

화면에는 상대 경로를 입력할 수 있으며 operation 실행 직전에 workspace 기준 절대 경로로 변환된다. 마지막 workspace는 `%APPDATA%\AutoLabel\workspace.json`에 저장된다.

Workspace 적용 후 UI는 다음 6개 탭으로 구성된다.

1. 대화형 작업
2. 형식 변환
3. 라벨 생성
4. 라벨 편집
5. 결과 리포트
6. 설정

## 4. 대화형 작업

### 4.1 역할

한국어 또는 영어 자연어 요청을 구조화된 `WorkflowPlan`으로 변환한다. 현재 지원 operation은 형식 변환, 라벨 생성, 실험 평가이다.

대화 입력은 다음 순서로 처리한다.

```text
규칙 기반 parser
  -> 성공: 기존 workspace 탐색과 WorkflowPlan 생성
  -> 의도/출력 포맷 해석 실패: LLM Intent Router
       -> 작업 의도: 허용된 파라미터만 기존 계획 생성기에 전달
       -> 일반 질문: 비실행 Chat Node 응답
       -> unknown 또는 confidence < 0.65: 구체적인 요청을 다시 확인
```

LLM Intent Router는 작업을 직접 실행하지 않는다. `convert_labels`, `generate_labels`,
`evaluate_labels` 등의 제한된 intent와 포맷, task, threshold 같은 구조화된 값만 반환한다.
실제 경로 탐색, Pydantic 계획 검증, 승인 및 실행은 기존 workflow가 담당한다. 규칙으로 이미
해석 가능한 요청에는 LLM을 호출하지 않는다.

`INTENT_ROUTER_MODEL`과 `CHAT_MODEL`을 별도로 설정할 수 있다. 생략하면 각각
`PLANNER_MODEL`, `LOW_MODEL` 순으로 fallback한다. 사용할 모델이 전혀 설정되지 않았거나
LLM 호출에 실패하면 기존 규칙 parser의 오류를 사용자에게 표시하며 임의 계획을 실행하지 않는다.

예시:

```text
현재 데이터셋의 라벨링 형식을 MS COCO 형식으로 바꿔줘
이미지에서 차량과 보행자를 찾아 세그멘테이션 라벨을 생성해줘
베이스라인과 Cascade 실행 결과를 비교해서 평가 리포트를 만들어줘
names: 0: person 1: car path: D:\project\autolabel\test_images 위 클래스 객체만 검출해줘
현재 이미지들을 YOLO로 객체 탐지해줘
```

대화형 라벨 생성은 초기 프롬프트로 재추론 여부를 바로 확정하지 않는다. 먼저 1차 specialist 추론 결과와 리포트를 확인한 뒤, 화면에서 `Specialist 재추론` 또는 `LMM 재생성 비교`를 선택적으로 실행한다. LMM 재생성 비교는 1차 Vision Model 결과와 Low/High LMM 결과 간 bbox IoU 기반 self_consistency를 계산하고, 임계치 미달 이미지만 라벨 편집 큐로 전달할 수 있다.

### 4.2 Workspace 자동 탐색

요청이 입력되면 workspace를 읽기 전용으로 탐색한다.

- 이미지 확장자: JPG, JPEG, PNG, WEBP, BMP
- 라벨 후보: YOLO, Pascal VOC, COCO, Vision JSON, CSV, generic JSON
- 라벨 CSV는 이미지명 필드와 `xmin`, `ymin`, `xmax`, `ymax` 필드가 있어야 한다.
- `.git`, `.venv`, build, dist, logs, 기존 converted·reports·workflow 결과는 자동 입력 탐색에서 제외한다.
- `data/labeled`는 AutoLabel 생성 결과의 기본 출력 폴더이고, `labels`/`data/labels`는 외부 데이터셋에서 흔한 라벨 입력 폴더명이다.
- 변환 요청은 `data/labeled`, `data/labels`, `labels`, `data/annotations`, `annotations`, `data/external_labels` 순으로 라벨 후보를 우선 탐색한다.
- 명시적인 `data/external_labels` 같은 workspace 상대 경로가 있으면 해당 경로를 우선한다.
- 변환에 필요한 이미지 디렉터리는 선택된 라벨 후보와 가장 가까운 `data/raw`, `data/images`, `images`, `img`, `JPEGImages` 계열 폴더를 연결한다.
- 후보가 여러 개이면 실행 계획 확인 사항에 실제 선택된 라벨 경로와 이미지 경로를 표시한다.
- 형식 변환과 평가는 workspace 내부 입력을 우선하며, LLM이 추출한 상대 경로도 workspace 경계 검사를 통과해야 한다.
- 라벨 생성은 프롬프트에 명시한 `path:` 이미지 폴더를 입력으로 사용할 수 있다. 이 경로가 workspace 밖이어도 허용하지만, 생성 결과는 현재 workspace의 출력 폴더에 저장한다.

### 4.3 자연어에서 추출하는 값

- 작업 종류: convert, generate, evaluate
- 입력·출력 포맷
- 생성 task: classification, object detection, segmentation, pose estimation, OCR, tracking
- workspace 상대 입력 경로 또는 라벨 생성용 명시적 `path:` 이미지 경로
- YOLO `names:` 클래스 블록 또는 클래스 매핑 파일 경로
- 대화형 라벨 생성의 1차 실행은 재추론 없이 시작하며, 재추론 advisor mode는 1차 결과 확인 이후 UI에서 선택
- 중복 IoU
- strict 검증 제외 여부
- 생성 threshold
- 복수 출력 포맷

### 4.4 승인과 실행

자연어를 바로 실행하지 않고 다음 항목을 포함한 계획을 먼저 표시한다.

- 선택된 입력 라벨
- 이미지 위치
- 출력 위치
- 출력 포맷
- IoU, strict, threshold 같은 추가 옵션
- 자동 선택과 관련된 경고

사용자가 `계획 실행`을 선택해야 실제 operation이 실행된다. `취소`를 선택하면 파일을 변경하지 않는다. 계획이 대기 중일 때는 중복 요청을 막기 위해 채팅 입력이 잠긴다.

### 4.5 결과 표시

완료 후 대화에 처리 건수, 감지된 입력 포맷, 출력 포맷과 리포트 경로를 표시한다. 전체 리포트 내용은 workspace에 저장된 JSON·CSV 파일에서 확인한다.

현재 대화형 평가는 ground truth 기반 YOLO 비교와 `run_metrics.csv` 실험 비교를 지원한다. 라벨이 원본 이미지의 객체와 의미적으로 정확히 일치하는지 판단하는 공간 정합성 평가는 지원하지 않으며, 해당 요청은 실행하지 않고 제한사항을 안내한다.

## 5. 형식 변환

### 5.1 입력 항목

| 항목 | 설명 | 기본값 |
| --- | --- | --- |
| 입력 라벨 경로 | 변환할 파일 또는 디렉터리 | `data/labeled` |
| 이미지 디렉터리 | 크기 계산과 이미지 연결에 사용할 경로 | `data/raw` |
| 출력 디렉터리 | 변환 결과 저장 위치 | `data/converted` |
| 클래스 매핑 파일 | YOLO 입력 class id를 실제 클래스명으로 해석할 `data.yaml`, `dataset.yaml`, `classes.txt` | 선택 |
| 입력 포맷 | auto 또는 명시 포맷 | `auto` |
| 출력 포맷 | 하나 이상의 출력 형식 | `yolo` |
| 중복 IoU | 혼합 라벨 병합 시 중복 판정 기준 | `0.85` |
| 불균형 비율 기준 | Dataset Insight 불균형 판정 기준 | `3.0` |
| strict | 검증 이슈가 있는 레코드 제외 | 해제 |

입력 포맷은 auto, YOLO, Pascal VOC, COCO, Vision JSON, CSV, generic JSON을 지원한다. 출력 포맷은 YOLO, Pascal VOC, COCO, Vision JSON을 지원한다.

`변환 사전 점검` 버튼은 파일을 생성하지 않고 라벨 소스 탐색, YOLO class mapping 탐색, 이미지 연결, validation issue를 먼저 확인한다. 사용자가 라벨 파일만 가져와 점검할 수 있도록 이미지 파일 연결 실패는 warning으로 표시한다. 실제 변환에서도 `missing_image`는 기본 모드에서 blocking issue로 처리하지 않으며, 가능한 출력은 계속 생성한다. 다만 이미지 크기나 실제 이미지 파일이 필요한 포맷에서 저장하지 못한 항목은 결과 파일 문제로 기록될 수 있다. YOLO class mapping이 없으면 변환은 가능하지만 category name이 실제 클래스명이 아니라 `"0"`, `"1"` 같은 문자열 id가 될 수 있으므로 warning으로 표시된다.

### 5.2 처리 순서

```text
입력 경로·포맷 검증
  -> 선택 시 변환 사전 점검으로 누락 정보 표시
  -> source schema 후보 구성
  -> 라벨 읽기 및 혼합 포맷 자동 감지
  -> 동일 이미지 레코드 병합과 중복 제거
  -> 이미지 연결 및 좌표·필수값 검증
  -> 가능한 레코드 repair
  -> strict·blocking issue 적용
  -> 선택한 포맷으로 export
  -> DatasetInsightAgent 클래스 분포·불균형 분석
  -> 변환 리포트 저장
```

### 5.3 출력

기본 출력 위치는 `data/converted`이며 다음 파일이 생성된다.

- 변환된 라벨 파일
- `conversion_report.json`
- `user_action_report.json`

변환 리포트에는 실제 감지된 입력 포맷, 읽은 레코드 수, 변환 성공 수, preflight 확인 사항, validation 요약, 출력 artifact 재검증, Dataset Insight, 레코드별 이슈와 artifact 경로가 포함된다. `preflight`는 변환 전후에 부족한 필수 정보와 권장 정보를 `critical`·`warning`·`info`로 분류하며, `user_action_report.json`은 문제를 심각도별로 분류하고 우선 조치와 권장 작업을 제공한다.

대화형 변환에서 라벨 후보는 찾았지만 이미지 디렉터리를 찾지 못하면 계획 확인 사항과 사전 점검 결과에 경고를 표시한다. 이미지가 없는 경우에도 가능한 변환은 진행할 수 있으며, 이미지 크기가 필요한 출력 포맷에서 누락된 항목은 결과 리포트의 결과 파일 문제로 확인한다.

## 6. 라벨 생성

### 6.1 입력 항목

| 항목 | 설명 | 기본값 |
| --- | --- | --- |
| 이미지 디렉터리 | 라벨링할 이미지 | `data/raw` |
| 라벨 출력 | 생성 라벨 저장 위치 | `data/labeled` |
| 시각화 출력 | 라벨 시각화 이미지 저장 위치 | `data/visualized` |
| Plugin 설정 | specialist plugin 설정 | `configs/plugins.json` |
| 클래스 매핑 파일 | Grounding DINO, Grounded-SAM2, classification 후보 클래스에 사용할 YOLO `data.yaml`, `dataset.yaml`, `classes.txt` | 선택 |
| 태스크 | 생성할 vision task | `object_detection` |
| 출력 포맷 | 라벨 export 형식 | `yolo` |
| 신뢰도 기준 | 고비용 검증 단계 판단 기준 | `0.75` |
| 불균형 비율 기준 | Dataset Insight 불균형 판정 기준 | `3.0` |
| 초안 추론 횟수 | low model 반복 추론 횟수 | `3` |
| 프롬프트 | 모델에 전달할 작업 지시 | 사용자 입력 |

프롬프트 입력창은 빈 값으로 시작하며 한국어 예시가 placeholder로만 표시된다. 빈 프롬프트는 실행할 수 없다.

라벨 생성 탭도 대화형 작업과 동일하게 초기 실행에서는 specialist 재추론이나 LMM 비교를 수행하지 않는다. 1차 결과 리포트를 확인한 뒤 결과 화면 아래에서 specialist 재추론 또는 LMM 재생성 비교를 선택적으로 이어서 실행한다. LMM 비교에서 self_consistency 임계치 미달로 판정된 이미지는 통과 이미지와 분리해 라벨 편집 탭으로 넘길 수 있다.

지원 task는 다음과 같다.

- object detection
- classification
- segmentation
- pose estimation
- OCR
- tracking
- all

출력 포맷별로 보존하는 label 종류가 다르다.

| 출력 포맷 | 현재 보존 범위 |
| --- | --- |
| YOLO | bounding box |
| Pascal VOC | bounding box |
| COCO | bounding box, polygon segmentation |
| Vision JSON | classification, box, segmentation, pose, OCR, tracking 전체 |

classification, pose estimation, OCR, tracking 또는 `all` task의 전체 결과를 보존하려면 Vision JSON을 사용해야 한다. UI에서 다른 포맷을 선택할 수 있더라도 해당 표준이 표현하지 못하는 label 종류는 출력 파일에 포함되지 않는다.

LMM fallback 또는 LMM 재생성 비교를 사용할 경우 API 호출 비용이 발생할 수 있으므로 `고비용 모델 API 호출 승인`을 선택해야 실행할 수 있다.

### 6.2 처리 순서

```text
이미지 목록과 모델 설정 검증
  -> classes_path 또는 prompt names 블록에서 후보 클래스 추출
  -> 기본 specialist plugin prepare 및 weight 로드/다운로드
  -> specialist_first 전략으로 전문 모델 우선 추론
  -> 결과가 있으면 VLM 호출 없이 validation/export 후보 사용
  -> 1차 추론 통계와 plugin record 기반 first-pass report 생성
  -> 선택 시 specialist 재추론 1회 및 base/rerun bbox self_consistency 계산
  -> 결과가 비어 있거나 vlm_first 전략이면 low model 반복 초안 생성
  -> 초안 consistency 계산
  -> 필요 시 specialist 결과와 VLM 결과 병합
  -> schema·라벨 validation과 repair
  -> threshold 미달 또는 검증 이슈 시 high model 검증
  -> 최종 라벨 export
  -> DatasetInsightAgent 클래스 분포·불균형 분석
  -> 시각화·지표·요약 리포트 저장
```

반복 추론 self-consistency는 task별 metric을 사용한다.

| Task | Self-consistency metric |
| --- | --- |
| Object detection | 동일 label bounding box IoU |
| Segmentation | 512 x 512로 rasterize한 polygon mask IoU |
| Pose estimation | visible keypoint 위치와 pose scale 기반 OKS-style 점수 |
| OCR | text region IoU 40% + NFC 문자 편집 유사도 60% |
| Classification | label 집합 Jaccard |
| Tracking | `track_id:label` 집합 Jaccard |

이 점수는 동일 이미지의 반복 추론 결과 간 일관성이며 Ground Truth 정확도가 아니다. 실제 segmentation Dice/mIoU, 공식 COCO pose OKS, OCR CER/WER를 평가하려면 별도의 task별 Ground Truth 평가 데이터가 필요하다.

기본 `specialist_first` 전략에서는 전문 모델 결과가 생성된 이미지는 low/high VLM API를 호출하지 않을 수 있다. 다만 specialist 결과가 비어 있을 때 VLM fallback이 가능해야 하므로, UI는 고비용 모델 호출 승인과 모델 설정 검증을 유지한다. object detection이 아닌 task에서 출력 포맷이 YOLO 하나뿐이면 데이터 손실을 피하기 위해 Vision JSON으로 전환한다.

Specialist 재추론은 report-only 검증 단계다. 1차 specialist 결과를 최종 라벨 후보로 유지하고, threshold/prompt/augmentation 설정을 조금 바꾼 specialist 결과와 bbox IoU self_consistency를 계산한다. 기본 object detection은 Grounding DINO 단독, 기본 segmentation은 Grounded-SAM2 pipeline(Grounding DINO + SAM2)을 사용한다. Advisor가 `low`, `high`, `both`이면 선택한 LMM은 설정 patch만 제안하며 클래스 목록, bbox, 최종 라벨은 직접 수정하지 못한다.

LMM 재생성 비교는 1차 Vision Model 결과를 pseudo-reference로 두고 사용자가 선택한 Low/High LMM의 재생성 결과와 비교한다. 이 비교는 최종 라벨을 자동 교체하지 않으며, `Prediction Self-Consistency`와 임계치 미달 여부를 결과 리포트에 표시한다. 임계치 미달 이미지만 `라벨 편집` 탭의 이슈 큐로 전달할 수 있고, 임계치를 통과한 이미지는 해당 큐에 표시하지 않는다.

Advisor LLM을 호출하는 경우 1차 추론 결과 요약이 참고 prompt로 함께 전달된다. 포함 정보는 라벨 수, 클래스별 분포, confidence 통계, low-confidence 개수, plugin record, 현재 threshold 파라미터다. 이 정보는 patch 제안을 위한 참고용이며 LLM이 bbox나 class를 직접 수정하는 입력으로 사용하지 않는다.

### 6.3 출력

기본 출력은 다음 위치에 저장된다.

- 생성 라벨: `data/labeled`
- 시각화 이미지: `data/visualized`
- 실행 요약: `data/labeled/run_summary.json`
- 사용자 조치: `data/labeled/user_action_report.json`
- 이미지별 지표: `data/labeled/run_metrics.csv`
- 이미지별 JSON Lines 지표: `data/labeled/run_metrics.jsonl`

대화형 작업에서 `path:`로 workspace 밖 이미지 폴더를 지정해도 위 출력 경로는 선택한 workspace 기준이다. 예를 들어 workspace가 `C:\Users\wook\Documents\autolabel`이면 결과는 `C:\Users\wook\Documents\autolabel\data\labeled`와 `...\data\visualized`에 저장된다.

`run_summary.json`에는 task, `consistency_metric`, 포맷, 처리 이미지 수, 총 라벨 수, 실행 시간, low/high API 호출 수, escalation 수, 추정 효율 KPI, Dataset Insight, 출력 artifact 검증과 사용자 조치 리포트가 기록된다. 이미지별 지표에도 사용한 `consistency_metric`과 label 종류별 개수, consistency, confidence, uncertainty, plugin 결과, validation issue와 시각화 경로가 포함된다.

1차 specialist 리포트는 `first_pass_report`, `first_pass_total_labels`, `first_pass_mean_confidence`, `first_pass_low_confidence_count`로 저장된다. Specialist 재추론을 실행한 경우 `run_metrics.csv/jsonl`에는 `specialist_result_consistency`, `specialist_bbox_agreement`, `specialist_mean_matched_iou`, `specialist_rerun_patch`, `specialist_rerun_records`가 추가된다. `run_summary.json`에는 1차 리포트 요약, 재추론된 이미지 수와 평균 bbox self_consistency가 포함된다.

생성·변환은 공통 `DatasetInsightAgent.analyze()`를 직접 호출한다. Agent는 최종 export 대상 결과만 분석하며 상태를 `empty`, `single_class`, `balanced`, `imbalanced`로 구분한다. 불균형이면 희소 클래스 추가 수집, class-aware oversampling, copy-paste·crop·색상·기하 augmentation을 제안한다.

처리된 이미지가 없더라도 `run_summary.json`은 생성되지만 이미지별 metrics 파일은 생성되지 않는다. 수동 라벨 생성 탭은 operation output 전체를 Streamlit JSON으로 표시한다.

## 7. 라벨 편집

### 7.1 입력 항목

| 항목 | 설명 | 기본값 |
| --- | --- | --- |
| 편집할 라벨 경로 | 기존 라벨 파일 또는 라벨 폴더 | `data/labeled` |
| 이미지 폴더 경로 | 라벨과 연결되는 원본 이미지 폴더 | `data/raw` |
| 라벨 형식 | auto, YOLO, Pascal VOC, COCO, Vision JSON 등 | `auto` |
| 편집 태스크 | detection, segmentation, OCR, tracking, pose, classification | `object_detection` |
| 저장 경로 | 편집 결과 저장 위치 | `data/labeled` |

라벨 편집 탭은 bbox, polygon, OCR box, tracking box, pose/classification 레코드 편집을 지원한다. segmentation polygon은 vertex drag/drop과 별도 polygon 생성 모드를 제공하며, bbox가 필요한 출력 포맷에서는 polygon 외곽 좌표를 기반으로 bbox를 함께 저장한다.

대화형 작업, 형식 변환, 라벨 생성, 결과 리포트에서 문제가 있는 파일을 선택하면 해당 파일만 라벨 편집 큐로 전달할 수 있다. LMM 재생성 비교의 경우 self_consistency 임계치 미달 이미지만 큐에 들어가며, 통과 이미지는 편집 목록에서 제외된다.

### 7.2 저장

편집 저장은 가능한 경우 원본 포맷에 맞춰 수행한다. 표준 포맷이 표현하지 못하는 태스크 정보는 Vision JSON 사용이 권장된다.

평가 기능은 별도 CLI/스크립트 경로에서 유지되지만 현재 Streamlit 기본 탭은 `라벨 편집` 중심으로 구성된다.

## 8. 설정

설정 탭은 workspace와 모델·credential 설정을 저장한다.

- AWS Region, Profile, Access Key, Secret Key, Session Token
- OpenAI API Key
- Anthropic API Key
- Low Model
- High Model
- Planner Model

비밀값은 password input으로 표시한다. 설정은 `%APPDATA%\AutoLabel\.env`에 저장되고 현재 Streamlit process 환경에도 즉시 반영된다. 이 파일은 Streamlit과 Windows 데스크톱 앱이 공유한다.

## 9. 공통 실행 상태와 리포트 표시

생성·변환 결과는 처리량과 완료율 KPI 카드, 문제 파일과 우선 조치 표, 클래스 분포 표, 불균형 제안으로 표시한다. LMM 재생성 비교를 실행한 경우 self_consistency 요약과 임계치 미달 이미지 목록을 표시한다. 대화형 작업의 마지막 실행 결과는 별도 `결과 리포트` 탭에서 같은 리포트 UI로 확인한다.

리포트는 UI에만 존재하는 임시 데이터가 아니라 각 operation의 출력 디렉터리에 파일로 저장된다. Streamlit 다운로드 버튼으로 summary, conversion report, user action report와 생성 artifact를 받을 수 있으며 전체 workflow JSON은 expander에서 확인한다.

LangGraph 실행 이력은 현재 workspace가 아니라 Streamlit process의 실행 디렉터리를 기준으로 `data/workflow/<thread-id>/workflow_history.json`에 저장된다.

## 10. Streamlit 상태 관리

- `st.session_state.workspace`: 현재 workspace
- `st.session_state.chat_messages`: 대화 이력
- `st.session_state.pending_proposal`: 승인 대기 중인 대화형 계획

Streamlit은 위젯 조작마다 스크립트를 다시 실행한다. 장기 operation은 spinner와 함께 동기 실행되며, 실행 중 브라우저를 닫거나 process를 종료하면 해당 UI 세션을 이어서 제어할 수 없다.

## 11. 주요 코드 위치

| 파일 | 역할 |
| --- | --- |
| `web_app.py` | Streamlit 화면, 폼 검증, 계획 실행과 결과 표시 |
| `src/workflow/conversation.py` | workspace 탐색, 자연어 계획 생성, 대화 요약 |
| `src/workflow/conversation_router.py` | 규칙 실패 시 LLM 의도 분류, 신뢰도 확인, 일반 대화 Chat Node 분기 |
| `src/workflow/service.py` | UI에서 공통 LangGraph를 호출하는 진입점 |
| `src/workflow/graph.py` | plan 검증과 operation 상태 전이 |
| `src/workflow/runtime.py` | 생성, 변환, 평가의 실제 처리와 리포트 저장 |
| `src/core/workspace.py` | workspace 저장, 기본 구조, 상대 경로 해석 |
| `src/core/user_settings.py` | 사용자 credential과 모델 설정 저장 |
