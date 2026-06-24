# Agentic Workflow 기반 자동 라벨링 및 검증 시스템

## 1. Abstract

컴퓨터 비전 모델 개발에서 데이터 수집과 라벨링은 전체 파이프라인의 주요 병목이다. 기존 자동 라벨링 도구는 단일 모델 또는 고정된 변환 흐름에 의존하는 경우가 많아, 이종 라벨 포맷 처리, 결과 품질 검증, 클래스 불균형 분석, 오류 데이터 수정 안내가 분리되어 수행된다.

본 프로젝트는 이 문제를 해결하기 위해 태스크별 specialist vision model을 우선 실행하고, 필요한 경우 Low/High LMM을 선택적 검증·비교 단계에 사용하는 Agentic Workflow 기반 자동 라벨링 시스템을 구현했다. 제안 시스템은 YOLO `names:`/`data.yaml` 클래스 후보를 Grounding DINO, Grounded-SAM2, SigLIP, PaddleOCR 등 태스크별 모델에 전달하고, LMM 재생성 비교를 통해 1차 Vision Model 결과와의 IoU 기반 self_consistency를 계산한다. 또한 YOLO, Pascal VOC, COCO, Vision JSON 등 다양한 라벨 포맷을 통합 변환하고, 변환 전 부족 정보와 변환 후 오류 데이터를 사용자 조치 리포트 및 라벨 편집 큐로 제공한다.

최종 산출물은 연구용 CLI에 머물지 않고 Streamlit 기반 로컬 웹 UI와 Windows `setup.exe` 설치형 앱으로 배포 가능한 형태를 목표로 한다.

## 2. Motivation

### Problem

- 라벨링 작업은 시간과 비용이 많이 들며, 모델 학습 전 데이터 준비 단계의 병목이 된다.
- YOLO, Pascal VOC, COCO 등 라벨 포맷이 혼재하면 클래스 ID와 클래스명이 불일치할 수 있다.
- 단일 LMM의 self-reflection만으로는 모델이 만든 오류를 스스로 놓칠 가능성이 있다.
- 변환 실패 또는 오류 데이터가 발생했을 때 사용자는 어떤 파일을 어떻게 수정해야 하는지 직접 추적해야 한다.
- 연구용 스크립트만으로는 비개발자 사용과 반복 실험 재현성이 제한된다.

### Goal

- 자연어 또는 UI 입력으로 라벨 생성, 변환, 검증, 평가를 실행한다.
- 태스크별 Vision Specialist Model을 우선 사용하고, 필요한 경우 Low/High LMM을 선택적으로 호출해 품질 검토 비용을 관리한다.
- 다중 라벨 포맷을 한 번에 입력받아 canonical class space로 정렬한다.
- 오류 데이터와 부족 정보를 사용자 수정 제안 리포트로 제공한다.
- Streamlit과 Windows installer로 배포 가능한 사용 환경을 제공한다.

## 3. System Overview

```text
Input Images / Existing Labels
        |
        v
Intent Router / Workflow Planner
        |
        +--> Label Generation
        |       Specialist-first detection/segmentation
        |       Optional specialist rerun self_consistency
        |       Optional Low/High LMM regeneration self_consistency
        |
        +--> Label Format Conversion
        |       YOLO / Pascal VOC / COCO / Vision JSON import
        |       Class mapping normalization
        |       Canonical class list construction
        |
        +--> Validation & Reporting
        |       Label validation
        |       Artifact audit
        |       Preflight report
        |       User action report
        |
        v
YOLO / Pascal VOC / COCO / Vision JSON / Visualization / Metrics
```

## 4. Methodology

### 4.1 Agent Modules and Specialist-First Workflow

| Agent | Role | Output |
| --- | --- | --- |
| Labeling Agent | specialist 결과가 비어 있거나 `vlm_first` 전략일 때 Low LMM draft label 생성 | Draft labels, confidence |
| Hierarchical Verification Agent | VLM fallback consistency 계산과 선택적 High LMM 검증 | Verification result, escalation decision |
| Dataset Insight Agent | 클래스 분포와 불균형을 분석하고 증강 전략을 제안 | Distribution report, imbalance suggestions |

실제 기본 라벨 생성 경로는 Vision Specialist Model 우선 구조이다. Agent 모듈은 fallback, 검증, 분석 단계에서 결합된다.

### 4.2 Self-Consistency-Based Verification

1차 Vision Specialist 결과를 기준으로 두고, 선택적으로 specialist 재추론 또는 Low/High LMM 재생성 결과를 비교한다. bbox 기반 태스크에서는 같은 클래스 bbox 간 IoU matching으로 self_consistency를 계산하고, 임계치 미달 이미지만 라벨 편집 큐로 전달한다.

| Task | Verification/Consistency Metric |
| --- | --- |
| Object Detection | Bounding box IoU |
| Segmentation | Polygon/mask overlap |
| Pose Estimation | Keypoint 위치 기반 유사도 |
| OCR | Text region IoU + 문자 유사도 |
| Classification | Label set similarity |
| Tracking | Track label/id consistency |

VLM fallback 경로에서는 반복 추론 consistency와 confidence를 결합해 uncertainty를 계산한다.

```text
uncertainty_score = 1 - ((consistency_score + mean_confidence) / 2)
```

LMM 재생성 비교 경로에서는 최종 라벨을 자동 교체하지 않고 self_consistency와 검토 필요 여부를 리포트한다. 임계치 미달 이미지만 라벨 에디터로 전달하여 사용자가 직접 보정한다.

### 4.3 Multi-Format Label Conversion

지원 입력:

- YOLO txt + `data.yaml`, `dataset.yaml`, `classes.txt`
- Pascal VOC XML
- COCO JSON
- Vision JSONL
- bbox CSV / generic JSON

핵심 처리:

- 파일별 포맷 자동 감지
- 같은 이미지에 대한 라벨 병합
- IoU 기반 중복 제거
- YOLO class ID와 VOC/COCO class name의 canonical mapping 정렬
- unmapped numeric label이 class list를 먼저 점유하지 않도록 보정
- 변환 결과를 YOLO, Pascal VOC, COCO, Vision JSON으로 export

### 4.4 Preflight and User Action Report

변환 전후에 사용자가 확인해야 할 정보를 구조화한다.

| Report | Purpose |
| --- | --- |
| `preflight` | 실행 전 부족한 필수 정보, YOLO class mapping 누락, 이미지 누락, schema 문제를 안내 |
| `user_action_report` | 검증 실패와 artifact 오류를 심각도, 원인, 수정 지침으로 정리 |
| `conversion_report` | 입력 파일 처리 결과, 병합 내역, class normalization 내역 기록 |

`user_action_report`는 기존 필드와 호환성을 유지하면서 다음 정보를 추가한다.

- `affected_path`: 문제가 발생한 파일 경로
- `fix_instruction`: 사용자가 바로 수행할 수 있는 수정 지침
- `suggestions`: 같은 오류를 해결하기 위한 점검 목록

### 4.5 Deployment-Oriented Interface

연구 구현을 실제 사용 가능한 도구로 확장하기 위해 두 가지 인터페이스를 제공한다.

- Streamlit: 설치 전 localhost에서 workflow 확인
- Windows desktop app: PySide6 기반 GUI와 `setup.exe` installer 배포

Streamlit과 데스크톱 앱은 workspace, 환경 변수, 모델 설정을 공유하며, 라벨 생성·변환·평가 흐름을 동일한 runtime service로 실행한다.

## 5. Implementation

### Core Components

| Component | Description |
| --- | --- |
| `src/agents/labeling_agent.py` | Vision LMM 호출과 draft label 생성 |
| `src/agents/verification_agent.py` | consistency 기반 검증과 high-model escalation |
| `src/agents/insight_agent.py` | 클래스 분포와 데이터 불균형 분석 |
| `src/utils/label_importer.py` | 혼합 라벨 포맷 입력, YOLO class mapping 탐색, class normalization |
| `src/utils/format_converter.py` | YOLO, VOC, COCO, Vision JSON export |
| `src/reporting/issue_reporter.py` | 사용자 조치 리포트 생성 |
| `src/reporting/conversion_preflight.py` | 변환 전 부족 정보 안내 |
| `src/workflow/runtime.py` | CLI, Streamlit, desktop 공통 실행 계층 |
| `web_app.py` | Streamlit UI |
| `desktop_app.py` | Windows desktop UI |

### Supported Outputs

- YOLO `.txt`, `classes.txt`, `data.yaml`
- Pascal VOC `.xml`
- COCO `coco_annotations.json`
- Vision JSONL
- Bounding box visualization image
- `run_metrics.csv`, `run_metrics.jsonl`
- `conversion_report.json`
- `user_action_report.json`

## 6. Results

현재 구현은 다음 기능을 end-to-end로 지원한다.

### Functional Results

- 자연어 또는 UI 기반 generate/convert/evaluate workflow 실행
- Streamlit 기반 라벨 생성, 변환, 결과 리포트, 라벨 편집 workflow 실행
- Vision Specialist Model 우선 라벨 생성과 선택적 Low/High LMM 재생성 비교
- self_consistency threshold 기반 검토 대상 이미지 선별 및 라벨 편집 큐 전달
- YOLO, Pascal VOC, COCO 혼합 입력의 자동 감지와 병합
- YOLO `data.yaml`/`dataset.yaml`/`classes.txt` 기반 class mapping 적용
- 다중 입력 포맷에서 숫자 class ID와 실제 class name 혼용 문제 정규화
- 변환 전 preflight report와 변환 후 user action report 생성
- Streamlit 로컬 UI와 Windows installer 배포 구조 제공

### Verification Status

포스터 발표 시 아래 항목을 실험 결과 표 또는 차트로 제시할 수 있다.

| Metric | Description | Value |
| --- | --- | --- |
| Unit tests | 구현 회귀 테스트 개수 | 48 tests passed |
| Compile check | Python module syntax validation | Passed |
| Supported input formats | YOLO, Pascal VOC, COCO, Vision JSONL, CSV, generic JSON | 6 |
| Supported output formats | YOLO, Pascal VOC, COCO, Vision JSONL, custom template | 5 |
| Report types | preflight, conversion, user action, performance, dataset insight | 5 |

### Experimental Results To Fill

실제 데이터셋 실험 후 다음 값을 채워 넣는다.

| Experiment | Metric | Value |
| --- | --- | --- |
| Low model only | Precision / Recall / F1 | TBD |
| High model only | Precision / Recall / F1 | TBD |
| Cascade | Precision / Recall / F1 | TBD |
| Cascade | Escalation rate | TBD |
| Cascade | Estimated relative cost | TBD |
| Manual labeling baseline | Time saved percentage | TBD |
| Mixed-format conversion | Class normalization success rate | TBD |

## 7. Discussion

### Strengths

- 고성능 LMM을 모든 샘플에 호출하지 않고, 사용자가 선택한 검토 단계에서만 활용할 수 있다.
- Ground Truth가 없는 상황에서도 1차 Vision 결과와 LMM 재생성 결과 간 self_consistency로 검토 우선순위를 만들 수 있다.
- 라벨 생성, 변환, 검증, 리포트를 하나의 workflow로 묶어 반복 실험과 배포가 쉽다.
- YOLO class ID와 COCO/VOC class name이 섞이는 실제 데이터셋 문제를 canonical class list로 완화한다.
- 오류 발생 시 사용자에게 파일 경로와 수정 지침을 제공해 후처리 부담을 줄인다.

### Limitations

- Ground Truth 없이 계산한 consistency는 실제 정답 정확도를 완전히 대체하지 않는다.
- LMM 재생성 비교를 많이 실행하면 API 비용과 처리 시간이 증가한다.
- 복잡한 custom schema는 자동 변환보다 사용자 지정 mapping이 필요할 수 있다.
- Streamlit은 localhost 사용을 기본으로 하며 외부 공개 서비스 형태의 인증/권한 관리는 별도 설계가 필요하다.

## 8. Conclusion

본 프로젝트는 Agentic Workflow 기반 자동 라벨링 시스템을 통해 컴퓨터 비전 데이터 준비 과정의 라벨 생성, 포맷 변환, 검증, 결과 리포트, 라벨 편집을 통합했다. Vision Specialist Model 우선 구조는 태스크별 모델 성능을 활용하고, 선택적 LMM 재생성 비교는 임계치 미달 샘플을 선별해 사용자가 효율적으로 검토할 수 있게 한다. 다중 포맷 변환 구조는 실제 데이터셋에서 발생하는 class ID/class name 불일치 문제를 완화한다.

또한 Streamlit과 Windows `setup.exe` 배포를 고려한 구현 구조를 제공함으로써 연구용 프로토타입을 실제 사용자 도구로 확장할 수 있는 기반을 마련했다. 향후 실제 데이터셋 기반 정량 평가를 통해 태스크별 specialist 모델 성능, LMM self_consistency 기반 검토 효율, 사용자 수정 시간 감소 효과를 실험적으로 검증할 예정이다.

## 9. Future Work

- 실제 Ground Truth 데이터셋 기반 ablation study 수행
- specialist-only, specialist+LMM self_consistency 방식의 비용·정확도 비교
- 사용자 수정 제안 리포트의 수정 시간 절감 효과 측정
- custom schema mapping UI 추가
- 외부 plugin 모델의 설치 자동화와 모델 weight 관리 개선
- Streamlit/desktop UI의 결과 차트와 포스터용 export 기능 강화

## 10. Acknowledgements

본 프로젝트는 컴퓨터 비전 데이터셋 구축 과정에서 발생하는 라벨링 비용, 포맷 불일치, 검증 자동화 문제를 해결하기 위한 연구 및 구현 과제로 수행되었다. 시스템 구현에는 Python 기반 workflow orchestration, Vision LMM API, Streamlit, PySide6, 표준 라벨 포맷 변환 도구가 활용되었다.

## Poster Figure Suggestions

포스터 제작 시 다음 그림을 포함하면 좋다.

1. 전체 workflow diagram: Input → Agents → Validation → Outputs
2. Hierarchical verification diagram: Low model repeated inference → consistency → high model escalation
3. Multi-format conversion diagram: YOLO/VOC/COCO → canonical class space → target formats
4. Report example: `preflight`와 `user_action_report`의 사용자 수정 지침 예시
5. Result table: specialist-only, specialist+LMM self_consistency 비교 표

## One-Sentence Contribution

태스크별 Vision Specialist Model, 선택적 LMM self_consistency 검증, 다중 라벨 포맷 변환, 사용자 수정 제안 리포트와 라벨 편집을 통합해 컴퓨터 비전 데이터 준비 과정을 자동화 가능한 workflow로 구현했다.
