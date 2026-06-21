# 라벨 형식 변환 명세

## 1. 목적

이 문서는 외부 비전 데이터셋의 라벨을 프로젝트 내부 공통 표현으로 읽고, 검증한 뒤, 사용자가 선택한 출력 형식으로 변환하는 기능의 상세 명세와 작동 과정을 설명한다.

라벨 변환 기능의 실행 진입점은 `convert_labels.py`이며, 내부적으로 다음 세 모듈을 사용한다.

| 모듈 | 책임 |
| --- | --- |
| `src/utils/label_importer.py` | 파일별 입력 포맷 판별, 파싱, 내부 공통 모델 변환, 중복 병합 |
| `src/utils/label_validator.py` | 이미지와 라벨의 구조적 유효성 검사 |
| `src/utils/format_converter.py` | YOLO, Pascal VOC, COCO, Vision JSONL, custom 포맷 출력 |

이 기능은 새 라벨을 생성하는 자동 라벨링 파이프라인과 구분된다. `main.py`는 이미지에서 새 라벨을 생성하고, `convert_labels.py`는 이미 존재하는 라벨을 다른 형식으로 변환한다.

## 2. 범위

### 2.1 현재 지원 입력 형식

| `source_format` | 파일 형태 | 현재 지원 데이터 |
| --- | --- | --- |
| `auto` | 파일 또는 디렉터리 | 확장자와 JSON 구조를 사용한 휴리스틱 추론 |
| `yolo` | `.txt` 파일 또는 디렉터리 | object detection bbox |
| `pascal_voc` | `.xml` 파일 또는 디렉터리 | object detection bbox |
| `coco` | dataset 단위 `.json` | bbox와 polygon segmentation |
| `vision_json` | `.jsonl` 파일 또는 디렉터리 | 현재 importer 기준 bbox와 polygon segmentation |
| `csv` | `.csv` 파일 | 정규화 bbox |
| `generic_json` | `.json` 파일 | 중첩 구조에서 발견한 정규화 bbox |

### 2.2 현재 지원 출력 형식

| 출력 형식 | 산출물 | 주요 용도 |
| --- | --- | --- |
| `yolo` | 이미지별 `.txt`, `classes.txt` | YOLO object detection 학습 |
| `pascal_voc` | 이미지별 `.xml` | Pascal VOC object detection |
| `coco` | `coco_annotations.json` | COCO bbox/polygon segmentation |
| `vision_json` | `vision_annotations.jsonl` | 프로젝트 전체 태스크 공통 표현 |
| `custom` | 이미지별 사용자 지정 확장자 파일 | 템플릿 기반 외부 규격 연동 |
| `all` | YOLO, Pascal VOC, COCO, Vision JSONL | 기본 제공 출력 일괄 생성 |

`all`에는 `custom`이 포함되지 않는다. custom 출력은 반드시 템플릿 파일을 제공해야 하기 때문이다.

## 3. 전체 아키텍처

```text
외부 라벨 파일/디렉터리
        |
        v
라벨 파일 재귀 탐색 및 파일별 포맷 결정
  - 명시적 --source_format
  - 또는 auto schema fingerprint
        |
        v
포맷별 Importer
  - YOLO / VOC / COCO
  - Vision JSONL / CSV / generic JSON
        |
        v
내부 공통 표현 DetectionResult
  - normalized coordinate 0.0~1.0
  - label + confidence
  - boxes / segments / poses / texts / tracks 등
        |
        v
이미지 stem 기준 그룹화 및 중복 병합
  - 같은 label + IoU 임계값
  - confidence, source priority 순서로 선택
  - 다른 label 충돌은 보존하고 report에 기록
        |
        v
이미지 경로 탐색 및 Validation
  - 이미지 존재/크기
  - 좌표 범위/순서
  - 필수 label/keypoint/track id
        |
        +---------------------------+
        |                           |
        | blocking issue            | valid 또는 non-strict issue
        v                           v
      skip                    LabelExportWriter
                                    |
                                    v
                       선택한 타겟 포맷 파일 생성
                                    |
                                    v
                       conversion_report.json 생성
```

## 4. CLI 명세

기본 형식:

```bash
python convert_labels.py \
  --input <라벨 파일 또는 디렉터리> \
  --img_dir <원본 이미지 디렉터리> \
  --out_dir <변환 결과 디렉터리> \
  [--source_format auto] \
  [--target_formats yolo] \
  [--classes classes.txt] \
  [--custom_label_template template.txt] \
  [--custom_label_extension .json] \
  [--duplicate_iou 0.85] \
  [--strict]
```

Windows PowerShell 예시:

```powershell
python convert_labels.py `
  --input data/external_labels `
  --img_dir data/raw `
  --out_dir data/converted `
  --source_format auto `
  --target_formats yolo,pascal_voc,coco,vision_json
```

CLI 옵션:

| 옵션 | 필수 | 기본값 | 설명 |
| --- | --- | --- | --- |
| `--input` | 예 | 없음 | 입력 라벨 파일 또는 디렉터리 |
| `--img_dir` | 예 | 없음 | 원본 이미지 디렉터리 |
| `--out_dir` | 예 | 없음 | 출력 라벨과 report 저장 디렉터리 |
| `--source_format` | 아니오 | `auto` | 입력 포맷을 명시하거나 자동 추론 |
| `--target_formats` | 아니오 | `yolo` | 쉼표로 구분한 출력 포맷 목록 |
| `--classes` | 아니오 | 없음 | YOLO class id를 이름으로 복원할 `classes.txt` |
| `--custom_label_template` | 아니오 | 없음 | custom 출력 템플릿 경로 |
| `--custom_label_extension` | 아니오 | `.json` | custom 출력 파일 확장자 |
| `--duplicate_iou` | 아니오 | `0.85` | 혼합 입력에서 같은 클래스 공간 라벨을 중복으로 판단할 IoU 기준 |
| `--strict` | 아니오 | `False` | validation issue가 하나라도 있으면 해당 레코드 skip |

## 5. 상세 작동 과정

### 5.1 입력 포맷 결정

`--source_format`이 `auto`가 아니면 지정한 importer를 직접 사용한다.

`auto`에서 단일 파일은 `infer_label_format()`으로 판별한다. 디렉터리는 하위 파일을 재귀 탐색하고 각 파일에 schema fingerprint를 적용한다. 디렉터리 전체를 하나의 포맷으로 고정하지 않는다.

디렉터리 내 파일별 판별 규칙:

| 후보 | 판별 기준 |
| --- | --- |
| Pascal VOC | `.xml`이며 root element가 `annotation` |
| YOLO | `.txt`이며 비어 있지 않은 모든 행이 숫자 5개이고 좌표 4개가 `0~1`; 빈 파일은 같은 stem 이미지가 있을 때 허용 |
| COCO | `.json`이며 최상위에 `images`, `annotations`, `categories`가 존재 |
| Vision JSONL | `.jsonl` 첫 레코드에 `image_name` 또는 `task_type` 존재 |
| CSV | 이미지 이름 컬럼과 `xmin`, `ymin`, `xmax`, `ymax` 컬럼 존재 |

`classes.txt`는 라벨 소스가 아니라 YOLO class mapping으로 취급한다. 알려지지 않은 JSON/XML/TXT/JSONL/CSV schema는 임의로 해석하지 않고 `input_summary.skipped_files`에 이유와 함께 기록한다.

단일 파일 입력:

| 확장자 | 추론 결과 |
| --- | --- |
| `.xml` | `pascal_voc` |
| `.jsonl` | `vision_json` |
| `.csv` | `csv` |
| `.txt` | `yolo` |
| `.json` | COCO 필수 key가 있으면 `coco`, 아니면 `generic_json` |

COCO 판정에 사용하는 필수 key는 `images`, `annotations`, `categories`다.

혼합 디렉터리에서는 발견된 YOLO, Pascal VOC, COCO, Vision JSONL, CSV 소스를 각각 importer로 읽은 뒤 하나의 내부 레코드 집합으로 통합한다. `--source_format`을 명시하면 이전과 같이 해당 importer만 사용한다.

### 5.1.1 이미지별 병합과 충돌 처리

이미지 파일명의 확장자를 제외한 basename을 case-insensitive key로 사용해 여러 소스의 레코드를 그룹화한다.

- 같은 클래스의 bbox 또는 polygon bounding extent가 `--duplicate_iou` 이상 겹치면 중복으로 판단한다.
- confidence가 다르면 높은 annotation을 유지한다.
- confidence가 같으면 `COCO > Pascal VOC > YOLO > Vision JSONL > CSV > generic JSON` 순서로 유지한다.
- 같은 위치에서 클래스가 다르면 둘 다 보존하고 `input_summary.merge.conflicts`에 기록한다.
- classification은 label 기준, pose/OCR/tracking은 현재 완전 동일한 canonical payload 기준으로 중복을 제거한다.
- 병합된 레코드의 `plugin_metadata.conversion_sources`에 원본 포맷과 파일 경로를 남긴다.

이 병합은 여러 소스를 한 실행에서 읽고 exporter를 한 번만 실행하므로, 소스 포맷별로 같은 출력 디렉터리에 반복 실행할 때 발생하던 파일 덮어쓰기를 피한다.

### 5.2 이미지 파일 검색

각 라벨 레코드는 이미지 파일명과 연결되어야 한다. `find_image_path()`는 다음 순서로 이미지를 찾는다.

1. `--img_dir/<원래 이미지 파일명>` 정확히 일치하는 파일 검색
2. 파일명의 stem은 유지하고 확장자를 순서대로 대체
   - `.jpg`
   - `.jpeg`
   - `.png`
   - `.webp`
   - `.bmp`
3. 찾지 못하면 원래 조합 경로를 반환하고 validation에서 `missing_image`로 처리

YOLO 라벨은 이미지 확장자 정보를 포함하지 않으므로 importer가 우선 `<label stem>.jpg`를 생성하고, 이후 이미지 검색 단계가 실제 확장자를 탐색한다.

### 5.3 내부 공통 표현

모든 importer는 `(image_name, DetectionResult)` 레코드 목록을 반환한다.

내부 좌표 기준:

```text
xmin, ymin, xmax, ymax: 0.0~1.0 normalized coordinate
xmin < xmax
ymin < ymax
confidence: 0.0~1.0
```

대표 내부 bbox:

```json
{
  "task_type": "object_detection",
  "boxes": [
    {
      "label": "person",
      "xmin": 0.10,
      "ymin": 0.15,
      "xmax": 0.60,
      "ymax": 0.90,
      "confidence": 1.0
    }
  ]
}
```

내부 모델은 classification, detection, segmentation, pose, OCR, tracking 필드를 모두 보유한다. 다만 현재 기존 라벨 importer가 완전히 복원하는 태스크는 detection과 COCO/Vision JSONL polygon segmentation 중심이다. pose/OCR/tracking/classification 입력의 완전한 round-trip import는 향후 확장 항목이다.

### 5.4 좌표 정규화

Pascal VOC와 COCO는 이미지 크기 정보를 사용해 pixel 좌표를 normalized 좌표로 변환한다.

```text
normalized_x = pixel_x / image_width
normalized_y = pixel_y / image_height
```

YOLO 입력은 이미 normalized center 좌표이므로 다음처럼 corner 좌표로 변환한다.

```text
xmin = x_center - width / 2
ymin = y_center - height / 2
xmax = x_center + width / 2
ymax = y_center + height / 2
```

CSV와 generic JSON은 별도 image width/height 컬럼을 사용하지 않는다. 좌표 값이 `1.0`보다 크면 공통 `normalize_coordinate()`가 `0~1000` 스케일로 간주해 `1000`으로 나눈다. 따라서 임의 pixel 좌표를 가진 CSV/JSON을 정확히 변환하려면 먼저 normalized 또는 0~1000 좌표로 준비해야 한다.

### 5.5 Validation

`validate_result()`는 이미지별 변환 결과를 검사하고 문자열 issue 목록을 반환한다.

공통 검사:

- 전체 라벨 개수가 0이면 `empty_result`
- 이미지가 없으면 `missing_image:<path>`
- 이미지 열기 실패 시 `image_open_failed:<error>`
- 이미지 width 또는 height가 0 이하이면 `invalid_image_size`

bbox 계열 검사:

- label 또는 text 누락
- 좌표가 `0.0~1.0` 범위를 벗어남
- `xmin >= xmax` 또는 `ymin >= ymax`

segmentation 검사:

- label 누락
- polygon point가 3개 미만
- polygon point 좌표 범위 오류

pose 검사:

- keypoint가 없음
- keypoint name 누락
- keypoint 좌표 범위 오류

tracking 검사:

- bbox 오류
- `track_id` 누락

### 5.6 Skip 정책

다음 issue는 출력 포맷 변환에 이미지 크기가 필요하므로 `--strict` 여부와 관계없이 항상 skip한다.

- `missing_image:*`
- `image_open_failed:*`
- `invalid_image_size`

그 외 issue 처리:

| 모드 | 동작 |
| --- | --- |
| 기본 모드 | issue를 report에 기록하고 변환 계속 |
| `--strict` | issue가 하나라도 있으면 해당 레코드 skip |

현재 validator는 오류를 자동 수정하거나 LLM에 재질의하지 않는다. 구조적 오류를 탐지하고 report에 남기는 역할까지 수행한다.

### 5.7 타겟 포맷 출력

`LabelExportWriter`는 지정된 출력 포맷을 같은 실행에서 동시에 생성한다.

#### YOLO

이미지별 `.txt`:

```text
<class_id> <x_center> <y_center> <width> <height>
```

class id는 실행 중 class label이 처음 발견된 순서로 부여되며 `classes.txt`에 저장된다. 원본 데이터셋의 class id가 그대로 유지된다고 보장하지 않는다.

#### Pascal VOC

이미지별 XML을 생성하고 bbox 좌표를 원본 이미지 크기 기준 pixel 좌표로 복원한다.

주요 필드:

- `filename`
- `path`
- `size/width`, `size/height`, `size/depth`
- `object/name`
- `object/confidence`
- `object/bndbox`

#### COCO

전체 실행 단위 `coco_annotations.json`을 생성한다.

```text
images
annotations
categories
```

bbox는 `[x, y, width, height]` pixel 좌표로 저장한다. polygon segment는 COCO `segmentation` 배열로 저장한다.

#### Vision JSONL

전체 실행 단위 `vision_annotations.jsonl`을 생성한다. 이미지 한 장이 JSON 한 줄에 대응한다.

이 포맷은 다음 정보를 보존할 수 있다.

- task type
- source model
- consistency/confidence/uncertainty
- plugin score와 metadata
- classification
- bbox
- polygon segmentation
- pose
- OCR text region
- tracking

#### Custom Template

custom 출력은 Python `str.format` 방식 템플릿을 사용한다.

사용 가능한 placeholder:

| placeholder | 값 |
| --- | --- |
| `{image_name}` | 이미지 파일명 |
| `{image_path}` | 이미지 경로 |
| `{image_width}` | 이미지 width |
| `{image_height}` | 이미지 height |
| `{source_model}` | 라벨 생성 모델 |
| `{consistency_score}` | consistency score |
| `{mean_confidence}` | 평균 confidence |
| `{uncertainty_score}` | uncertainty score |
| `{object_count}` | bbox 개수 |
| `{objects_json}` | bbox의 전체 export 정보 |
| `{boxes_json}` | normalized bbox 목록 |
| `{labels_json}` | bbox label 목록 |
| `{result_json}` | 전체 task 결과 JSON |

JSON/XML 리터럴 중괄호는 `{{`, `}}`로 escape해야 한다.

예시:

```json
{{
  "image": "{image_name}",
  "size": [{image_width}, {image_height}],
  "annotations": {result_json}
}}
```

실행:

```powershell
python convert_labels.py `
  --input data/external/annotations.json `
  --img_dir data/raw `
  --out_dir data/converted `
  --source_format generic_json `
  --target_formats custom `
  --custom_label_template templates/custom.json `
  --custom_label_extension .json
```

### 5.8 Finalize

모든 이미지별 저장이 끝난 뒤 `LabelExportWriter.finalize()`가 dataset 단위 산출물을 생성한다.

- `classes.txt`
- `coco_annotations.json`
- `vision_annotations.jsonl`

이미지별 파일과 달리 이 파일들은 처리 루프 중간이 아니라 전체 레코드 처리가 끝난 뒤 확정된다.

### 5.9 Conversion Report

모든 실행은 `<out_dir>/conversion_report.json`을 생성한다.
LangGraph와 Streamlit 변환은 같은 디렉터리에 `<out_dir>/user_action_report.json`도 생성한다.

구조:

```json
{
  "input": "data/external_labels",
  "source_format": "auto",
  "input_summary": {
    "mode": "mixed_auto",
    "files_scanned": 108,
    "sources_discovered": 102,
    "sources_processed": 102,
    "sources_failed": 0,
    "formats": {"coco": 1, "pascal_voc": 1, "yolo": 100},
    "records_before_merge": 200,
    "records_after_merge": 100,
    "processed_files": [],
    "failed_files": [],
    "skipped_files": [],
    "merge": {
      "duplicate_iou": 0.85,
      "duplicates_removed": 100,
      "conflicts": [],
      "image_identity_collisions": []
    }
  },
  "target_formats": ["yolo", "coco"],
  "records_read": 100,
  "records_converted": 97,
  "validation": {
    "total_records": 100,
    "valid_records": 95,
    "failed_records": 5,
    "issue_counts": {
      "missing_image": 3,
      "empty_result": 2
    }
  },
  "artifacts": {
    "classes": "data/converted/classes.txt",
    "coco": "data/converted/coco_annotations.json"
  },
  "records": [
    {
      "image": "sample.jpg",
      "issues": []
    }
  ]
}
```

`source_format`에는 CLI 입력값을 기록한다. 실제 발견된 포맷과 파일별 처리 여부는 `input_summary`에서 확인한다. `records_before_merge`와 `records_after_merge`를 비교하면 이미지별 통합 정도를 알 수 있고, 누락 후보는 `failed_files`와 `skipped_files`로 확인할 수 있다.

LangGraph·Streamlit의 `conversion_report.json`에는 기존 필드에 다음 항목을 추가로 기록한다.

- `report_version`: 리포트 확장 스키마 버전
- `export_validation`: 이미지별 출력 파일과 dataset artifact 재검증 결과
- `user_action_report`: critical/high/medium 분류, 완료율, 우선 조치와 문제 파일
- `dataset_insight`: `DatasetInsightAgent`가 최종 변환 대상에서 계산한 클래스 분포, 설정 가능한 불균형 기준과 희소 클래스 수집·oversampling·augmentation 제안
- `exports`: 이미지별 출력 경로와 audit issue

기존 `input_summary`, `validation`, `records`, `artifacts` 필드는 유지된다.

## 6. 포맷별 변환 예시

### 6.1 YOLO에서 COCO로 변환

```powershell
python convert_labels.py `
  --input data/yolo_labels `
  --img_dir data/images `
  --out_dir data/converted/coco `
  --source_format yolo `
  --classes data/yolo_labels/classes.txt `
  --target_formats coco
```

### 6.2 Pascal VOC에서 YOLO로 변환

```powershell
python convert_labels.py `
  --input data/voc/Annotations `
  --img_dir data/voc/JPEGImages `
  --out_dir data/converted/yolo `
  --source_format pascal_voc `
  --target_formats yolo
```

### 6.3 COCO에서 YOLO와 Vision JSONL로 동시 변환

```powershell
python convert_labels.py `
  --input data/coco/annotations/instances_train.json `
  --img_dir data/coco/images `
  --out_dir data/converted/multi `
  --source_format coco `
  --target_formats yolo,vision_json
```

### 6.4 CSV에서 Pascal VOC로 strict 변환

CSV 예시:

```csv
image,label,xmin,ymin,xmax,ymax,confidence
sample.jpg,person,0.10,0.15,0.60,0.90,0.95
```

```powershell
python convert_labels.py `
  --input data/external/labels.csv `
  --img_dir data/images `
  --out_dir data/converted/voc `
  --source_format csv `
  --target_formats pascal_voc `
  --strict
```

## 7. 실패 처리

현재 CLI는 다음 오류를 실행 실패로 처리한다.

- 입력 포맷을 추론할 수 없음
- 입력 JSON/XML이 문법적으로 손상됨
- 지원하지 않는 source/target format
- custom 포맷을 요청했지만 템플릿이 없음
- pixel 좌표 변환에 필요한 width/height가 0 이하
- 출력 파일 쓰기 실패

이미지별 validation issue는 전체 실행을 중단하지 않고 report에 기록한다. 혼합 자동 모드에서 특정 소스 importer가 실패하면 다른 소스는 계속 처리하고 해당 파일을 `input_summary.failed_files`에 기록한다. 명시적 단일 포맷 입력도 실패 정보가 report에 반영되지만, 읽힌 레코드가 없을 수 있다.

## 8. 정확성 및 재현성 주의사항

### 8.1 클래스 ID

YOLO 출력 class id는 importer의 원본 id가 아니라 변환 실행 중 발견된 label 순서로 재구성된다. GT와 prediction을 비교할 때 양쪽 `classes.txt`의 class 순서를 반드시 일치시켜야 한다.

### 8.2 좌표 반올림

normalized 좌표를 Pascal VOC/COCO pixel 좌표로 변환할 때 `round()`를 사용한다. 따라서 포맷을 여러 번 왕복 변환하면 최대 수 pixel 수준의 차이가 생길 수 있다.

### 8.3 Confidence

YOLO와 Pascal VOC 표준 GT는 confidence를 제공하지 않는 경우가 많아 importer가 기본값 `1.0`을 사용한다. COCO의 `score`가 있으면 해당 값을 사용한다.

### 8.4 Empty Annotation

현재 COCO importer는 annotation이 하나도 없는 image를 결과 목록에 생성하지 않는다. 따라서 negative image를 보존해야 하는 데이터셋은 importer 확장이 필요하다.

### 8.5 COCO RLE

현재 COCO segmentation importer는 polygon list를 지원하지만 RLE segmentation dict는 복원하지 않는다.

## 9. 현재 미지원 또는 제한 사항

- LLM을 사용한 임의 schema 의미 추론
- 혼합 디렉터리에서 등록되지 않은 사용자 정의 JSON schema 자동 변환
- validation 실패 시 LLM을 통한 자동 코드 생성/재변환
- AIHub 데이터셋별 전용 schema adapter
- arbitrary pixel-coordinate CSV/JSON의 width/height 컬럼 기반 정규화
- COCO RLE mask import
- Vision JSONL의 classification/pose/OCR/tracking 완전한 round-trip import
- 원본 class id를 강제로 유지하는 class mapping 파일
- 변환 전후 의미적 동등성 자동 평가
- 같은 basename을 사용하는 서로 다른 하위 디렉터리 이미지의 완전한 구분

따라서 현재의 `generic_json`은 모든 임의 JSON을 이해하는 LLM 기반 변환기가 아니다. 정확한 key인 `xmin`, `ymin`, `xmax`, `ymax`가 중첩 JSON 안에 있을 때 이를 재귀적으로 발견하는 휴리스틱 importer다.

## 10. 확장 방향

### 10.1 Importer 추가

새 포맷은 다음 계약을 따르는 함수를 `label_importer.py`에 추가한다.

```python
def import_new_format(input_path: str) -> list[tuple[str, DetectionResult]]:
    ...
```

그 후 다음 위치에 연결한다.

1. `convert_labels.py`의 `--source_format` choices
2. `import_labels()` dispatch
3. 필요한 경우 `infer_label_format()`
4. README 및 이 문서의 지원 형식 표

### 10.2 LLM 기반 Schema Adapter

향후 임의 schema를 지원하려면 다음 구조가 권장된다.

```text
샘플 레코드/JSON Schema 추출
        |
        v
LLM이 field mapping proposal 생성
        |
        v
허용된 mapping DSL로 제한
        |
        v
deterministic converter 실행
        |
        v
validator + 샘플 round-trip 검사
        |
        v
실패 시 mapping만 재생성
```

LLM이 임의 Python 코드를 직접 실행하는 방식보다, source field와 canonical field 사이의 mapping DSL을 생성하게 하는 편이 보안성과 재현성이 높다.

### 10.3 Validation 재시도

향후 자동 재시도는 issue 유형별로 분리하는 것이 적절하다.

| issue | 권장 처리 |
| --- | --- |
| `missing_image` | 파일 stem/manifest 재탐색 |
| `empty_result` | source schema 재분석 |
| coordinate error | coordinate system 재판정 |
| missing label | category table 또는 상위 객체 참조 |
| malformed polygon | polygon 복구 또는 bbox fallback |

## 11. 관련 파일

| 파일 | 설명 |
| --- | --- |
| `convert_labels.py` | 변환 CLI 및 전체 orchestration |
| `src/utils/label_importer.py` | 입력 포맷 importer |
| `src/utils/label_validator.py` | validation 및 summary |
| `src/utils/format_converter.py` | 출력 exporter |
| `src/core/models.py` | canonical label data model |
| `README.md` | 설치 및 일반 사용법 |
| `changelog.md` | 변경 이력 |
