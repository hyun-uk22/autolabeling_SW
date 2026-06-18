# 라벨 형식 변환 명세

## 1. 목적

이 문서는 외부 비전 데이터셋의 라벨을 프로젝트 내부 공통 표현으로 읽고, 검증한 뒤, 사용자가 선택한 출력 형식으로 변환하는 기능의 상세 명세와 작동 과정을 설명한다.

라벨 변환 기능의 실행 진입점은 `convert_labels.py`이며, 내부적으로 다음 세 모듈을 사용한다.

| 모듈 | 책임 |
| --- | --- |
| `src/utils/label_importer.py` | 입력 포맷 추론, 파싱, 내부 공통 모델 변환 |
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
입력 포맷 결정
  - 명시적 --source_format
  - 또는 auto 휴리스틱
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
| `--strict` | 아니오 | `False` | validation issue가 하나라도 있으면 해당 레코드 skip |

## 5. 상세 작동 과정

### 5.1 입력 포맷 결정

`--source_format`이 `auto`가 아니면 지정한 importer를 직접 사용한다.

`auto`인 경우 `infer_label_format()`이 다음 규칙을 적용한다.

디렉터리 입력 우선순위:

1. `.xml` 파일이 하나라도 있으면 `pascal_voc`
2. 그렇지 않고 `.txt` 파일이 하나라도 있으면 `yolo`
3. 그렇지 않고 `.jsonl` 파일이 하나라도 있으면 `vision_json`
4. 어느 조건에도 맞지 않으면 포맷 추론 실패

단일 파일 입력:

| 확장자 | 추론 결과 |
| --- | --- |
| `.xml` | `pascal_voc` |
| `.jsonl` | `vision_json` |
| `.csv` | `csv` |
| `.txt` | `yolo` |
| `.json` | COCO 필수 key가 있으면 `coco`, 아니면 `generic_json` |

COCO 판정에 사용하는 필수 key는 `images`, `annotations`, `categories`다.

현재 `auto`는 한 입력 디렉터리 안의 여러 포맷을 각각 판별해 혼합 처리하지 않는다. XML과 TXT가 함께 있으면 우선순위가 높은 Pascal VOC importer만 선택한다. 혼합 포맷 데이터셋은 포맷별로 디렉터리를 분리하거나 `--source_format`을 명시해 별도로 실행해야 한다.

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

구조:

```json
{
  "input": "data/external_labels",
  "source_format": "auto",
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

`source_format`에는 CLI에서 입력한 값이 기록된다. `auto`를 사용한 경우 현재 report는 최종 추론된 실제 포맷명 대신 `auto`를 기록한다.

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

이미지별 validation issue는 전체 실행을 중단하지 않고 report에 기록한다. 단, importer 자체에서 예외가 발생하면 현재는 전체 실행이 중단된다.

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

- 한 디렉터리에 섞인 여러 입력 포맷의 레코드별 자동 판별
- LLM을 사용한 임의 schema 의미 추론
- validation 실패 시 LLM을 통한 자동 코드 생성/재변환
- AIHub 데이터셋별 전용 schema adapter
- arbitrary pixel-coordinate CSV/JSON의 width/height 컬럼 기반 정규화
- COCO RLE mask import
- Vision JSONL의 classification/pose/OCR/tracking 완전한 round-trip import
- 원본 class id를 강제로 유지하는 class mapping 파일
- 변환 전후 의미적 동등성 자동 평가
- importer 오류의 이미지별 격리 처리

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
