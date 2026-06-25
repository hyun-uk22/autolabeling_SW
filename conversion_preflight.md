# 변환 사전 점검 항목 정리

이 문서는 Streamlit `형식 변환` 탭의 `변환 사전 점검`에서 사용자에게 표시될 수 있는 항목을 정리한다.

사전 점검은 크게 두 단계의 정보를 보여준다.

- `실행 전 확인 사항`: 변환 실행 전에 사용자가 알아야 하는 요약 이슈
- `입력 데이터 문제 상세`: 파일별로 실제로 발견된 구체적인 검증 이슈

## 실행 전 확인 사항

| 항목 | 심각도 | 의미 | 필요 조치 |
|---|---|---|---|
| `no_label_sources` | critical | 입력 경로에서 변환 가능한 라벨 파일을 찾지 못함 | 입력 라벨 경로와 입력 파일 형식을 확인한다. |
| `no_records` | critical | 라벨 파일은 발견했지만 변환 가능한 라벨 record가 없음 | 라벨 파일이 비어 있거나 지원 형식인지 확인한다. |
| `import_failed_files` | warning | 라벨 후보 파일을 읽는 중 import/parsing 실패 | 실패 원인을 확인하고, 리포트/metrics 파일은 라벨 입력 폴더에서 분리한다. |
| `skipped_unrecognized_files` | info | 파일은 있었지만 지원 라벨 파일 형식으로 판별되지 않아 건너뜀 | 의도한 라벨이면 입력 파일 형식을 명시하거나 파일 구조를 정리한다. |
| `missing_yolo_class_mapping` | warning | YOLO txt는 찾았지만 `data.yaml`, `dataset.yaml`, `classes.txt`를 찾지 못함 | class mapping 파일을 라벨 폴더에 두거나 직접 지정한다. |
| `label_conflicts` | warning | 같은 이미지/위치 라벨에서 서로 다른 클래스명이 충돌 | class taxonomy, alias, class id mapping을 정리한다. |
| `numeric_label_normalized` | info | 숫자 클래스 라벨을 다른 포맷의 클래스명과 비교해 정규화함 | 추론된 class id와 클래스명이 의도와 맞는지 확인한다. |
| `missing_images` | warning | 라벨과 연결되는 이미지 파일을 찾지 못함 | 라벨만 변환할 수는 있지만, COCO/VOC처럼 이미지 크기가 필요한 출력은 이미지 경로를 연결한다. |
| `invalid_images` | critical | 이미지 파일을 열 수 없거나 이미지 크기를 읽을 수 없음 | 손상 이미지 또는 지원하지 않는 이미지 형식을 교체한다. |
| `missing_labels` | warning | 객체 라벨의 클래스명이 비어 있음 | 누락된 클래스명을 채운다. |
| `empty_class_list` | warning | YOLO 출력에 사용할 클래스 목록이 비어 있음 | 입력 라벨의 클래스명 또는 YOLO class mapping 파일을 확인한다. |

## 입력 데이터 문제 상세

| 항목 | 의미 | 주로 발생하는 상황 |
|---|---|---|
| `missing_image` | 라벨에 대응되는 이미지 파일 없음 | 라벨 파일만 있고 원본 이미지가 없거나 파일명이 다름 |
| `image_open_failed` | 이미지 파일 열기 실패 | 손상 이미지 또는 지원하지 않는 이미지 형식 |
| `invalid_image_size` | 이미지 width/height가 유효하지 않음 | 이미지 크기를 읽을 수 없거나 0 이하 |
| `missing_label` | 클래스명 누락 | 객체 annotation에 label/name/category 정보가 없음 |
| `coordinate_out_of_range` | 좌표가 0~1 범위를 벗어남 | 정규화 좌표 대신 픽셀 좌표가 들어가 있음 |
| `invalid_box_order` | bbox 좌표 순서 오류 | `xmin >= xmax` 또는 `ymin >= ymax` |
| `confidence_out_of_range` | confidence가 0~1 범위를 벗어남 | percentage 값을 그대로 넣었거나 비정상 confidence |
| `too_few_points` | polygon point가 3개 미만 | segmentation polygon이 선/점 형태로 저장됨 |
| `missing_name` | pose keypoint 이름 누락 | pose keypoint에 name 필드가 없음 |
| `empty_keypoints` | pose keypoint가 비어 있음 | pose 객체는 있으나 keypoint 목록이 없음 |
| `missing_text` | OCR text가 비어 있음 | text region은 있으나 OCR 문자열이 없음 |
| `missing_track_id` | tracking label의 track id 누락 | tracking 객체에 track_id가 없음 |
| `unrecognized_txt_schema` | TXT가 YOLO 라벨 구조로 보이지 않음 | txt 행이 `class x y w h` 구조가 아님 |
| `unrecognized_xml_schema` | XML이 Pascal VOC 구조로 보이지 않음 | `annotation/object/bndbox` 구조가 없음 |
| `unrecognized_json_schema` | JSON이 COCO/Vision/Generic 구조로 보이지 않음 | 임의 JSON 구조이거나 필수 키가 없음 |
| `unrecognized_jsonl_schema` | JSONL 구조를 지원 포맷으로 판별하지 못함 | 각 행이 지원 라벨 record 구조가 아님 |
| `unrecognized_csv_schema` | CSV 컬럼 구성이 지원 bbox 라벨 구조와 다름 | image, label, bbox 좌표 컬럼을 찾지 못함 |
| `schema_read_failed` | 파일 읽기 또는 schema 해석 실패 | 인코딩 오류, JSON/XML 파싱 오류 등 |
| `import_failed_files` | 라벨 파일 import 자체 실패 | 지정한 source format과 실제 파일 구조가 다름 |
| `skipped_unrecognized_files` | 라벨 후보에서 제외된 파일 | 결과 리포트, metrics, metadata 등 라벨이 아닌 파일 |

## 출력 포맷 관련 검증 이슈

변환 사전 점검 이후 변환을 실행하면 출력 파일 검증 단계에서 다음 이슈가 추가로 표시될 수 있다.

| 항목 | 의미 |
|---|---|
| `missing_output_path` | 출력 경로가 생성되지 않음 |
| `missing_output_file` | 출력 파일이 생성되지 않음 |
| `empty_output_file` | 출력 파일이 비어 있음 |
| `output_parse_failed` | 출력 파일을 다시 읽을 수 없음 |
| `no_label_rows` | YOLO 출력 txt에 라벨 행이 없음 |
| `invalid_row_shape` | YOLO 행 구조가 올바르지 않음 |
| `non_numeric_row` | YOLO 행에 숫자가 아닌 값이 있음 |
| `negative_class_id` | YOLO class id가 음수 |
| `invalid_box_size` | bbox width 또는 height가 0 이하 |
| `no_objects` | Pascal VOC object가 없음 |
| `invalid_object` | Pascal VOC object 구조가 잘못됨 |
| `non_numeric_box` | Pascal VOC bbox 좌표가 숫자가 아님 |
| `no_images` | COCO images 배열이 없음 |
| `no_annotations` | COCO annotations 배열이 없음 |
| `no_categories` | COCO categories 배열이 없음 |
| `invalid_bbox` | COCO bbox 구조가 잘못됨 |
| `non_numeric_bbox` | COCO bbox 값이 숫자가 아님 |
| `invalid_bbox_size` | COCO bbox width 또는 height가 0 이하 |
| `missing_category_id` | COCO annotation의 category_id가 없음 |
| `no_label_records` | Vision JSON에 유효한 라벨 record가 없음 |
| `no_labels` | 클래스 목록이 비어 있음 |

## 현재 동작 기준 주의사항

- `missing_image`는 사전 점검에서 `warning`으로 처리된다. 즉 이미지가 없어도 라벨 파일만으로 가능한 변환은 계속 진행할 수 있다.
- 다만 COCO/Pascal VOC처럼 이미지 크기가 필요한 출력은 이미지가 없으면 일부 출력이 제한되거나 export issue가 남을 수 있다.
- 입력 라벨 경로에 `converted`, `reports`, `visualized`, `run_metrics.csv`, `run_summary.json` 같은 결과 파일이 섞이면 `skipped_unrecognized_files` 또는 `import_failed_files`가 증가할 수 있다.
- YOLO 입력에서 class mapping이 없으면 class id가 `"0"`, `"1"` 같은 문자열 라벨로 처리될 수 있다.
- 여러 포맷이 같은 폴더에 함께 있으면 자동 탐색이 가능한 라벨을 모두 읽고 이미지 기준으로 병합한다. 이때 class id/class name 충돌이 있으면 `label_conflicts` 또는 `numeric_label_normalized`가 발생할 수 있다.
