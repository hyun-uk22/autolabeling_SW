# User Action Report 가이드

## 개요

변환 작업 후 생성되는 `user_action_report.json` 파일은 문제가 있는 데이터에 대해 **사용자가 무엇을 해야 하는지 명확하게 안내**합니다.

## 리포트 구조

### 1. 전체 요약 (Summary)

```json
{
  "status": "partial_success",  // success | partial_success | needs_review
  "message": "5개 파일에 문제가 발견되었습니다",
  "summary": {
    "total_records": 100,
    "clean": 95,              // 문제 없는 파일
    "needs_review": 5,        // 검토 필요한 파일
    "blocked": 2,             // 치명적 문제로 처리 불가
    "needs_attention": 3,     // 높은 우선순위 문제
    "warning": 0              // 경미한 문제
  },
  "completion_rate": "95.0%"
}
```

### 2. 상위 이슈 (Top Issues)

가장 많이 발생한 문제 TOP 5:

```json
{
  "top_issues": [
    {
      "issue_type": "coordinate_out_of_range",
      "count": 15,
      "percentage": "15.0%",
      "action": "좌표를 정규화하거나 원본 라벨을 수정해주세요"
    },
    {
      "issue_type": "missing_label",
      "count": 8,
      "percentage": "8.0%",
      "action": "각 객체에 클래스 라벨을 지정해주세요"
    }
  ]
}
```

### 3. 권장 조치사항 (Recommended Actions)

```json
{
  "recommended_actions": [
    "⚠️ 2개 파일이 블로킹 상태입니다. 우선적으로 처리가 필요합니다.",
    "🏷️ 라벨 품질 문제가 23건 발견되었습니다. 라벨링 도구 설정을 확인해주세요.",
    "📁 입력 데이터 문제가 5건 발견되었습니다. 원본 데이터를 확인해주세요."
  ]
}
```

### 4. 상세 레코드 (Detailed Records)

각 파일별 문제 상세:

```json
{
  "detailed_records": [
    {
      "image": "IMG_001.jpg",
      "status": "blocked",  // blocked | needs_attention | warning
      "total_issues": 3,
      "issues_by_severity": {
        "critical": 1,
        "high": 2,
        "medium": 0
      },
      "issues_by_category": {
        "input_data": 1,
        "label_quality": 2
      },
      "priority_actions": [
        "이미지 파일 경로를 확인하거나 파일을 복원해주세요",
        "좌표를 정규화하거나 원본 라벨을 수정해주세요"
      ],
      "detailed_issues": [
        {
          "severity": "critical",
          "category": "input_data",
          "message": "이미지 파일을 찾을 수 없습니다",
          "user_action": "이미지 파일 경로를 확인하거나 파일을 복원해주세요",
          "suggestions": [
            "파일 경로가 올바른지 확인",
            "파일이 실수로 삭제되었는지 확인",
            "상대 경로 대신 절대 경로 사용"
          ],
          "original_issue": "missing_image:path/to/IMG_001.jpg"
        }
      ]
    }
  ]
}
```

## 문제 심각도 (Severity)

| 심각도 | 의미 | 처리 방법 |
|--------|------|-----------|
| **critical** | 처리 불가능한 치명적 문제 | 즉시 수정 필요 (파일 누락, 데이터 비어있음) |
| **high** | 변환 결과에 영향을 주는 중요한 문제 | 우선적으로 수정 권장 (좌표 오류, 라벨 누락) |
| **medium** | 경미하지만 수정하면 좋은 문제 | 시간 날 때 수정 (confidence 범위) |
| **unknown** | 분류되지 않은 문제 | 수동 검토 필요 |

## 문제 카테고리 (Category)

| 카테고리 | 설명 | 예시 |
|---------|------|------|
| **input_data** | 입력 데이터 문제 | 빈 결과, 데이터 없음 |
| **file_system** | 파일 시스템 문제 | 파일 누락, 권한 오류 |
| **file_format** | 파일 포맷 문제 | 이미지 열기 실패, 손상된 파일 |
| **label_quality** | 라벨 품질 문제 | 좌표 오류, 라벨 누락, 값 범위 초과 |
| **segmentation** | 세그멘테이션 문제 | 폴리곤 포인트 부족 |
| **pose** | 포즈 추정 문제 | 키포인트 누락 |
| **output_format** | 출력 포맷 문제 | YOLO 형식 오류, 객체 없음 |

## 사용 예시

### 예시 1: 좌표 범위 초과 문제

**리포트:**
```json
{
  "severity": "high",
  "category": "label_quality",
  "message": "좌표값이 유효 범위(0-1)를 벗어났습니다",
  "user_action": "좌표를 정규화하거나 원본 라벨을 수정해주세요",
  "suggestions": [
    "라벨링 도구의 좌표 형식 확인 (픽셀 vs 정규화)",
    "변환 스크립트에서 정규화 로직 확인",
    "수동으로 좌표 재조정"
  ]
}
```

**해결 방법:**
1. 원본 라벨 파일 확인
2. 좌표가 픽셀 단위인지 정규화(0-1)인지 확인
3. 필요시 변환 스크립트 수정:
```python
# 픽셀 → 정규화
xmin_norm = xmin_pixel / image_width
ymin_norm = ymin_pixel / image_height
```

### 예시 2: 이미지 파일 누락

**리포트:**
```json
{
  "severity": "critical",
  "category": "file_system",
  "message": "이미지 파일을 찾을 수 없습니다",
  "user_action": "이미지 파일 경로를 확인하거나 파일을 복원해주세요",
  "suggestions": [
    "파일 경로가 올바른지 확인",
    "파일이 실수로 삭제되었는지 확인",
    "상대 경로 대신 절대 경로 사용"
  ]
}
```

**해결 방법:**
1. 파일 경로 확인
2. 파일 존재 여부 확인
3. 필요시 파일 복사 또는 복원

### 예시 3: YOLO 형식 오류

**리포트:**
```json
{
  "severity": "high",
  "category": "output_format",
  "message": "YOLO 파일 형식이 잘못되었습니다 (각 행은 5개 값 필요)",
  "user_action": "YOLO 형식을 확인하고 수정해주세요",
  "suggestions": [
    "형식: <class_id> <x_center> <y_center> <width> <height>",
    "파일을 수동으로 검토하고 수정"
  ]
}
```

**해결 방법:**
1. YOLO 파일 열어서 확인:
```
0 0.5 0.5 0.2 0.3
1 0.3 0.4 0.15 0.25
```
2. 각 행이 정확히 5개 값(공백으로 구분)인지 확인
3. 잘못된 행 수정

## 자동화된 복구

시스템이 자동으로 처리하는 것들:

✅ **자동 수정됨:**
- 잘못된 항목 필터링 (라벨 없는 객체 제거)
- 범위 벗어난 좌표 클리핑
- 포맷 불일치 시 대체 포맷 선택 (예: yolo → vision_json)
- 빈 값, null 값 정리

❌ **사용자 수정 필요:**
- 파일 누락/손상 (파일 복원 필요)
- 원본 데이터 비어있음 (재라벨링 필요)
- 이미지 크기 0x0 (올바른 이미지로 교체)
- 구조적 오류 (원본 수정 필요)

## 워크플로우

```
1. 변환 실행
   ↓
2. user_action_report.json 생성
   ↓
3. 리포트 확인
   ├─ status: "success" → ✅ 완료
   ├─ status: "partial_success" → ⚠️ 일부 문제, 검토 필요
   └─ status: "needs_review" → ❌ 심각한 문제, 수정 필요
   ↓
4. priority_actions 확인
   ↓
5. blocked 파일 우선 처리
   ↓
6. needs_attention 파일 검토
   ↓
7. 수정 후 재실행
```

## CLI에서 확인하기

```bash
# 리포트 확인
cat output/user_action_report.json | jq '.summary'

# TOP 이슈 확인
cat output/user_action_report.json | jq '.top_issues'

# 블로킹된 파일만 확인
cat output/user_action_report.json | jq '.detailed_records[] | select(.status=="blocked")'

# 우선순위 액션만 확인
cat output/user_action_report.json | jq '.recommended_actions[]'
```

## 문의

리포트에 나오지 않는 문제나 이해되지 않는 부분은:
1. `conversion_report.json` 전체 로그 확인
2. 원본 데이터 검증
3. 개발팀 문의 (이슈 트래커에 리포트 첨부)
