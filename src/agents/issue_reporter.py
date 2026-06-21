"""
Issue Reporter - 문제가 있는 데이터에 대해 사용자가 취해야 할 액션을 명확하게 제시
"""
from typing import Dict, List, Optional


class IssueActionReporter:
    """문제를 분석하고 사용자가 해야 할 작업을 제안"""

    ISSUE_ACTIONS = {
        # 입력 데이터 문제
        "empty_result": {
            "severity": "critical",
            "category": "input_data",
            "message": "라벨 데이터가 비어있습니다",
            "user_action": "원본 데이터를 확인하고 다시 라벨링이 필요합니다",
            "suggestions": [
                "이미지에 객체가 실제로 있는지 확인",
                "라벨링 도구에서 제대로 저장되었는지 확인",
                "다른 포맷으로 내보내기 시도"
            ]
        },
        "missing_image": {
            "severity": "critical",
            "category": "file_system",
            "message": "이미지 파일을 찾을 수 없습니다",
            "user_action": "이미지 파일 경로를 확인하거나 파일을 복원해주세요",
            "suggestions": [
                "파일 경로가 올바른지 확인",
                "파일이 실수로 삭제되었는지 확인",
                "상대 경로 대신 절대 경로 사용"
            ]
        },
        "image_open_failed": {
            "severity": "critical",
            "category": "file_format",
            "message": "이미지 파일을 열 수 없습니다",
            "user_action": "이미지 파일이 손상되지 않았는지 확인해주세요",
            "suggestions": [
                "이미지 뷰어로 파일이 열리는지 확인",
                "파일 형식이 지원되는지 확인 (PNG, JPG, JPEG)",
                "파일을 다시 내보내거나 변환"
            ]
        },
        "invalid_image_size": {
            "severity": "critical",
            "category": "file_format",
            "message": "이미지 크기가 유효하지 않습니다 (0x0)",
            "user_action": "올바른 이미지 파일로 교체해주세요",
            "suggestions": [
                "이미지 파일이 완전히 다운로드되었는지 확인",
                "이미지를 다시 생성하거나 복원"
            ]
        },

        # 라벨 품질 문제
        "missing_label": {
            "severity": "high",
            "category": "label_quality",
            "message": "라벨 이름이 누락되었습니다",
            "user_action": "각 객체에 클래스 라벨을 지정해주세요",
            "suggestions": [
                "라벨링 도구에서 클래스 이름 확인",
                "classes.txt 파일이 올바른지 확인"
            ]
        },
        "coordinate_out_of_range": {
            "severity": "high",
            "category": "label_quality",
            "message": "좌표값이 유효 범위(0-1)를 벗어났습니다",
            "user_action": "좌표를 정규화하거나 원본 라벨을 수정해주세요",
            "suggestions": [
                "라벨링 도구의 좌표 형식 확인 (픽셀 vs 정규화)",
                "변환 스크립트에서 정규화 로직 확인",
                "수동으로 좌표 재조정"
            ]
        },
        "invalid_box_order": {
            "severity": "high",
            "category": "label_quality",
            "message": "박스 좌표 순서가 잘못되었습니다 (xmin >= xmax 또는 ymin >= ymax)",
            "user_action": "좌표 순서를 올바르게 수정해주세요",
            "suggestions": [
                "xmin < xmax, ymin < ymax 되도록 수정",
                "라벨링 도구에서 박스를 다시 그리기"
            ]
        },
        "confidence_out_of_range": {
            "severity": "medium",
            "category": "label_quality",
            "message": "confidence 값이 0-1 범위를 벗어났습니다",
            "user_action": "confidence 값을 확인하고 수정해주세요",
            "suggestions": [
                "값을 0-1 사이로 정규화",
                "백분율(0-100)을 소수(0-1)로 변환"
            ]
        },
        "too_few_points": {
            "severity": "high",
            "category": "segmentation",
            "message": "폴리곤 포인트가 3개 미만입니다",
            "user_action": "최소 3개 이상의 포인트로 폴리곤을 다시 그려주세요",
            "suggestions": [
                "세그멘테이션을 다시 작성",
                "너무 간단한 폴리곤은 박스로 변환"
            ]
        },
        "empty_keypoints": {
            "severity": "high",
            "category": "pose",
            "message": "포즈 키포인트가 비어있습니다",
            "user_action": "키포인트를 추가하거나 해당 항목을 제거해주세요",
            "suggestions": [
                "포즈 라벨링 다시 수행",
                "해당 객체를 일반 박스로 변환"
            ]
        },

        # 출력 포맷 문제
        "no_label_rows": {
            "severity": "critical",
            "category": "output_format",
            "message": "YOLO 파일에 라벨이 없습니다",
            "user_action": "입력 데이터에 박스가 있는지 확인해주세요",
            "suggestions": [
                "원본 데이터 확인",
                "다른 포맷(vision_json)으로 먼저 변환하여 내용 검증"
            ]
        },
        "invalid_row_shape": {
            "severity": "high",
            "category": "output_format",
            "message": "YOLO 파일 형식이 잘못되었습니다 (각 행은 5개 값 필요)",
            "user_action": "YOLO 형식을 확인하고 수정해주세요",
            "suggestions": [
                "형식: <class_id> <x_center> <y_center> <width> <height>",
                "파일을 수동으로 검토하고 수정"
            ]
        },
        "no_objects": {
            "severity": "critical",
            "category": "output_format",
            "message": "Pascal VOC 파일에 객체가 없습니다",
            "user_action": "입력 데이터에 박스가 있는지 확인해주세요",
            "suggestions": [
                "원본 데이터 확인",
                "변환 과정에서 필터링되었는지 확인"
            ]
        },
        "no_annotations": {
            "severity": "critical",
            "category": "output_format",
            "message": "COCO 파일에 어노테이션이 없습니다",
            "user_action": "입력 데이터를 확인해주세요",
            "suggestions": [
                "원본 데이터에 라벨이 있는지 확인",
                "변환 임계값(threshold) 설정 확인"
            ]
        },
        "no_label_records": {
            "severity": "critical",
            "category": "output_format",
            "message": "Vision JSON 파일에 유효한 라벨이 없습니다",
            "user_action": "입력 데이터를 확인해주세요",
            "suggestions": [
                "원본 데이터 검증",
                "변환 로그에서 에러 메시지 확인"
            ]
        }
    }

    def categorize_issue(self, issue_str: str) -> Dict[str, str]:
        """이슈 문자열을 분석하여 카테고리화"""
        issue_key = issue_str.split(":", 1)[0].split("[")[0]

        if issue_key in self.ISSUE_ACTIONS:
            action_info = self.ISSUE_ACTIONS[issue_key].copy()
            action_info["original_issue"] = issue_str
            return action_info

        # 동적 이슈 처리
        if "missing_image:" in issue_str:
            info = self.ISSUE_ACTIONS["missing_image"].copy()
            info["original_issue"] = issue_str
            info["message"] = f"이미지 파일을 찾을 수 없습니다: {issue_str.split(':', 1)[1]}"
            return info

        if "image_open_failed:" in issue_str:
            info = self.ISSUE_ACTIONS["image_open_failed"].copy()
            info["original_issue"] = issue_str
            return info

        # 기본값
        return {
            "severity": "unknown",
            "category": "unknown",
            "message": issue_str,
            "user_action": "이슈를 수동으로 검토해주세요",
            "suggestions": ["로그 파일 확인", "지원팀 문의"],
            "original_issue": issue_str
        }

    def generate_action_report(
        self,
        image_name: str,
        issues: List[str],
        result_data: Optional[Dict] = None
    ) -> Dict:
        """이미지별 액션 리포트 생성"""
        categorized_issues = [self.categorize_issue(issue) for issue in issues]

        # 심각도별 분류
        critical = [i for i in categorized_issues if i["severity"] == "critical"]
        high = [i for i in categorized_issues if i["severity"] == "high"]
        medium = [i for i in categorized_issues if i["severity"] == "medium"]

        # 카테고리별 그룹화
        by_category = {}
        for issue in categorized_issues:
            cat = issue["category"]
            if cat not in by_category:
                by_category[cat] = []
            by_category[cat].append(issue)

        return {
            "image": image_name,
            "status": "blocked" if critical else ("needs_attention" if high else "warning"),
            "total_issues": len(issues),
            "issues_by_severity": {
                "critical": len(critical),
                "high": len(high),
                "medium": len(medium)
            },
            "issues_by_category": {cat: len(items) for cat, items in by_category.items()},
            "detailed_issues": categorized_issues,
            "priority_actions": [
                issue["user_action"] for issue in critical + high
            ][:3],  # 최대 3개 우선순위 액션
            "metadata": {
                "has_labels": result_data and result_data.get("objects", 0) > 0 if result_data else False
            }
        }

    def generate_summary_report(self, records: List[Dict]) -> Dict:
        """전체 요약 리포트 생성"""
        total = len(records)
        blocked = sum(1 for r in records if r.get("status") == "blocked")
        needs_attention = sum(1 for r in records if r.get("status") == "needs_attention")
        warning = sum(1 for r in records if r.get("status") == "warning")
        clean = total - blocked - needs_attention - warning

        # 가장 흔한 이슈 TOP 5
        all_issues = []
        for record in records:
            if "detailed_issues" in record:
                all_issues.extend(record["detailed_issues"])

        issue_freq = {}
        for issue in all_issues:
            key = issue["original_issue"].split(":", 1)[0]
            issue_freq[key] = issue_freq.get(key, 0) + 1

        top_issues = sorted(issue_freq.items(), key=lambda x: x[1], reverse=True)[:5]

        return {
            "summary": {
                "total_records": total,
                "clean": clean,
                "needs_review": blocked + needs_attention + warning,
                "blocked": blocked,
                "needs_attention": needs_attention,
                "warning": warning
            },
            "completion_rate": f"{(clean / total * 100):.1f}%" if total > 0 else "0%",
            "top_issues": [
                {
                    "issue_type": issue_type,
                    "count": count,
                    "percentage": f"{(count / total * 100):.1f}%",
                    "action": self.ISSUE_ACTIONS.get(issue_type, {}).get("user_action", "검토 필요")
                }
                for issue_type, count in top_issues
            ],
            "recommended_actions": self._generate_recommended_actions(records),
            "records": records
        }

    def _generate_recommended_actions(self, records: List[Dict]) -> List[str]:
        """전체 데이터셋에 대한 권장 조치사항"""
        actions = []

        # 블로킹 이슈가 많으면
        blocked_count = sum(1 for r in records if r.get("status") == "blocked")
        if blocked_count > len(records) * 0.3:
            actions.append(f"⚠️ {blocked_count}개 파일이 블로킹 상태입니다. 우선적으로 처리가 필요합니다.")

        # 특정 이슈가 많으면
        all_categories = {}
        for record in records:
            if "issues_by_category" in record:
                for cat, count in record["issues_by_category"].items():
                    all_categories[cat] = all_categories.get(cat, 0) + count

        if all_categories.get("input_data", 0) > 0:
            actions.append(
                f"📁 입력 데이터 문제가 {all_categories['input_data']}건 발견되었습니다. "
                "원본 데이터를 확인해주세요."
            )

        if all_categories.get("label_quality", 0) > 0:
            actions.append(
                f"🏷️ 라벨 품질 문제가 {all_categories['label_quality']}건 발견되었습니다. "
                "라벨링 도구 설정을 확인해주세요."
            )

        if all_categories.get("file_system", 0) > 0:
            actions.append(
                f"📂 파일 시스템 문제가 {all_categories['file_system']}건 발견되었습니다. "
                "파일 경로와 권한을 확인해주세요."
            )

        if not actions:
            actions.append("✅ 대부분의 데이터가 정상적으로 처리되었습니다.")

        return actions


def generate_user_report(validation_records: List[Dict], export_records: List[Dict] = None) -> Dict:
    """
    사용자에게 전달할 최종 리포트 생성

    Args:
        validation_records: 입력 검증 결과 [{"image": str, "issues": [str], ...}]
        export_records: 출력 검증 결과 [{"image": str, "issues": [str], "paths": {...}}]

    Returns:
        사용자 액션이 포함된 리포트
    """
    reporter = IssueActionReporter()

    # 입력 검증 리포트
    input_reports = []
    for record in validation_records:
        if record.get("issues"):
            input_reports.append(
                reporter.generate_action_report(
                    record["image"],
                    record["issues"],
                    record.get("result")
                )
            )

    # 출력 검증 리포트
    output_reports = []
    if export_records:
        for record in export_records:
            if record.get("issues"):
                output_reports.append(
                    reporter.generate_action_report(
                        record["image"],
                        record["issues"]
                    )
                )

    # 통합 리포트
    all_reports = input_reports + output_reports

    if not all_reports:
        return {
            "status": "success",
            "message": "모든 데이터가 성공적으로 처리되었습니다",
            "summary": {
                "total_records": len(validation_records),
                "clean": len(validation_records),
                "needs_review": 0
            }
        }

    summary = reporter.generate_summary_report(all_reports)

    return {
        "status": "partial_success" if summary["summary"]["clean"] > 0 else "needs_review",
        "message": f"{summary['summary']['needs_review']}개 파일에 문제가 발견되었습니다",
        "summary": summary["summary"],
        "completion_rate": summary["completion_rate"],
        "top_issues": summary["top_issues"],
        "recommended_actions": summary["recommended_actions"],
        "detailed_records": all_reports
    }
