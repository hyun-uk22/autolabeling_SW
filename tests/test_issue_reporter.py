"""
Issue Reporter 테스트
"""
import json
import pytest

from src.agents.issue_reporter import IssueActionReporter, generate_user_report


class TestIssueActionReporter:
    def setup_method(self):
        self.reporter = IssueActionReporter()

    def test_categorize_empty_result(self):
        result = self.reporter.categorize_issue("empty_result")
        assert result["severity"] == "critical"
        assert result["category"] == "input_data"
        assert "비어있습니다" in result["message"]
        assert len(result["suggestions"]) > 0

    def test_categorize_coordinate_out_of_range(self):
        result = self.reporter.categorize_issue("coordinate_out_of_range")
        assert result["severity"] == "high"
        assert result["category"] == "label_quality"
        assert "정규화" in result["user_action"]

    def test_categorize_missing_image_with_path(self):
        result = self.reporter.categorize_issue("missing_image:/path/to/image.jpg")
        assert result["severity"] == "critical"
        assert result["category"] == "file_system"
        assert "/path/to/image.jpg" in result["message"]

    def test_categorize_unknown_issue(self):
        result = self.reporter.categorize_issue("unknown_issue_xyz")
        assert result["severity"] == "unknown"
        assert result["category"] == "unknown"
        assert "수동으로 검토" in result["user_action"]

    def test_generate_action_report_critical(self):
        issues = ["empty_result", "missing_image:/test.jpg"]
        report = self.reporter.generate_action_report("test.jpg", issues)

        assert report["image"] == "test.jpg"
        assert report["status"] == "blocked"
        assert report["total_issues"] == 2
        assert report["issues_by_severity"]["critical"] == 2
        assert len(report["priority_actions"]) > 0

    def test_generate_action_report_high(self):
        issues = ["coordinate_out_of_range", "missing_label"]
        report = self.reporter.generate_action_report("test.jpg", issues)

        assert report["status"] == "needs_attention"
        assert report["issues_by_severity"]["high"] == 2
        assert report["issues_by_severity"]["critical"] == 0

    def test_generate_action_report_medium(self):
        issues = ["confidence_out_of_range"]
        report = self.reporter.generate_action_report("test.jpg", issues)

        assert report["status"] == "warning"
        assert report["issues_by_severity"]["medium"] == 1

    def test_generate_summary_report_clean(self):
        records = [
            self.reporter.generate_action_report("img1.jpg", []),
            self.reporter.generate_action_report("img2.jpg", []),
        ]
        # 빈 issues는 리포트가 생성되지 않으므로 직접 생성
        summary = self.reporter.generate_summary_report([])

        assert summary["summary"]["total_records"] == 0
        assert summary["summary"]["clean"] == 0

    def test_generate_summary_report_with_issues(self):
        records = [
            self.reporter.generate_action_report("img1.jpg", ["empty_result"]),
            self.reporter.generate_action_report("img2.jpg", ["coordinate_out_of_range"]),
            self.reporter.generate_action_report("img3.jpg", ["confidence_out_of_range"]),
        ]

        summary = self.reporter.generate_summary_report(records)

        assert summary["summary"]["total_records"] == 3
        assert summary["summary"]["blocked"] == 1  # empty_result
        assert summary["summary"]["needs_attention"] == 1  # coordinate_out_of_range
        assert summary["summary"]["warning"] == 1  # confidence_out_of_range
        assert len(summary["top_issues"]) > 0
        assert len(summary["recommended_actions"]) > 0

    def test_recommended_actions_many_blocked(self):
        # 30% 이상 블로킹되면 경고
        records = []
        for i in range(10):
            if i < 4:
                issues = ["empty_result"]
            else:
                issues = ["confidence_out_of_range"]
            records.append(self.reporter.generate_action_report(f"img{i}.jpg", issues))

        summary = self.reporter.generate_summary_report(records)

        assert any("블로킹 상태" in action for action in summary["recommended_actions"])

    def test_recommended_actions_label_quality(self):
        records = [
            self.reporter.generate_action_report(f"img{i}.jpg", ["coordinate_out_of_range"])
            for i in range(5)
        ]

        summary = self.reporter.generate_summary_report(records)

        assert any("라벨 품질" in action for action in summary["recommended_actions"])


class TestGenerateUserReport:
    def test_generate_user_report_success(self):
        validation_records = [
            {"image": "img1.jpg", "issues": []},
            {"image": "img2.jpg", "issues": []},
        ]

        report = generate_user_report(validation_records)

        assert report["status"] == "success"
        assert report["summary"]["clean"] == 2
        assert report["summary"]["needs_review"] == 0

    def test_generate_user_report_partial_success(self):
        validation_records = [
            {"image": "img1.jpg", "issues": []},
            {"image": "img2.jpg", "issues": ["coordinate_out_of_range"]},
        ]

        report = generate_user_report(validation_records)

        assert report["status"] == "partial_success"
        assert report["summary"]["needs_review"] == 1
        assert len(report["detailed_records"]) == 1
        assert len(report["top_issues"]) > 0

    def test_generate_user_report_needs_review(self):
        validation_records = [
            {"image": "img1.jpg", "issues": ["empty_result"]},
            {"image": "img2.jpg", "issues": ["missing_image:/test.jpg"]},
        ]

        report = generate_user_report(validation_records)

        assert report["status"] == "needs_review"
        assert report["summary"]["clean"] == 0
        assert report["summary"]["needs_review"] == 2

    def test_generate_user_report_with_export_records(self):
        validation_records = [
            {"image": "img1.jpg", "issues": []},
        ]
        export_records = [
            {"image": "img1.jpg", "issues": ["no_label_rows"]},
        ]

        report = generate_user_report(validation_records, export_records)

        assert report["status"] == "partial_success"
        assert len(report["detailed_records"]) == 1
        assert report["detailed_records"][0]["issues_by_category"]["output_format"] == 1

    def test_generate_user_report_json_serializable(self):
        validation_records = [
            {"image": "img1.jpg", "issues": ["empty_result"]},
        ]

        report = generate_user_report(validation_records)

        # JSON 직렬화 가능한지 확인
        try:
            json.dumps(report, ensure_ascii=False)
        except (TypeError, ValueError) as e:
            pytest.fail(f"Report is not JSON serializable: {e}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
