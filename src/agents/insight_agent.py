from collections import Counter
from typing import Iterable, Optional

from ..core.models import DetectionResult


class DatasetInsightAgent:
    """Analyze final labels and recommend dataset balancing actions."""

    def __init__(self, imbalance_ratio_threshold: float = 3.0):
        if imbalance_ratio_threshold <= 1.0:
            raise ValueError("imbalance_ratio_threshold must be greater than 1.0")
        self.imbalance_ratio_threshold = float(imbalance_ratio_threshold)
        self.all_labels = []

    @staticmethod
    def _result_labels(result: DetectionResult) -> list[str]:
        labels = []
        labels.extend(item.label for item in result.classifications)
        labels.extend(item.label for item in result.boxes)
        labels.extend(item.label for item in result.segments)
        labels.extend(item.label for item in result.poses)
        labels.extend("text" for _item in result.texts)
        labels.extend(item.label for item in result.tracks)
        return labels

    def add_result(self, result: DetectionResult) -> None:
        self.all_labels.extend(self._result_labels(result))

    def add_results(self, results: Iterable[DetectionResult]) -> None:
        for result in results:
            self.add_result(result)

    def reset(self) -> None:
        self.all_labels.clear()

    def analyze(self, results: Optional[Iterable[DetectionResult]] = None) -> dict:
        labels = list(self.all_labels)
        if results is not None:
            labels = [
                label
                for result in results
                for label in self._result_labels(result)
            ]
        counts = Counter(labels)
        total = sum(counts.values())
        distribution = {
            label: {
                "count": count,
                "percentage": (count / total * 100.0) if total else 0.0,
            }
            for label, count in counts.most_common()
        }

        if not counts:
            status = "empty"
            imbalance = None
            suggestions = ["분석할 유효 라벨이 없습니다. 입력 데이터와 라벨 생성·변환 결과를 확인하세요."]
        elif len(counts) == 1:
            only_class = next(iter(counts))
            status = "single_class"
            imbalance = None
            suggestions = [
                f"'{only_class}' 단일 클래스만 존재합니다. 의도된 데이터셋인지 확인하세요.",
                "다중 클래스 태스크라면 누락 클래스를 수집하거나 라벨 taxonomy를 검토하세요.",
            ]
        else:
            majority, majority_count = max(counts.items(), key=lambda item: item[1])
            minority, minority_count = min(counts.items(), key=lambda item: item[1])
            ratio = majority_count / minority_count if minority_count else float("inf")
            detected = ratio > self.imbalance_ratio_threshold
            status = "imbalanced" if detected else "balanced"
            imbalance = {
                "detected": detected,
                "ratio": ratio,
                "threshold": self.imbalance_ratio_threshold,
                "majority_class": majority,
                "majority_count": majority_count,
                "minority_class": minority,
                "minority_count": minority_count,
            }
            suggestions = []
            if detected:
                suggestions.extend([
                    f"희소 클래스 '{minority}' 데이터를 우선적으로 추가 수집하세요.",
                    f"학습 sampler에서 '{minority}' 클래스 oversampling을 검토하세요.",
                    f"'{minority}' 클래스에 copy-paste, crop, 색상·기하 augmentation 적용을 검토하세요.",
                ])

        return {
            "agent": "DatasetInsightAgent",
            "status": status,
            "total_labels": total,
            "class_count": len(counts),
            "distribution": distribution,
            "imbalance": imbalance,
            "suggestions": suggestions,
        }

    def get_report(self) -> str:
        insight = self.analyze()
        if insight["status"] == "empty":
            return "No data processed yet."
        lines = ["\n--- Dataset Insight Report ---"]
        for label, values in insight["distribution"].items():
            lines.append(f"- {label}: {values['count']} ({values['percentage']:.1f}%)")
        for suggestion in insight["suggestions"]:
            lines.append(f"- Suggest: {suggestion}")
        return "\n".join(lines)
