from typing import List
from ..core.models import DetectionResult
from collections import Counter

class DatasetInsightAgent:
    """
    Analyzes class distribution and suggests augmentation strategies.
    """
    def __init__(self):
        self.all_labels = []

    def add_result(self, result: DetectionResult):
        for item in result.classifications:
            self.all_labels.append(item.label)
        for box in result.boxes:
            self.all_labels.append(box.label)
        for segment in result.segments:
            self.all_labels.append(segment.label)
        for pose in result.poses:
            self.all_labels.append(pose.label)
        for text in result.texts:
            self.all_labels.append("text")
        for track in result.tracks:
            self.all_labels.append(track.label)

    def get_report(self) -> str:
        counts = Counter(self.all_labels)
        total = sum(counts.values())
        
        report = ["\n--- Dataset Insight Report ---"]
        if not counts:
            return "No data processed yet."

        for label, count in counts.items():
            percentage = (count / total) * 100
            report.append(f"- {label}: {count} ({percentage:.1f}%)")
        
        # Simple imbalance detection
        if len(counts) > 1:
            max_label = max(counts, key=counts.get)
            min_label = min(counts, key=counts.get)
            ratio = counts[max_label] / counts[min_label]
            if ratio > 3:
                report.append(f"\n[!] Imbalance detected: '{min_label}' is under-represented compared to '{max_label}'.")
                report.append("Suggest: Targeted data collection or augmentation for rare classes.")
        
        return "\n".join(report)
