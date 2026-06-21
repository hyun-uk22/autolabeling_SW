def build_generation_performance(
    images: int,
    total_elapsed_sec: float,
    low_api_attempts: int,
    high_api_attempts: int,
    escalation_count: int,
    manual_time_per_image_sec: float = 45.0,
) -> dict:
    estimated_manual = images * manual_time_per_image_sec
    return {
        "avg_elapsed_sec": total_elapsed_sec / images if images else 0.0,
        "manual_time_per_image_sec": manual_time_per_image_sec,
        "estimated_manual_time_sec": estimated_manual,
        "estimated_time_saved_pct": (
            (estimated_manual - total_elapsed_sec) / estimated_manual * 100.0
            if estimated_manual else 0.0
        ),
        "low_api_attempts": low_api_attempts,
        "high_api_attempts": high_api_attempts,
        "escalation_rate": escalation_count / images if images else 0.0,
        "estimated_high_model_avoidance_rate": (
            (images - escalation_count) / images if images else 0.0
        ),
        "estimation_notice": "수동 시간과 high-model 회피율은 설정된 기준에 따른 추정치이며 실제 비용이 아닙니다.",
    }
