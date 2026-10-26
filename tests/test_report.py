from video_stabilization_analyzer.report import build_report


def test_report_contains_quality_and_camera_model(synthetic_result) -> None:
    report = build_report(synthetic_result)

    assert report["schema_version"] == 2
    for key in (
        "video",
        "config",
        "tracking",
        "background_separation",
        "score",
        "metrics",
        "quality",
        "warnings",
    ):
        assert key in report

    assert report["score"]["camera_model"].startswith("similarity")
    assert report["quality"]["confidence"] in {"low", "medium", "high"}
    assert report["quality"]["analyzed_frames"] == 60
    assert report["metrics"]["rotation_jitter_deg"] >= 0.0
    assert report["metrics"]["scale_jitter_px"] >= 0.0
    assert report["score"]["high_pass_cutoff_hz"] == 1.0
