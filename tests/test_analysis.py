import json
from pathlib import Path

import numpy as np
import pytest

from video_stabilization_analyzer.analysis import evaluate_frames
from video_stabilization_analyzer.config import AnalysisConfig
from video_stabilization_analyzer.report import build_report
from video_stabilization_analyzer.tracking.base import TrackResult, TrackingError
from video_stabilization_analyzer.video import VideoMetadata


class FakeTracker:
    name = "fake"

    def __init__(self, tracks: np.ndarray, visibility: np.ndarray | None = None) -> None:
        self.tracks = tracks
        self.visibility = (
            np.ones(tracks.shape[:2], dtype=bool)
            if visibility is None
            else visibility
        )

    def track(self, frames: np.ndarray) -> TrackResult:
        return TrackResult(
            tracks=self.tracks,
            visibility=self.visibility,
            backend=self.name,
        )


def make_metadata(frames: int, width: int = 320, height: int = 240) -> VideoMetadata:
    return VideoMetadata(
        path=Path("synthetic.mp4"),
        source_width=width,
        source_height=height,
        width=width,
        height=height,
        fps=30.0,
        source_total_frames=frames,
        sampled_frames=frames,
        frame_stride=1,
    )


def make_frames(frames: int, width: int = 320, height: int = 240) -> np.ndarray:
    return np.zeros((frames, height, width, 3), dtype=np.uint8)


def make_config(**overrides: object) -> AnalysisConfig:
    values = {
        "max_frames": 200,
        "max_side": None,
        "smooth_window": 15,
        "min_background_points": 10,
    }
    values.update(overrides)
    return AnalysisConfig(**values)


def test_static_video_scores_high() -> None:
    frames, points = 60, 64
    tracks = np.zeros((frames, points, 2), dtype=np.float32)
    tracks[:, :, 0] = 160.0
    tracks[:, :, 1] = 120.0
    metadata = make_metadata(frames)

    result = evaluate_frames(
        make_frames(frames),
        FakeTracker(tracks),
        metadata,
        make_config(),
    )

    assert result.metrics.stability_score > 99.9
    assert result.background_mask.sum() == points
    assert result.warnings == ()


def test_shaky_video_scores_lower() -> None:
    frames, points = 120, 64
    time = np.arange(frames, dtype=np.float32)
    tracks = np.zeros((frames, points, 2), dtype=np.float32)
    tracks[:, :, 0] = (5.0 * np.sin(2.0 * np.pi * time / 4.0))[:, None]
    tracks[:, :, 1] = (3.0 * np.cos(2.0 * np.pi * time / 4.0))[:, None]
    metadata = make_metadata(frames)

    result = evaluate_frames(
        make_frames(frames),
        FakeTracker(tracks),
        metadata,
        make_config(),
    )

    assert result.metrics.stability_score < 90.0
    assert result.metrics.rms_jitter_px > 3.0


def test_subject_motion_is_reported_separately() -> None:
    frames, points = 90, 100
    time = np.arange(frames, dtype=np.float32)[:, None]
    tracks = np.zeros((frames, points, 2), dtype=np.float32)
    tracks[:, :, 0] = 0.3 * time
    tracks[:, :, 1] = 120.0
    tracks[:, 80:, 0] += 8.0 * time
    metadata = make_metadata(frames)

    result = evaluate_frames(
        make_frames(frames),
        FakeTracker(tracks),
        metadata,
        make_config(),
    )

    assert result.metrics.background_points == 80
    assert result.metrics.foreground_points == 20
    assert result.metrics.subject_motion_px == pytest.approx(8.0, abs=1e-3)
    assert result.metrics.stability_score > 99.0


def test_short_video_emits_warning() -> None:
    frames, points = 4, 16
    tracks = np.zeros((frames, points, 2), dtype=np.float32)
    metadata = make_metadata(frames)

    result = evaluate_frames(
        make_frames(frames),
        FakeTracker(tracks),
        metadata,
        make_config(),
    )

    assert any("very few frames" in warning for warning in result.warnings)


def test_explicit_duration_does_not_warn_about_truncation() -> None:
    frames, points = 60, 16
    tracks = np.zeros((frames, points, 2), dtype=np.float32)
    metadata = VideoMetadata(
        path=Path("synthetic.mp4"),
        source_width=320,
        source_height=240,
        width=320,
        height=240,
        fps=30.0,
        source_total_frames=300,
        sampled_frames=frames,
        frame_stride=1,
        start_seconds=1.0,
    )

    result = evaluate_frames(
        make_frames(frames),
        FakeTracker(tracks),
        metadata,
        make_config(duration_seconds=2.0),
    )

    assert not any("only" in warning for warning in result.warnings)
    assert not any("requested" in warning for warning in result.warnings)


def test_truncated_analysis_warns_about_skipped_video() -> None:
    frames, points = 60, 16
    tracks = np.zeros((frames, points, 2), dtype=np.float32)
    metadata = VideoMetadata(
        path=Path("synthetic.mp4"),
        source_width=320,
        source_height=240,
        width=320,
        height=240,
        fps=30.0,
        source_total_frames=300,
        sampled_frames=frames,
        frame_stride=1,
    )

    result = evaluate_frames(
        make_frames(frames),
        FakeTracker(tracks),
        metadata,
        make_config(),
    )

    assert any("only 2.0s of 10.0s analyzed" in warning for warning in result.warnings)


def test_report_is_json_serializable() -> None:
    frames, points = 30, 16
    tracks = np.zeros((frames, points, 2), dtype=np.float32)
    metadata = make_metadata(frames)
    result = evaluate_frames(
        make_frames(frames),
        FakeTracker(tracks),
        metadata,
        make_config(),
    )

    report = build_report(result)
    encoded = json.dumps(report)
    assert '"stability_score"' in encoded
    assert report["tracking"]["backend"] == "fake"
    assert report["score"]["scope"] == "camera/background stability"
    assert report["score"]["subject_motion_included"] is False
    assert report["score"]["high_pass_cutoff_hz"] == 1.0


def test_frame_metadata_mismatch_raises() -> None:
    frames, points = 30, 16
    tracks = np.zeros((frames, points, 2), dtype=np.float32)
    metadata = make_metadata(frames, width=640, height=480)

    with pytest.raises(ValueError):
        evaluate_frames(
            make_frames(frames),
            FakeTracker(tracks),
            metadata,
            make_config(),
        )


def test_tracker_frame_count_mismatch_raises() -> None:
    frames, points = 30, 16
    tracks = np.zeros((frames - 1, points, 2), dtype=np.float32)
    metadata = make_metadata(frames)

    with pytest.raises(TrackingError):
        evaluate_frames(
            make_frames(frames),
            FakeTracker(tracks),
            metadata,
            make_config(),
        )
