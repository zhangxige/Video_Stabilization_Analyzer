from pathlib import Path

import cv2
import numpy as np
import pytest

from video_stabilization_analyzer.analysis import evaluate_frames
from video_stabilization_analyzer.config import AnalysisConfig
from video_stabilization_analyzer.tracking.base import TrackResult
from video_stabilization_analyzer.video import read_video
from video_stabilization_analyzer.visualization import (
    annotate_frame,
    render_tracking_video,
)


class FakeTracker:
    name = "fake"

    def __init__(self, tracks: np.ndarray) -> None:
        self.tracks = tracks

    def track(self, frames: np.ndarray) -> TrackResult:
        return TrackResult(
            tracks=self.tracks,
            visibility=np.ones(self.tracks.shape[:2], dtype=bool),
            backend=self.name,
        )


def test_annotate_frame_draws_colored_points() -> None:
    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    tracks = np.zeros((3, 4, 2), dtype=np.float32)
    tracks[:, 0] = (250.0, 200.0)
    tracks[:, 1] = (280.0, 210.0)
    tracks[:, 2] = (300.0, 220.0)
    tracks[:, 3] = (260.0, 230.0)
    visibility = np.ones((3, 4), dtype=bool)
    background = np.array([True, True, False, False])

    annotated = annotate_frame(
        frame,
        tracks,
        visibility,
        background,
        frame_index=1,
        score=50.0,
        rms_jitter=2.0,
    )

    assert annotated.shape == frame.shape
    assert annotated.dtype == np.uint8
    assert not np.array_equal(annotated, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    background_pixel = annotated[200, 250]
    foreground_pixel = annotated[220, 300]
    assert background_pixel[1] > background_pixel[2]
    assert foreground_pixel[2] > foreground_pixel[1]


def test_render_tracking_video_writes_sampled_frames(tmp_path: Path) -> None:
    input_path = tmp_path / "input.avi"
    writer = cv2.VideoWriter(
        str(input_path),
        cv2.VideoWriter_fourcc(*"MJPG"),
        10.0,
        (160, 120),
    )
    if not writer.isOpened():
        pytest.skip("MJPG video writer is unavailable")
    try:
        for index in range(8):
            writer.write(np.full((120, 160, 3), index * 20, dtype=np.uint8))
    finally:
        writer.release()

    frames, metadata = read_video(input_path, max_side=None)
    tracks = np.zeros((len(frames), 9, 2), dtype=np.float32)
    for point in range(9):
        tracks[:, point, 0] = 30.0 + 10.0 * point
        tracks[:, point, 1] = 40.0 + 5.0 * point
    config = AnalysisConfig(max_frames=None, max_side=None)
    result = evaluate_frames(frames, FakeTracker(tracks), metadata, config)

    output_path = render_tracking_video(result, tmp_path / "tracked.avi")

    assert output_path.is_file()
    assert output_path.stat().st_size > 0
    capture = cv2.VideoCapture(str(output_path))
    try:
        assert int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) == len(frames)
        assert int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)) == 160
        assert int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)) == 120
    finally:
        capture.release()
