from pathlib import Path

import cv2
import numpy as np
import pytest

from video_stabilization_analyzer.video import VideoReadError, read_video


def _write_video(path: Path, frames: int, width: int = 64, height: int = 48) -> None:
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"MJPG"),
        10.0,
        (width, height),
    )
    if not writer.isOpened():
        pytest.skip("MJPG video writer is unavailable")
    try:
        for index in range(frames):
            frame = np.full((height, width, 3), (index * 10) % 256, dtype=np.uint8)
            writer.write(frame)
    finally:
        writer.release()


def test_read_video_samples_and_resizes(tmp_path: Path) -> None:
    path = tmp_path / "input.avi"
    _write_video(path, frames=12)

    frames, metadata = read_video(
        path,
        max_frames=4,
        frame_stride=2,
        max_side=32,
    )

    assert frames.shape == (4, 24, 32, 3)
    assert frames.dtype == np.uint8
    assert metadata.sampled_frames == 4
    assert metadata.frame_stride == 2
    assert metadata.source_width == 64
    assert metadata.source_height == 48
    assert metadata.width == 32
    assert metadata.height == 24


def test_read_video_rejects_missing_file(tmp_path: Path) -> None:
    with pytest.raises(VideoReadError):
        read_video(tmp_path / "missing.mp4")


def test_read_video_rejects_bad_stride() -> None:
    with pytest.raises(ValueError):
        read_video("unused.mp4", frame_stride=0)


def test_read_video_reports_sampled_fps(tmp_path: Path) -> None:
    path = tmp_path / "input.avi"
    _write_video(path, frames=20)

    frames, metadata = read_video(path, max_frames=6, frame_stride=2, max_side=None)

    assert frames.shape[0] == 6
    assert metadata.sampled_fps == pytest.approx(5.0, rel=0.1)
    assert metadata.variable_frame_rate is False
    assert metadata.orientation_metadata_degrees == 0.0


def test_read_video_honors_start_and_duration(tmp_path: Path) -> None:
    path = tmp_path / "input.avi"
    _write_video(path, frames=30)

    frames, metadata = read_video(
        path,
        max_frames=None,
        max_side=None,
        start_seconds=1.0,
        duration_seconds=0.5,
    )

    assert metadata.start_seconds == 1.0
    assert 4 <= frames.shape[0] <= 6


def test_read_video_rejects_bad_window() -> None:
    with pytest.raises(ValueError):
        read_video("unused.mp4", start_seconds=-1.0)
    with pytest.raises(ValueError):
        read_video("unused.mp4", duration_seconds=0.0)
