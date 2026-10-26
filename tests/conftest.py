from pathlib import Path

import numpy as np
import pytest

from video_stabilization_analyzer.analysis import evaluate_frames
from video_stabilization_analyzer.config import AnalysisConfig
from video_stabilization_analyzer.tracking.base import TrackResult
from video_stabilization_analyzer.video import VideoMetadata


class StaticTracker:
    name = "fake"

    def __init__(self, tracks: np.ndarray) -> None:
        self.tracks = tracks

    def track(self, frames: np.ndarray) -> TrackResult:
        return TrackResult(
            tracks=self.tracks,
            visibility=np.ones(self.tracks.shape[:2], dtype=bool),
            backend=self.name,
        )


@pytest.fixture
def synthetic_result():
    frames, points = 60, 64
    tracks = np.zeros((frames, points, 2), dtype=np.float32)
    tracks[:, :, 0] = 160.0
    tracks[:, :, 1] = 120.0
    metadata = VideoMetadata(
        path=Path("synthetic.mp4"),
        source_width=320,
        source_height=240,
        width=320,
        height=240,
        fps=30.0,
        source_total_frames=frames,
        sampled_frames=frames,
        frame_stride=1,
    )
    return evaluate_frames(
        np.zeros((frames, 240, 320, 3), dtype=np.uint8),
        StaticTracker(tracks),
        metadata,
        AnalysisConfig(max_frames=200, max_side=None),
    )
