from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class AnalysisConfig:
    max_frames: int | None = 300
    frame_stride: int = 1
    max_side: int | None = 960
    start_seconds: float = 0.0
    duration_seconds: float | None = None
    grid_size: int = 12
    grid_margin: float = 0.05
    smooth_window: int = 15
    high_pass_cutoff_hz: float = 1.0
    reference_jitter_ratio: float = 0.005
    min_background_points: int = 10
    min_foreground_points: int = 3
    chunk_frames: int | None = None
    chunk_overlap: int = 20
    video_source_resolution: bool = False
    device: str = "auto"
    checkpoint: Path | None = None
    cotracker_repo: Path | None = None

    def __post_init__(self) -> None:
        if self.frame_stride < 1:
            raise ValueError("frame_stride must be >= 1")
        if self.max_frames is not None and self.max_frames < 1:
            raise ValueError("max_frames must be >= 1 or None")
        if self.max_side is not None and self.max_side < 16:
            raise ValueError("max_side must be >= 16 or None")
        if self.start_seconds < 0.0:
            raise ValueError("start_seconds must be >= 0")
        if self.duration_seconds is not None and self.duration_seconds <= 0.0:
            raise ValueError("duration_seconds must be > 0 or None")
        if self.chunk_frames is not None:
            if self.chunk_frames < 2:
                raise ValueError("chunk_frames must be >= 2 or None")
            if self.chunk_overlap < 1:
                raise ValueError("chunk_overlap must be >= 1 when chunking")
            if self.chunk_overlap >= self.chunk_frames:
                raise ValueError("chunk_overlap must be smaller than chunk_frames")
        if self.grid_size < 2:
            raise ValueError("grid_size must be >= 2")
        if not 0.0 <= self.grid_margin < 0.5:
            raise ValueError("grid_margin must be in [0, 0.5)")
        if self.smooth_window < 1:
            raise ValueError("smooth_window must be >= 1")
        if self.high_pass_cutoff_hz <= 0.0:
            raise ValueError("high_pass_cutoff_hz must be > 0")
        if not 0.0 < self.reference_jitter_ratio <= 1.0:
            raise ValueError("reference_jitter_ratio must be in (0, 1]")
        if self.min_background_points < 1:
            raise ValueError("min_background_points must be >= 1")
        if self.min_foreground_points < 1:
            raise ValueError("min_foreground_points must be >= 1")

    def to_dict(self) -> dict[str, object]:
        return {
            "max_frames": self.max_frames,
            "frame_stride": self.frame_stride,
            "max_side": self.max_side,
            "start_seconds": self.start_seconds,
            "duration_seconds": self.duration_seconds,
            "grid_size": self.grid_size,
            "grid_margin": self.grid_margin,
            "smooth_window": self.smooth_window,
            "high_pass_cutoff_hz": self.high_pass_cutoff_hz,
            "reference_jitter_ratio": self.reference_jitter_ratio,
            "min_background_points": self.min_background_points,
            "min_foreground_points": self.min_foreground_points,
            "chunk_frames": self.chunk_frames,
            "chunk_overlap": self.chunk_overlap,
            "video_source_resolution": self.video_source_resolution,
            "device": self.device,
            "checkpoint": str(self.checkpoint) if self.checkpoint else None,
            "cotracker_repo": (
                str(self.cotracker_repo) if self.cotracker_repo else None
            ),
        }
