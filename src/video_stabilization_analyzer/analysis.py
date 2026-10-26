from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .config import AnalysisConfig
from .metrics import (
    CameraEstimate,
    JitterSeries,
    StabilityMetrics,
    classify_background,
    compute_metric_series,
)
from .tracking.base import TrackResult, Tracker, TrackingError
from .tracking.cotracker import CoTrackerTracker
from .video import VideoMetadata, read_video


@dataclass(frozen=True)
class AnalysisResult:
    metadata: VideoMetadata
    config: AnalysisConfig
    track_result: TrackResult
    background_mask: np.ndarray
    metrics: StabilityMetrics
    camera: CameraEstimate
    global_trajectory: np.ndarray
    smoothed_trajectory: np.ndarray
    jitter_series: JitterSeries
    warnings: tuple[str, ...]


def evaluate_frames(
    frames: np.ndarray,
    tracker: Tracker,
    metadata: VideoMetadata,
    config: AnalysisConfig,
) -> AnalysisResult:
    frames = np.asarray(frames)
    if frames.ndim != 4 or frames.shape[-1] != 3:
        raise ValueError("frames must have shape (T, H, W, 3)")
    if frames.shape[0] != metadata.sampled_frames:
        raise ValueError("frame count does not match metadata.sampled_frames")
    if frames.shape[1] != metadata.height or frames.shape[2] != metadata.width:
        raise ValueError("frame dimensions do not match metadata")
    if frames.shape[0] < 2:
        raise ValueError("at least two frames are required")

    track_result = tracker.track(frames)
    if track_result.frames != frames.shape[0]:
        raise TrackingError("tracker returned a different number of frames")

    sampled_fps = metadata.sampled_fps
    if sampled_fps <= 0.0 and metadata.fps > 0:
        sampled_fps = metadata.fps / metadata.frame_stride
    warnings: list[str] = []
    if frames.shape[0] < 5:
        warnings.append("very few frames analyzed; the score has low confidence")
    if frames.shape[0] < config.smooth_window:
        warnings.append("smooth_window was reduced to fit the video length")
    if metadata.fps <= 0:
        warnings.append("video FPS is unavailable")
    if metadata.variable_frame_rate:
        warnings.append(
            f"variable frame rate detected; using measured {sampled_fps:.2f} Hz "
            "sample rate for the high-pass cutoff"
        )
    if metadata.orientation_metadata_degrees:
        warnings.append(
            f"video carries {metadata.orientation_metadata_degrees:.0f} degree "
            "rotation metadata; verify the decoded orientation"
        )
    if metadata.fps > 0 and metadata.source_total_frames > 0:
        source_duration = metadata.source_total_frames / metadata.fps
        analyzed_span = (
            metadata.sampled_frames * metadata.frame_stride / metadata.fps
        )
        analyzed_end = metadata.start_seconds + analyzed_span
        if (
            config.duration_seconds is None
            and analyzed_end < source_duration - 1.0
        ):
            warnings.append(
                f"only {analyzed_span:.1f}s of {source_duration:.1f}s analyzed; "
                "raise --max-frames or set --duration-seconds"
            )
        elif (
            config.duration_seconds is not None
            and analyzed_span < config.duration_seconds - 1.0
        ):
            warnings.append(
                f"requested {config.duration_seconds:.1f}s but only "
                f"{analyzed_span:.1f}s analyzed; --max-frames or the end of the "
                "video limited the window"
            )
    if sampled_fps > 0.0 and config.high_pass_cutoff_hz >= 0.5 * sampled_fps:
        warnings.append(
            "high_pass_cutoff_hz is at or above Nyquist; the filter was clamped"
        )
    if track_result.points < config.min_background_points:
        warnings.append(
            "fewer tracked points than min_background_points; "
            "background separation may be unreliable"
        )
    if float(track_result.visibility.mean()) < 0.5:
        warnings.append(
            "less than half of tracked point observations are visible; "
            "the trajectory may be unreliable"
        )

    background_mask = classify_background(
        tracks=track_result.tracks,
        visibility=track_result.visibility,
        frame_diagonal=metadata.frame_diagonal,
        min_points=config.min_background_points,
        min_foreground_points=config.min_foreground_points,
    )
    if not background_mask.any():
        raise TrackingError("no background points could be selected")
    if background_mask.sum() < config.min_background_points:
        warnings.append(
            "fewer background points than requested; metrics may be noisy"
        )
    if background_mask.mean() < 0.2:
        warnings.append(
            "less than 20% of tracked points match the dominant camera motion; "
            "the subject may dominate the frame"
        )

    metrics, camera, smoothed, jitter_series = compute_metric_series(
        tracks=track_result.tracks,
        visibility=track_result.visibility,
        background_mask=background_mask,
        frame_diagonal=metadata.frame_diagonal,
        smooth_window=config.smooth_window,
        reference_jitter_ratio=config.reference_jitter_ratio,
        high_pass_cutoff_hz=config.high_pass_cutoff_hz,
        fps=sampled_fps,
        frame_center=(metadata.width / 2.0, metadata.height / 2.0),
    )

    if metrics.confidence == "low":
        warnings.append(
            "low-confidence result: few usable frames or points, low visibility, "
            "or inconsistent background motion"
        )

    return AnalysisResult(
        metadata=metadata,
        config=config,
        track_result=track_result,
        background_mask=background_mask,
        metrics=metrics,
        camera=camera,
        global_trajectory=camera.position,
        smoothed_trajectory=smoothed,
        jitter_series=jitter_series,
        warnings=tuple(warnings),
    )


def evaluate_video(
    path: Path | str,
    config: AnalysisConfig | None = None,
    tracker: Tracker | None = None,
) -> AnalysisResult:
    config = config or AnalysisConfig()
    frames, metadata = read_video(
        path=path,
        max_frames=config.max_frames,
        frame_stride=config.frame_stride,
        max_side=config.max_side,
        start_seconds=config.start_seconds,
        duration_seconds=config.duration_seconds,
    )
    if tracker is None:
        tracker = CoTrackerTracker(
            grid_size=config.grid_size,
            grid_margin=config.grid_margin,
            device=config.device,
            checkpoint=config.checkpoint,
            repo_dir=config.cotracker_repo,
            chunk_frames=config.chunk_frames,
            chunk_overlap=config.chunk_overlap,
        )
    return evaluate_frames(frames, tracker, metadata, config)
