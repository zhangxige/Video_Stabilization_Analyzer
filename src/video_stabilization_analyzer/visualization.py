from pathlib import Path

import cv2
import numpy as np

from .analysis import AnalysisResult
from .video import read_video

BACKGROUND_COLOR = (0, 220, 0)
FOREGROUND_COLOR = (0, 0, 255)
TEXT_COLOR = (255, 255, 255)
PANEL_COLOR = (0, 0, 0)
TRAIL_FRAMES = 8


class VisualizationError(RuntimeError):
    pass


def _open_writer(path: Path, fps: float, size: tuple[int, int]):
    suffix = path.suffix.lower()
    if suffix in {".mp4", ".mov", ".m4v"}:
        codecs = ("mp4v", "avc1")
    elif suffix == ".avi":
        codecs = ("MJPG",)
    else:
        codecs = ("mp4v", "MJPG")

    for codec in codecs:
        writer = cv2.VideoWriter(
            str(path),
            cv2.VideoWriter_fourcc(*codec),
            fps,
            size,
        )
        if writer.isOpened():
            return writer
        writer.release()
    raise VisualizationError(
        f"could not create video writer for {path}; try an .avi output path"
    )


def _draw_panel(
    frame: np.ndarray,
    frame_index: int,
    frames: int,
    score: float,
    rms_jitter: float,
    background_points: int,
    foreground_points: int,
) -> np.ndarray:
    height, width = frame.shape[:2]
    if width < 220 or height < 130:
        return frame

    font_scale = 0.5 if width < 900 else 0.6
    line_height = int(24 if width < 900 else 28)
    lines = [
        f"Stability {score:.1f}/100",
        f"High-pass jitter {rms_jitter:.2f} px",
        f"Background {background_points} | Foreground {foreground_points}",
        f"Frame {frame_index + 1}/{frames}",
    ]
    panel_width = min(width - 20, int(360 if width < 900 else 430))
    panel_height = line_height * len(lines) + 16
    panel = frame.copy()
    cv2.rectangle(panel, (10, 10), (10 + panel_width, 10 + panel_height), PANEL_COLOR, -1)
    frame = cv2.addWeighted(panel, 0.6, frame, 0.4, 0.0)

    for index, line in enumerate(lines):
        y = 10 + 8 + line_height * (index + 1) - 6
        cv2.putText(
            frame,
            line,
            (20, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            TEXT_COLOR,
            1,
            cv2.LINE_AA,
        )
    return frame


def annotate_frame(
    frame: np.ndarray,
    tracks: np.ndarray,
    visibility: np.ndarray,
    background_mask: np.ndarray,
    frame_index: int,
    score: float,
    rms_jitter: float,
    trail_frames: int = TRAIL_FRAMES,
) -> np.ndarray:
    frame_bgr = cv2.cvtColor(np.asarray(frame), cv2.COLOR_RGB2BGR)
    tracks = np.asarray(tracks, dtype=np.float64)
    visibility = np.asarray(visibility, dtype=bool)
    background_mask = np.asarray(background_mask, dtype=bool)
    frame_index = int(np.clip(frame_index, 0, tracks.shape[0] - 1))
    height, width = frame_bgr.shape[:2]

    trail_layer = frame_bgr.copy()
    for point in range(tracks.shape[1]):
        start = max(0, frame_index - trail_frames)
        trail = []
        for index in range(start, frame_index + 1):
            if not visibility[index, point]:
                continue
            x, y = tracks[index, point]
            if 0 <= x < width and 0 <= y < height:
                trail.append((int(round(x)), int(round(y))))
        if len(trail) >= 2:
            color = BACKGROUND_COLOR if background_mask[point] else FOREGROUND_COLOR
            cv2.polylines(
                trail_layer,
                [np.asarray(trail, dtype=np.int32)],
                False,
                color,
                1,
                cv2.LINE_AA,
            )
    annotated = cv2.addWeighted(trail_layer, 0.65, frame_bgr, 0.35, 0.0)

    for point in range(tracks.shape[1]):
        if not visibility[frame_index, point]:
            continue
        x, y = tracks[frame_index, point]
        color = BACKGROUND_COLOR if background_mask[point] else FOREGROUND_COLOR
        radius = 3 if width >= 500 else 2
        cv2.circle(
            annotated,
            (int(round(x)), int(round(y))),
            radius,
            color,
            -1,
            cv2.LINE_AA,
        )

    background_points = int(background_mask.sum())
    foreground_points = int((~background_mask).sum())
    annotated = _draw_panel(
        annotated,
        frame_index=frame_index,
        frames=tracks.shape[0],
        score=score,
        rms_jitter=rms_jitter,
        background_points=background_points,
        foreground_points=foreground_points,
    )
    return annotated


def render_tracking_video(
    result: AnalysisResult,
    output_path: Path | str,
) -> Path:
    path = Path(output_path)
    source = result.metadata.path.resolve()
    if path.resolve() == source:
        raise VisualizationError("output video must not overwrite the input video")
    path.parent.mkdir(parents=True, exist_ok=True)

    config = result.config
    frames, metadata = read_video(
        source,
        max_frames=config.max_frames,
        frame_stride=config.frame_stride,
        max_side=None if config.video_source_resolution else config.max_side,
        start_seconds=config.start_seconds,
        duration_seconds=config.duration_seconds,
    )
    if frames.shape[0] != result.track_result.frames:
        raise VisualizationError(
            "decoded frame count changed between analysis and visualization"
        )

    scale_x = frames.shape[2] / result.metadata.width
    scale_y = frames.shape[1] / result.metadata.height
    tracks = np.asarray(result.track_result.tracks, dtype=np.float64)
    if not (np.isclose(scale_x, 1.0) and np.isclose(scale_y, 1.0)):
        tracks = tracks * np.array((scale_x, scale_y), dtype=np.float64)

    sampled_fps = metadata.sampled_fps
    if sampled_fps <= 0.0:
        sampled_fps = (
            metadata.fps / metadata.frame_stride if metadata.fps > 0 else 30.0
        )
    height, width = frames.shape[1:3]
    writer = _open_writer(path, sampled_fps, (width, height))
    try:
        for frame_index, frame in enumerate(frames):
            annotated = annotate_frame(
                frame=frame,
                tracks=tracks,
                visibility=result.track_result.visibility,
                background_mask=result.background_mask,
                frame_index=frame_index,
                score=result.metrics.stability_score,
                rms_jitter=result.metrics.rms_jitter_px,
            )
            writer.write(annotated)
    finally:
        writer.release()
    return path
