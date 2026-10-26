from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


class VideoReadError(RuntimeError):
    pass


@dataclass(frozen=True)
class VideoMetadata:
    path: Path
    source_width: int
    source_height: int
    width: int
    height: int
    fps: float
    source_total_frames: int
    sampled_frames: int
    frame_stride: int
    start_seconds: float = 0.0
    sampled_fps: float = 0.0
    variable_frame_rate: bool = False
    orientation_metadata_degrees: float = 0.0

    @property
    def duration_seconds(self) -> float | None:
        if self.fps <= 0 or self.source_total_frames <= 0:
            return None
        return self.source_total_frames / self.fps

    @property
    def analyzed_duration_seconds(self) -> float | None:
        if self.fps <= 0:
            return None
        return self.sampled_frames * self.frame_stride / self.fps

    @property
    def frame_diagonal(self) -> float:
        return float(np.hypot(self.width, self.height))

    def to_dict(self) -> dict[str, object]:
        return {
            "path": str(self.path),
            "source_width": self.source_width,
            "source_height": self.source_height,
            "analysis_width": self.width,
            "analysis_height": self.height,
            "fps": self.fps,
            "source_total_frames": self.source_total_frames,
            "sampled_frames": self.sampled_frames,
            "frame_stride": self.frame_stride,
            "start_seconds": self.start_seconds,
            "sampled_fps": self.sampled_fps,
            "variable_frame_rate": self.variable_frame_rate,
            "orientation_metadata_degrees": self.orientation_metadata_degrees,
            "duration_seconds": self.duration_seconds,
            "analyzed_duration_seconds": self.analyzed_duration_seconds,
        }


def _target_size(width: int, height: int, max_side: int | None) -> tuple[int, int]:
    if max_side is None:
        return width, height
    longest = max(width, height)
    if longest <= max_side:
        return width, height
    scale = max_side / longest
    return max(2, int(round(width * scale))), max(2, int(round(height * scale)))


def read_video(
    path: Path | str,
    max_frames: int | None = None,
    frame_stride: int = 1,
    max_side: int | None = 960,
    start_seconds: float = 0.0,
    duration_seconds: float | None = None,
) -> tuple[np.ndarray, VideoMetadata]:
    if frame_stride < 1:
        raise ValueError("frame_stride must be >= 1")
    if start_seconds < 0.0:
        raise ValueError("start_seconds must be >= 0")
    if duration_seconds is not None and duration_seconds <= 0.0:
        raise ValueError("duration_seconds must be > 0 or None")
    video_path = Path(path)
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise VideoReadError(f"could not open video: {video_path}")

    reported_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    reported_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    reported_total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    try:
        orientation = float(capture.get(cv2.CAP_PROP_ORIENTATION_META))
    except (AttributeError, ValueError, cv2.error):
        orientation = 0.0
    if not np.isfinite(orientation) or orientation < 0.0:
        orientation = 0.0

    if start_seconds > 0.0 and fps > 0.0:
        capture.set(cv2.CAP_PROP_POS_FRAMES, int(round(start_seconds * fps)))

    frames: list[np.ndarray] = []
    sampled_times: list[float] = []
    source_width = reported_width
    source_height = reported_height
    target_size: tuple[int, int] | None = None
    frame_index = 0
    frame_tolerance = 0.5 / fps if fps > 0.0 else 0.0

    try:
        if start_seconds > 0.0:
            guard = 0
            max_guard = int(start_seconds * max(fps, 1.0)) + int(max(fps, 1.0))
            while guard < max_guard:
                stamp = float(capture.get(cv2.CAP_PROP_POS_MSEC) or 0.0) / 1000.0
                position = float(capture.get(cv2.CAP_PROP_POS_FRAMES) or 0.0)
                current = (
                    stamp
                    if stamp > 0.0
                    else (position / fps if fps > 0.0 else start_seconds)
                )
                if current >= start_seconds - frame_tolerance:
                    break
                ok, _ = capture.read()
                if not ok:
                    break
                guard += 1

        while max_frames is None or len(frames) < max_frames:
            if duration_seconds is not None and fps > 0.0:
                if frame_index / fps >= duration_seconds:
                    break
            stamp = float(capture.get(cv2.CAP_PROP_POS_MSEC) or 0.0) / 1000.0
            ok, frame = capture.read()
            if not ok:
                break
            if target_size is None:
                source_height, source_width = frame.shape[:2]
                target_size = _target_size(source_width, source_height, max_side)
            if frame_index % frame_stride == 0:
                if (frame.shape[1], frame.shape[0]) != target_size:
                    frame = cv2.resize(frame, target_size, interpolation=cv2.INTER_AREA)
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                if stamp > 0.0:
                    sampled_times.append(stamp)
                elif fps > 0.0:
                    sampled_times.append(start_seconds + frame_index / fps)
            frame_index += 1
    finally:
        capture.release()

    if not frames:
        raise VideoReadError(f"video contains no decodable frames: {video_path}")

    nominal_sampled_fps = fps / frame_stride if fps > 0.0 else 0.0
    measured_sampled_fps = 0.0
    variable_frame_rate = False
    if len(sampled_times) >= 3:
        deltas = np.diff(np.asarray(sampled_times, dtype=np.float64))
        deltas = deltas[deltas > 0.0]
        if len(deltas) >= 2:
            median_delta = float(np.median(deltas))
            if median_delta > 0.0:
                measured_sampled_fps = 1.0 / median_delta
                if nominal_sampled_fps > 0.0:
                    deviation = abs(measured_sampled_fps - nominal_sampled_fps)
                    variable_frame_rate = deviation / nominal_sampled_fps > 0.05
    sampled_fps = (
        measured_sampled_fps if measured_sampled_fps > 0.0 else nominal_sampled_fps
    )
    sampled_fps = round(sampled_fps, 6)

    height, width = frames[0].shape[:2]
    total_frames = reported_total if reported_total > 0 else frame_index
    metadata = VideoMetadata(
        path=video_path,
        source_width=source_width,
        source_height=source_height,
        width=width,
        height=height,
        fps=fps,
        source_total_frames=total_frames,
        sampled_frames=len(frames),
        frame_stride=frame_stride,
        start_seconds=start_seconds,
        sampled_fps=sampled_fps,
        variable_frame_rate=variable_frame_rate,
        orientation_metadata_degrees=orientation,
    )
    return np.stack(frames, axis=0), metadata
