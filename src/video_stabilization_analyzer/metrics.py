from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class StabilityMetrics:
    stability_score: float
    rms_jitter_px: float
    p95_jitter_px: float
    max_jitter_px: float
    rms_jitter_ratio: float
    rotation_jitter_deg: float
    rotation_jitter_px: float
    scale_jitter_ratio: float
    scale_jitter_px: float
    camera_motion_px: float
    camera_motion_ratio: float
    subject_motion_px: float
    subject_motion_ratio: float
    background_points: int
    foreground_points: int
    background_fraction: float
    visibility_rate: float
    spatial_residual_px: float
    analyzed_frames: int
    valid_jitter_frames: int
    valid_camera_frames: int
    confidence: str

    def to_dict(self) -> dict[str, float | int | str]:
        return {
            "stability_score": self.stability_score,
            "rms_jitter_px": self.rms_jitter_px,
            "p95_jitter_px": self.p95_jitter_px,
            "max_jitter_px": self.max_jitter_px,
            "rms_jitter_ratio": self.rms_jitter_ratio,
            "rotation_jitter_deg": self.rotation_jitter_deg,
            "rotation_jitter_px": self.rotation_jitter_px,
            "scale_jitter_ratio": self.scale_jitter_ratio,
            "scale_jitter_px": self.scale_jitter_px,
            "camera_motion_px": self.camera_motion_px,
            "camera_motion_ratio": self.camera_motion_ratio,
            "subject_motion_px": self.subject_motion_px,
            "subject_motion_ratio": self.subject_motion_ratio,
            "background_points": self.background_points,
            "foreground_points": self.foreground_points,
            "background_fraction": self.background_fraction,
            "visibility_rate": self.visibility_rate,
            "spatial_residual_px": self.spatial_residual_px,
            "analyzed_frames": self.analyzed_frames,
            "valid_jitter_frames": self.valid_jitter_frames,
            "valid_camera_frames": self.valid_camera_frames,
            "confidence": self.confidence,
        }


@dataclass(frozen=True)
class CameraEstimate:
    """Accumulated camera motion in the analysis coordinate system."""

    position: np.ndarray
    rotation: np.ndarray
    scale: np.ndarray
    valid: np.ndarray
    residual_px: float


@dataclass(frozen=True)
class JitterSeries:
    translation: np.ndarray
    rotation_px: np.ndarray
    scale_px: np.ndarray
    magnitude: np.ndarray
    lever_arm: float
    edge_trim_frames: int


def _effective_window(window: int, frames: int) -> int:
    if window < 3 or frames < 3:
        return 1
    effective = min(window, frames)
    if effective % 2 == 0:
        effective -= 1
    return max(1, effective)


def smooth_trajectory(trajectory: np.ndarray, window: int) -> np.ndarray:
    trajectory = np.asarray(trajectory, dtype=np.float64)
    if trajectory.ndim != 2 or trajectory.shape[1] != 2:
        raise ValueError("trajectory must have shape (T, 2)")
    effective = _effective_window(window, trajectory.shape[0])
    if effective == 1:
        return trajectory.copy()

    pad = effective // 2
    padded = np.pad(trajectory, ((pad, pad), (0, 0)), mode="edge")
    kernel = np.ones(effective, dtype=np.float64) / effective
    smoothed = np.empty_like(trajectory)
    smoothed[:, 0] = np.convolve(padded[:, 0], kernel, mode="valid")
    smoothed[:, 1] = np.convolve(padded[:, 1], kernel, mode="valid")
    return smoothed


def _high_pass_section(
    values: np.ndarray,
    b0: float,
    b1: float,
    b2: float,
    a1: float,
    a2: float,
) -> np.ndarray:
    filtered = np.empty_like(values)
    previous_input_1 = values[0].copy()
    previous_input_2 = values[0].copy()
    previous_output_1 = np.zeros(values.shape[1], dtype=np.float64)
    previous_output_2 = np.zeros(values.shape[1], dtype=np.float64)
    for index in range(values.shape[0]):
        current_input = values[index]
        current_output = (
            b0 * current_input
            + b1 * previous_input_1
            + b2 * previous_input_2
            - a1 * previous_output_1
            - a2 * previous_output_2
        )
        filtered[index] = current_output
        previous_input_2 = previous_input_1
        previous_input_1 = current_input
        previous_output_2 = previous_output_1
        previous_output_1 = current_output
    return filtered


def _odd_reflect_pad(trajectory: np.ndarray, pad: int) -> np.ndarray:
    left = 2.0 * trajectory[0] - trajectory[1 : pad + 1][::-1]
    right = 2.0 * trajectory[-1] - trajectory[-pad - 1 : -1][::-1]
    return np.vstack((left, trajectory, right))


# For a 2nd-order Butterworth section applied forward and backward, the
# cascade is -6 dB at the section cutoff. Scaling the section cutoff by this
# factor places the cascade's -3 dB point at the requested cutoff.
_ZERO_PHASE_CUTOFF_FACTOR = 0.8028


def high_pass_trajectory(
    trajectory: np.ndarray,
    fps: float,
    cutoff_hz: float,
) -> np.ndarray:
    """Zero-phase 4th-order Butterworth high-pass, -3 dB at ``cutoff_hz``."""

    trajectory = np.asarray(trajectory, dtype=np.float64)
    if trajectory.ndim != 2 or trajectory.shape[1] != 2:
        raise ValueError("trajectory must have shape (T, 2)")
    if trajectory.shape[0] < 3 or fps <= 0 or cutoff_hz <= 0:
        return np.zeros_like(trajectory)

    cutoff = min(float(cutoff_hz), 0.45 * float(fps))
    design_cutoff = min(_ZERO_PHASE_CUTOFF_FACTOR * cutoff, 0.49 * fps)
    angular = 2.0 * np.pi * design_cutoff / fps
    alpha = np.sin(angular) / np.sqrt(2.0)
    cosine = np.cos(angular)
    normalization = 1.0 + alpha
    b0 = (1.0 + cosine) / 2.0 / normalization
    b1 = -(1.0 + cosine) / normalization
    b2 = b0
    a1 = -2.0 * cosine / normalization
    a2 = (1.0 - alpha) / normalization

    pad = min(
        trajectory.shape[0] - 1,
        max(8, int(round(fps / max(cutoff, 0.1)))),
    )
    padded = _odd_reflect_pad(trajectory, pad)
    filtered = _high_pass_section(padded, b0, b1, b2, a1, a2)
    filtered = _high_pass_section(filtered[::-1], b0, b1, b2, a1, a2)[::-1]
    return filtered[pad : pad + trajectory.shape[0]]


def _median_visible(values: np.ndarray, valid: np.ndarray) -> np.ndarray:
    frames = values.shape[0]
    result = np.full((frames, 2), np.nan, dtype=np.float64)
    for frame in range(frames):
        mask = valid[frame] & np.isfinite(values[frame]).all(axis=1)
        if mask.any():
            result[frame] = np.median(values[frame][mask], axis=0)
    return result


def _interpolate_trajectory(trajectory: np.ndarray) -> np.ndarray:
    frames = trajectory.shape[0]
    valid = np.isfinite(trajectory).all(axis=1)
    if not valid.any():
        return np.zeros_like(trajectory)
    if valid.sum() == 1:
        return np.repeat(trajectory[valid], frames, axis=0)

    indices = np.arange(frames)
    result = trajectory.copy()
    for dimension in range(2):
        result[:, dimension] = np.interp(
            indices,
            indices[valid],
            trajectory[valid, dimension],
        )
    return result


def _background_mask_from_residuals(
    residuals: np.ndarray,
    frame_diagonal: float,
    min_points: int,
) -> np.ndarray:
    points = len(residuals)
    finite_indices = np.flatnonzero(np.isfinite(residuals))
    if len(finite_indices) == 0:
        return np.ones(points, dtype=bool)

    finite_values = residuals[finite_indices]
    sorted_values = np.sort(finite_values)
    noise_floor = max(1.0, 0.0015 * frame_diagonal)
    threshold = np.inf

    if len(sorted_values) >= 2 * min_points:
        best_gap = 0.0
        best_index = -1
        for index in range(min_points - 1, len(sorted_values) - 1):
            gap = float(sorted_values[index + 1] - sorted_values[index])
            if gap > best_gap:
                best_gap = gap
                best_index = index
        if best_index >= 0:
            low_median = float(np.median(sorted_values[: best_index + 1]))
            required_gap = max(3.0 * noise_floor, 0.5 * low_median + noise_floor)
            if best_gap > required_gap:
                threshold = 0.5 * (
                    float(sorted_values[best_index])
                    + float(sorted_values[best_index + 1])
                )
    else:
        center = float(np.median(finite_values))
        mad = float(np.median(np.abs(finite_values - center)))
        threshold = center + max(3.0 * noise_floor, 4.5 * mad)

    mask = np.isfinite(residuals) & (residuals <= threshold)
    if mask.sum() < min(min_points, points):
        mask = np.zeros(points, dtype=bool)
        order = finite_indices[np.argsort(finite_values)]
        mask[order[: min(min_points, len(order))]] = True
    return mask


def _displacement_residuals(
    tracks: np.ndarray,
    visibility: np.ndarray,
    reference_mask: np.ndarray,
) -> np.ndarray:
    deltas = np.diff(tracks, axis=0)
    pair_visibility = visibility[1:] & visibility[:-1]
    reference = deltas[:, reference_mask, :]
    reference_visibility = pair_visibility[:, reference_mask]
    global_delta = _median_visible(reference, reference_visibility)

    residuals = np.full(tracks.shape[1], np.inf, dtype=np.float64)
    for point in range(tracks.shape[1]):
        valid = pair_visibility[:, point] & np.isfinite(global_delta).all(axis=1)
        if valid.any():
            residuals[point] = np.median(
                np.linalg.norm(deltas[valid, point, :] - global_delta[valid], axis=1)
            )
    return residuals


def _cumulative_residuals(
    tracks: np.ndarray,
    visibility: np.ndarray,
    reference_mask: np.ndarray,
) -> np.ndarray:
    deltas = np.diff(tracks, axis=0)
    pair_visibility = visibility[1:] & visibility[:-1]
    reference = deltas[:, reference_mask, :]
    reference_visibility = pair_visibility[:, reference_mask]
    global_delta = _median_visible(reference, reference_visibility)
    global_delta = _interpolate_trajectory(global_delta)

    residuals = np.full(tracks.shape[1], np.inf, dtype=np.float64)
    for point in range(tracks.shape[1]):
        valid = pair_visibility[:, point] & np.isfinite(global_delta).all(axis=1)
        if valid.any():
            relative = deltas[valid, point, :] - global_delta[valid]
            cumulative = np.cumsum(relative, axis=0)
            residuals[point] = np.median(np.linalg.norm(cumulative, axis=1))
    return residuals


def classify_background(
    tracks: np.ndarray,
    visibility: np.ndarray,
    frame_diagonal: float,
    min_points: int,
    min_foreground_points: int = 3,
) -> np.ndarray:
    tracks = np.asarray(tracks, dtype=np.float64)
    visibility = np.asarray(visibility, dtype=bool)
    points = tracks.shape[1]
    if tracks.shape[0] < 2:
        return np.ones(points, dtype=bool)

    all_points = np.ones(points, dtype=bool)
    residuals = _displacement_residuals(tracks, visibility, all_points)
    background_mask = _background_mask_from_residuals(
        residuals, frame_diagonal, min_points
    )
    if tracks.shape[0] >= 3 and points >= 2 * min_points:
        cumulative = _cumulative_residuals(tracks, visibility, all_points)
        cumulative_background = _background_mask_from_residuals(
            cumulative, frame_diagonal, min_points
        )
        if cumulative_background.sum() >= max(min_points, points // 2 + 1):
            combined = background_mask & cumulative_background
            if combined.sum() >= min_points:
                background_mask = combined

    if points - int(background_mask.sum()) < min_foreground_points:
        return np.ones(points, dtype=bool)
    return background_mask


def robust_global_trajectory(
    tracks: np.ndarray,
    visibility: np.ndarray,
    background_mask: np.ndarray,
) -> np.ndarray:
    positions = np.asarray(tracks, dtype=np.float64)[:, background_mask, :]
    visible = np.asarray(visibility, dtype=bool)[:, background_mask]
    frames = positions.shape[0]
    if frames == 0:
        return np.zeros((0, 2), dtype=np.float64)

    first_visible = positions[0][visible[0]]
    anchor = (
        np.median(first_visible, axis=0)
        if len(first_visible)
        else np.zeros(2, dtype=np.float64)
    )
    if frames == 1:
        return anchor[None, :]

    deltas = np.diff(positions, axis=0)
    pair_visibility = visible[1:] & visible[:-1]
    global_delta = _median_visible(deltas, pair_visibility)
    global_delta = _interpolate_trajectory(global_delta)
    trajectory = anchor[None, :] + np.cumsum(global_delta, axis=0)
    return np.vstack((anchor[None, :], trajectory))


def _robust_similarity(
    source: np.ndarray,
    target: np.ndarray,
    iterations: int = 6,
) -> tuple[float, float, float, float, float]:
    """Fit target = a * source + b * rot90(source) + t with Huber IRLS."""

    source = np.asarray(source, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    count = len(source)
    if count < 3:
        raise ValueError("at least three correspondences are required")

    source_center = source.mean(axis=0)
    target_center = target.mean(axis=0)
    centered_source = source - source_center
    centered_target = target - target_center

    design = np.zeros((2 * count, 4), dtype=np.float64)
    design[0::2, 0] = centered_source[:, 0]
    design[0::2, 1] = -centered_source[:, 1]
    design[0::2, 2] = 1.0
    design[1::2, 0] = centered_source[:, 1]
    design[1::2, 1] = centered_source[:, 0]
    design[1::2, 3] = 1.0
    rhs = centered_target.reshape(-1)

    weights = np.ones(count, dtype=np.float64)
    a, b, tx, ty = 1.0, 0.0, 0.0, 0.0
    residual = np.zeros(count, dtype=np.float64)
    for _ in range(iterations):
        root = np.repeat(np.sqrt(weights), 2)
        solution, *_ = np.linalg.lstsq(
            design * root[:, None],
            rhs * root,
            rcond=None,
        )
        a, b, tx, ty = (float(value) for value in solution)
        predicted = np.column_stack(
            (
                a * centered_source[:, 0] - b * centered_source[:, 1],
                b * centered_source[:, 0] + a * centered_source[:, 1],
            )
        )
        residual = np.linalg.norm(centered_target - predicted, axis=1)
        spread = 1.4826 * np.median(np.abs(residual - np.median(residual)))
        scale = max(float(spread), 1e-3)
        ratio = residual / (1.345 * scale)
        weights = np.where(ratio <= 1.0, 1.0, 1.0 / np.maximum(ratio, 1e-12))

    tx += target_center[0] - (a * source_center[0] - b * source_center[1])
    ty += target_center[1] - (b * source_center[0] + a * source_center[1])
    return a, b, float(tx), float(ty), float(np.median(residual))


def estimate_camera_motion(
    tracks: np.ndarray,
    visibility: np.ndarray,
    background_mask: np.ndarray,
    center: np.ndarray | tuple[float, float],
) -> CameraEstimate:
    """Estimate per-frame camera translation, rotation, and scale."""

    tracks = np.asarray(tracks, dtype=np.float64)
    visibility = np.asarray(visibility, dtype=bool)
    background_mask = np.asarray(background_mask, dtype=bool)
    if tracks.ndim != 3 or tracks.shape[2] != 2:
        raise ValueError("tracks must have shape (T, P, 2)")
    if visibility.shape != tracks.shape[:2]:
        raise ValueError("visibility must match tracks shape (T, P)")
    if background_mask.shape != (tracks.shape[1],):
        raise ValueError("background_mask must have shape (P,)")
    if not background_mask.any():
        raise ValueError("background_mask must contain at least one point")

    center_array = np.asarray(center, dtype=np.float64).reshape(2)
    frames = tracks.shape[0]
    positions = np.zeros((frames, 2), dtype=np.float64)
    rotations = np.zeros(frames, dtype=np.float64)
    scales = np.ones(frames, dtype=np.float64)
    valid = np.zeros(frames, dtype=bool)
    residuals = np.full(frames, np.nan, dtype=np.float64)
    if frames == 0:
        return CameraEstimate(positions, rotations, scales, valid, float("nan"))

    positions[0] = center_array
    valid[0] = True

    points = tracks[:, background_mask, :]
    seen = visibility[:, background_mask]
    transform = np.eye(3, dtype=np.float64)
    for frame in range(1, frames):
        usable = (
            seen[frame - 1]
            & seen[frame]
            & np.isfinite(points[frame - 1]).all(axis=1)
            & np.isfinite(points[frame]).all(axis=1)
        )
        source = points[frame - 1][usable]
        target = points[frame][usable]
        step: np.ndarray | None = None
        if len(source) >= 3:
            centered = source - source.mean(axis=0)
            singular = np.linalg.svd(centered, compute_uv=False)
            full_rank = len(singular) >= 2 and singular[1] > 1e-6
            if full_rank:
                a, b, tx, ty, residual = _robust_similarity(source, target)
                angle = np.arctan2(b, a)
                step_scale = float(np.hypot(a, b))
                if (
                    np.isfinite((a, b, tx, ty, residual)).all()
                    and 0.5 <= step_scale <= 2.0
                    and abs(angle) <= 0.5
                ):
                    step = np.array(
                        ((a, -b, tx), (b, a, ty), (0.0, 0.0, 1.0)),
                        dtype=np.float64,
                    )
                    residuals[frame] = residual
            else:
                # Rotation and scale are unobservable with collinear points;
                # fall back to a robust translation-only step.
                shift = np.median(target - source, axis=0)
                deviation = np.linalg.norm(target - source - shift, axis=1)
                if np.isfinite(shift).all() and np.max(deviation) <= 1.0:
                    step = np.array(
                        ((1.0, 0.0, shift[0]), (0.0, 1.0, shift[1]), (0.0, 0.0, 1.0)),
                        dtype=np.float64,
                    )
                    residuals[frame] = float(np.median(deviation))
        if step is not None:
            transform = step @ transform
            valid[frame] = True

        position = transform @ np.array(
            (center_array[0], center_array[1], 1.0), dtype=np.float64
        )
        positions[frame] = position[:2]
        rotations[frame] = np.arctan2(transform[1, 0], transform[0, 0])
        scales[frame] = float(np.hypot(transform[0, 0], transform[1, 0]))

    if not valid.all():
        indices = np.arange(frames)
        valid_indices = indices[valid]
        for dimension in range(2):
            positions[~valid, dimension] = np.interp(
                indices[~valid],
                valid_indices,
                positions[valid, dimension],
            )
        rotations[~valid] = np.interp(
            indices[~valid], valid_indices, rotations[valid]
        )
        scales[~valid] = np.interp(indices[~valid], valid_indices, scales[valid])

    residual_value = (
        float(np.nanmedian(residuals[valid]))
        if np.isfinite(residuals[valid]).any()
        else float("nan")
    )
    return CameraEstimate(
        position=positions,
        rotation=rotations,
        scale=scales,
        valid=valid,
        residual_px=residual_value,
    )


def _edge_trim(frames: int, fps: float, cutoff_hz: float) -> int:
    """Frames excluded at each edge to skip the filter transient."""

    if frames < 3 or fps <= 0 or cutoff_hz <= 0:
        return 0
    time_constant_frames = fps / (2.0 * np.pi * cutoff_hz)
    trim = int(round(time_constant_frames))
    return int(min(frames // 4, max(1, trim)))


def _confidence_level(
    frames: int,
    background_fraction: float,
    visibility_rate: float,
    valid_fraction: float,
    residual_px: float,
    frame_diagonal: float,
) -> str:
    if (
        frames < 10
        or background_fraction < 0.2
        or visibility_rate < 0.4
        or valid_fraction < 0.5
        or (np.isfinite(residual_px) and residual_px > 0.01 * frame_diagonal)
    ):
        return "low"
    if (
        frames >= 30
        and background_fraction >= 0.5
        and visibility_rate >= 0.7
        and valid_fraction >= 0.9
        and (not np.isfinite(residual_px) or residual_px <= 0.0025 * frame_diagonal)
    ):
        return "high"
    return "medium"


def compute_metric_series(
    tracks: np.ndarray,
    visibility: np.ndarray,
    background_mask: np.ndarray,
    frame_diagonal: float,
    smooth_window: int,
    reference_jitter_ratio: float,
    high_pass_cutoff_hz: float = 1.0,
    fps: float = 30.0,
    frame_center: np.ndarray | tuple[float, float] | None = None,
) -> tuple[StabilityMetrics, CameraEstimate, np.ndarray, JitterSeries]:
    tracks = np.asarray(tracks, dtype=np.float64)
    visibility = np.asarray(visibility, dtype=bool)
    background_mask = np.asarray(background_mask, dtype=bool)
    if not background_mask.any():
        raise ValueError("background_mask must contain at least one point")

    if frame_center is None:
        first_visible = tracks[0][background_mask][visibility[0][background_mask]]
        center = (
            np.median(first_visible, axis=0)
            if len(first_visible)
            else np.zeros(2, dtype=np.float64)
        )
    else:
        center = np.asarray(frame_center, dtype=np.float64).reshape(2)

    camera = estimate_camera_motion(
        tracks,
        visibility,
        background_mask,
        center,
    )
    smoothed = smooth_trajectory(camera.position, smooth_window)

    sampled_fps = fps if fps > 0 else 30.0
    translation_jitter = high_pass_trajectory(
        camera.position,
        fps=sampled_fps,
        cutoff_hz=high_pass_cutoff_hz,
    )
    angular_jitter = high_pass_trajectory(
        np.column_stack((camera.rotation, camera.scale)),
        fps=sampled_fps,
        cutoff_hz=high_pass_cutoff_hz,
    )

    lever_arm = 0.5 * float(frame_diagonal)
    rotation_px = angular_jitter[:, 0] * lever_arm
    scale_px = angular_jitter[:, 1] * lever_arm
    magnitude = np.sqrt(
        np.sum(np.square(translation_jitter), axis=1)
        + np.square(rotation_px)
        + np.square(scale_px)
    )

    edge_trim = _edge_trim(len(magnitude), sampled_fps, high_pass_cutoff_hz)
    evaluated = (
        magnitude[edge_trim : len(magnitude) - edge_trim]
        if edge_trim > 0
        else magnitude
    )
    angular_evaluated = (
        angular_jitter[edge_trim : len(angular_jitter) - edge_trim]
        if edge_trim > 0
        else angular_jitter
    )
    rotation_px_evaluated = angular_evaluated[:, 0] * lever_arm
    scale_px_evaluated = angular_evaluated[:, 1] * lever_arm

    rms_jitter = float(np.sqrt(np.mean(np.square(evaluated))))
    p95_jitter = float(np.percentile(evaluated, 95))
    max_jitter = float(np.max(evaluated))
    rms_ratio = rms_jitter / frame_diagonal
    score = float(100.0 * np.exp(-rms_ratio / reference_jitter_ratio))

    camera_steps = np.linalg.norm(np.diff(smoothed, axis=0), axis=1)
    camera_motion = float(np.sum(camera_steps))

    foreground_mask = ~background_mask
    subject_motion = 0.0
    if foreground_mask.any() and len(tracks) > 1:
        residuals = _displacement_residuals(tracks, visibility, background_mask)
        foreground_residuals = residuals[foreground_mask]
        finite = foreground_residuals[np.isfinite(foreground_residuals)]
        if len(finite) > 0:
            subject_motion = float(np.median(finite))

    points = int(tracks.shape[1])
    background_fraction = int(background_mask.sum()) / points if points else 0.0
    visibility_rate = (
        float(visibility[:, background_mask].mean())
        if background_mask.any()
        else 0.0
    )
    valid_fraction = float(camera.valid.mean()) if len(camera.valid) else 0.0

    metrics = StabilityMetrics(
        stability_score=score,
        rms_jitter_px=rms_jitter,
        p95_jitter_px=p95_jitter,
        max_jitter_px=max_jitter,
        rms_jitter_ratio=rms_ratio,
        rotation_jitter_deg=float(
            np.degrees(np.sqrt(np.mean(np.square(angular_evaluated[:, 0]))))
        ),
        rotation_jitter_px=float(
            np.sqrt(np.mean(np.square(rotation_px_evaluated)))
        ),
        scale_jitter_ratio=float(
            np.sqrt(np.mean(np.square(angular_evaluated[:, 1])))
        ),
        scale_jitter_px=float(
            np.sqrt(np.mean(np.square(scale_px_evaluated)))
        ),
        camera_motion_px=camera_motion,
        camera_motion_ratio=camera_motion / frame_diagonal,
        subject_motion_px=subject_motion,
        subject_motion_ratio=subject_motion / frame_diagonal,
        background_points=int(background_mask.sum()),
        foreground_points=int(foreground_mask.sum()),
        background_fraction=background_fraction,
        visibility_rate=visibility_rate,
        spatial_residual_px=camera.residual_px,
        analyzed_frames=int(len(tracks)),
        valid_jitter_frames=int(len(evaluated)),
        valid_camera_frames=int(camera.valid.sum()),
        confidence=_confidence_level(
            frames=int(len(tracks)),
            background_fraction=background_fraction,
            visibility_rate=visibility_rate,
            valid_fraction=valid_fraction,
            residual_px=camera.residual_px,
            frame_diagonal=frame_diagonal,
        ),
    )
    series = JitterSeries(
        translation=translation_jitter,
        rotation_px=rotation_px,
        scale_px=scale_px,
        magnitude=magnitude,
        lever_arm=lever_arm,
        edge_trim_frames=edge_trim,
    )
    return metrics, camera, smoothed, series


def compute_metrics(
    tracks: np.ndarray,
    visibility: np.ndarray,
    background_mask: np.ndarray,
    frame_diagonal: float,
    smooth_window: int,
    reference_jitter_ratio: float,
    high_pass_cutoff_hz: float = 1.0,
    fps: float = 30.0,
    frame_center: np.ndarray | tuple[float, float] | None = None,
) -> tuple[StabilityMetrics, np.ndarray, np.ndarray, np.ndarray]:
    metrics, camera, smoothed, series = compute_metric_series(
        tracks=tracks,
        visibility=visibility,
        background_mask=background_mask,
        frame_diagonal=frame_diagonal,
        smooth_window=smooth_window,
        reference_jitter_ratio=reference_jitter_ratio,
        high_pass_cutoff_hz=high_pass_cutoff_hz,
        fps=fps,
        frame_center=frame_center,
    )
    return metrics, camera.position, smoothed, series.translation
