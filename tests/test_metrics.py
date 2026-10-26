import numpy as np
import pytest

from video_stabilization_analyzer.metrics import (
    classify_background,
    compute_metrics,
    high_pass_trajectory,
    robust_global_trajectory,
    smooth_trajectory,
)


def _panning_tracks(frames: int = 90, points: int = 100) -> np.ndarray:
    time = np.arange(frames, dtype=np.float32)[:, None]
    tracks = np.zeros((frames, points, 2), dtype=np.float32)
    tracks[:, :, 0] = 0.3 * time
    tracks[:, :, 1] = 100.0
    return tracks


def test_smooth_trajectory_preserves_linear_motion_in_interior() -> None:
    time = np.arange(60, dtype=np.float64)[:, None]
    trajectory = np.column_stack((2.0 * time[:, 0], 0.5 * time[:, 0]))
    smoothed = smooth_trajectory(trajectory, window=15)
    interior = slice(7, -7)
    assert np.allclose(smoothed[interior], trajectory[interior], atol=1e-9)


def test_smooth_trajectory_short_input_is_unchanged() -> None:
    trajectory = np.array([[1.0, 2.0], [3.0, 4.0]])
    smoothed = smooth_trajectory(trajectory, window=15)
    assert np.allclose(smoothed, trajectory)


def test_high_pass_filter_separates_low_and_high_frequency_motion() -> None:
    fps = 30.0
    frames = 240
    time = np.arange(frames, dtype=np.float64) / fps
    interior = slice(30, -30)
    low = np.column_stack(
        (np.sin(2.0 * np.pi * 0.25 * time), np.zeros(frames))
    )
    high = np.column_stack(
        (np.sin(2.0 * np.pi * 5.0 * time), np.zeros(frames))
    )

    low_filtered = high_pass_trajectory(low, fps, 1.0)[:, 0]
    high_filtered = high_pass_trajectory(high, fps, 1.0)[:, 0]
    low_gain = np.sqrt(np.mean(low_filtered[interior] ** 2)) / np.sqrt(
        np.mean(low[interior, 0] ** 2)
    )
    high_gain = np.sqrt(np.mean(high_filtered[interior] ** 2)) / np.sqrt(
        np.mean(high[interior, 0] ** 2)
    )

    assert low_gain < 0.15
    assert high_gain > 0.9


def test_high_pass_filter_removes_linear_camera_motion() -> None:
    frames = 180
    time = np.arange(frames, dtype=np.float64)[:, None]
    trajectory = np.column_stack((2.0 * time[:, 0], -0.5 * time[:, 0]))

    filtered = high_pass_trajectory(trajectory, 30.0, 1.0)

    assert np.max(np.linalg.norm(filtered[30:-30], axis=1)) < 0.5


def test_high_pass_filter_has_three_db_point_at_cutoff() -> None:
    fps = 30.0
    frames = 600
    time = np.arange(frames, dtype=np.float64) / fps
    interior = slice(60, -60)

    for cutoff in (1.0, 2.0):
        x = np.sin(2.0 * np.pi * cutoff * time)
        y = high_pass_trajectory(
            np.column_stack((x, np.zeros(frames))), fps, cutoff
        )[:, 0]
        gain = np.sqrt(np.mean(y[interior] ** 2)) / np.sqrt(
            np.mean(x[interior] ** 2)
        )
        assert gain == pytest.approx(2.0**-0.5, abs=0.03)

        slow = np.sin(2.0 * np.pi * cutoff / 5.0 * time)
        slow_y = high_pass_trajectory(
            np.column_stack((slow, np.zeros(frames))), fps, cutoff
        )[:, 0]
        slow_gain = np.sqrt(np.mean(slow_y[interior] ** 2)) / np.sqrt(
            np.mean(slow[interior] ** 2)
        )
        assert slow_gain < 0.02


def test_static_tracks_score_high() -> None:
    frames, points = 90, 64
    tracks = np.zeros((frames, points, 2), dtype=np.float32)
    tracks[:, :, 0] = 320.0
    tracks[:, :, 1] = 240.0
    visibility = np.ones((frames, points), dtype=bool)
    mask = classify_background(tracks, visibility, 800.0, 10)
    metrics, _, _, _ = compute_metrics(
        tracks, visibility, mask, 800.0, 15, 0.005
    )
    assert mask.all()
    assert metrics.stability_score > 99.9
    assert metrics.rms_jitter_px < 1e-4


def test_shaky_tracks_score_lower() -> None:
    frames, points = 120, 64
    time = np.arange(frames, dtype=np.float32)
    shake = 5.0 * np.sin(2.0 * np.pi * time / 4.0)
    bob = 3.0 * np.cos(2.0 * np.pi * time / 4.0)
    tracks = np.zeros((frames, points, 2), dtype=np.float32)
    tracks[:, :, 0] = shake[:, None]
    tracks[:, :, 1] = bob[:, None]
    visibility = np.ones((frames, points), dtype=bool)
    mask = classify_background(tracks, visibility, 800.0, 10)
    metrics, _, _, _ = compute_metrics(
        tracks, visibility, mask, 800.0, 15, 0.005
    )
    assert mask.all()
    assert metrics.stability_score < 90.0
    assert metrics.rms_jitter_px > 3.0


def test_high_frequency_motion_scores_lower_than_low_frequency_motion() -> None:
    frames, points = 240, 32
    fps = 30.0
    time = np.arange(frames, dtype=np.float64) / fps
    visibility = np.ones((frames, points), dtype=bool)

    low_tracks = np.zeros((frames, points, 2), dtype=np.float64)
    low_tracks[:, :, 0] = (5.0 * np.sin(2.0 * np.pi * 0.25 * time))[:, None]
    high_tracks = np.zeros((frames, points, 2), dtype=np.float64)
    high_tracks[:, :, 0] = (5.0 * np.sin(2.0 * np.pi * 5.0 * time))[:, None]

    low_mask = classify_background(low_tracks, visibility, 800.0, 10)
    high_mask = classify_background(high_tracks, visibility, 800.0, 10)
    low_metrics, _, _, _ = compute_metrics(
        low_tracks,
        visibility,
        low_mask,
        800.0,
        15,
        0.005,
        high_pass_cutoff_hz=1.0,
        fps=fps,
    )
    high_metrics, _, _, _ = compute_metrics(
        high_tracks,
        visibility,
        high_mask,
        800.0,
        15,
        0.005,
        high_pass_cutoff_hz=1.0,
        fps=fps,
    )

    assert low_metrics.rms_jitter_px < 0.5
    assert high_metrics.rms_jitter_px > 3.0
    assert low_metrics.stability_score > high_metrics.stability_score


def test_moving_points_are_separated_from_background() -> None:
    frames, points = 90, 100
    tracks = _panning_tracks(frames, points)
    time = np.arange(frames, dtype=np.float32)[:, None]
    tracks[:, 80:, 0] += 8.0 * time
    visibility = np.ones((frames, points), dtype=bool)

    mask = classify_background(tracks, visibility, 800.0, 10)
    assert mask[:80].all()
    assert not mask[80:].any()

    metrics, _, _, _ = compute_metrics(
        tracks, visibility, mask, 800.0, 15, 0.005
    )
    assert metrics.background_points == 80
    assert metrics.foreground_points == 20
    assert metrics.subject_motion_px == pytest.approx(8.0, abs=1e-3)


def test_slow_moving_points_are_separated_by_cumulative_motion() -> None:
    frames, points = 120, 100
    tracks = _panning_tracks(frames, points)
    time = np.arange(frames, dtype=np.float32)[:, None]
    tracks[:, 80:, 0] += 1.5 * time
    visibility = np.ones((frames, points), dtype=bool)

    mask = classify_background(tracks, visibility, 800.0, 10)
    assert mask[:80].all()
    assert not mask[80:].any()


def test_isolated_foreground_points_are_treated_as_background() -> None:
    frames, points = 120, 100
    tracks = _panning_tracks(frames, points)
    time = np.arange(frames, dtype=np.float32)[:, None]
    tracks[:, 80:82, 0] += 1.5 * time
    visibility = np.ones((frames, points), dtype=bool)

    mask = classify_background(tracks, visibility, 800.0, 10)
    assert mask.all()


def test_global_trajectory_survives_point_dropout() -> None:
    frames, points = 60, 100
    time = np.arange(frames, dtype=np.float64)[:, None]
    tracks = np.zeros((frames, points, 2), dtype=np.float64)
    tracks[:, :, 0] = 1.5 * time
    tracks[:, :, 1] = 100.0
    visibility = np.ones((frames, points), dtype=bool)
    visibility[30:, :20] = False
    tracks[30:, :20, :] = 0.0

    trajectory = robust_global_trajectory(
        tracks,
        visibility,
        np.ones(points, dtype=bool),
    )
    steps = np.diff(trajectory[:, 0])
    assert np.allclose(steps, 1.5, atol=1e-9)


def test_compute_metrics_requires_background() -> None:
    tracks = _panning_tracks(10, 10)
    visibility = np.ones((10, 10), dtype=bool)
    mask = np.zeros(10, dtype=bool)
    with pytest.raises(ValueError):
        compute_metrics(tracks, visibility, mask, 800.0, 15, 0.005)


def _similarity_tracks(
    frames: int,
    points: int = 64,
    rotation_rad: np.ndarray | None = None,
    scale: np.ndarray | None = None,
    center: tuple[float, float] = (400.0, 400.0),
) -> np.ndarray:
    grid_x, grid_y = np.meshgrid(
        np.linspace(100.0, 700.0, points // 8),
        np.linspace(100.0, 700.0, 8),
    )
    base = np.column_stack((grid_x.ravel(), grid_y.ravel()))
    center_array = np.asarray(center)
    rotation = (
        np.zeros(frames) if rotation_rad is None else np.asarray(rotation_rad)
    )
    zoom = np.ones(frames) if scale is None else np.asarray(scale)
    tracks = np.empty((frames, len(base), 2), dtype=np.float64)
    for frame in range(frames):
        angle = rotation[frame]
        matrix = zoom[frame] * np.array(
            ((np.cos(angle), -np.sin(angle)), (np.sin(angle), np.cos(angle)))
        )
        tracks[frame] = center_array + (base - center_array) @ matrix.T
    return tracks


def test_score_is_independent_of_smooth_window() -> None:
    frames, points = 120, 64
    time = np.arange(frames, dtype=np.float64)
    tracks = np.zeros((frames, points, 2), dtype=np.float64)
    tracks[:, :, 0] = (5.0 * np.sin(2.0 * np.pi * time / 4.0))[:, None]
    tracks[:, :, 1] = (3.0 * np.cos(2.0 * np.pi * time / 4.0))[:, None]
    visibility = np.ones((frames, points), dtype=bool)
    mask = classify_background(tracks, visibility, 800.0, 10)

    scores = []
    rms_values = []
    for window in (3, 15, 31):
        metrics, _, _, _ = compute_metrics(
            tracks, visibility, mask, 800.0, window, 0.005
        )
        scores.append(metrics.stability_score)
        rms_values.append(metrics.rms_jitter_px)

    assert np.allclose(scores, scores[0], atol=1e-9)
    assert np.allclose(rms_values, rms_values[0], atol=1e-9)


def test_rotation_jitter_is_detected() -> None:
    frames, fps = 240, 30.0
    time = np.arange(frames, dtype=np.float64) / fps
    steady = _similarity_tracks(frames, rotation_rad=np.full(frames, 0.1))
    shaking = _similarity_tracks(
        frames,
        rotation_rad=0.1 + 0.02 * np.sin(2.0 * np.pi * 5.0 * time),
    )
    visibility = np.ones((frames, steady.shape[1]), dtype=bool)

    steady_metrics, _, _, _ = compute_metrics(
        steady, visibility, np.ones(steady.shape[1], bool), 1131.0, 15, 0.005
    )
    shaking_metrics, _, _, _ = compute_metrics(
        shaking, visibility, np.ones(shaking.shape[1], bool), 1131.0, 15, 0.005
    )

    assert steady_metrics.rotation_jitter_deg < 1e-6
    assert shaking_metrics.rotation_jitter_deg > 0.5
    assert shaking_metrics.stability_score < steady_metrics.stability_score - 5.0


def test_scale_jitter_is_detected() -> None:
    frames, fps = 240, 30.0
    time = np.arange(frames, dtype=np.float64) / fps
    steady = _similarity_tracks(frames, scale=np.full(frames, 1.05))
    zooming = _similarity_tracks(
        frames,
        scale=1.05 + 0.01 * np.sin(2.0 * np.pi * 5.0 * time),
    )
    visibility = np.ones((frames, steady.shape[1]), dtype=bool)

    steady_metrics, _, _, _ = compute_metrics(
        steady, visibility, np.ones(steady.shape[1], bool), 1131.0, 15, 0.005
    )
    zooming_metrics, _, _, _ = compute_metrics(
        zooming, visibility, np.ones(zooming.shape[1], bool), 1131.0, 15, 0.005
    )

    assert steady_metrics.scale_jitter_px < 1e-6
    assert zooming_metrics.scale_jitter_px > 1.5
    assert zooming_metrics.stability_score < steady_metrics.stability_score - 5.0


def test_slow_rotation_and_zoom_are_not_jitter() -> None:
    frames, fps = 240, 30.0
    time = np.arange(frames, dtype=np.float64) / fps
    tracks = _similarity_tracks(
        frames,
        rotation_rad=0.15 * np.sin(2.0 * np.pi * 0.15 * time),
        scale=1.0 + 0.05 * np.sin(2.0 * np.pi * 0.15 * time + 0.5),
    )
    visibility = np.ones((frames, tracks.shape[1]), dtype=bool)

    metrics, _, _, _ = compute_metrics(
        tracks, visibility, np.ones(tracks.shape[1], bool), 1131.0, 15, 0.005
    )

    assert metrics.rms_jitter_px < 1.0
    assert metrics.stability_score > 95.0


def test_low_visibility_lowers_confidence() -> None:
    frames, points = 60, 64
    tracks = np.zeros((frames, points, 2), dtype=np.float64)
    tracks[:, :, 0] = 400.0
    tracks[:, :, 1] = 400.0
    visibility = np.ones((frames, points), dtype=bool)
    visibility[10:] = False

    metrics, _, _, _ = compute_metrics(
        tracks, visibility, np.ones(points, bool), 800.0, 15, 0.005
    )

    assert metrics.visibility_rate < 0.5
    assert metrics.confidence == "low"
