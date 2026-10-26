import numpy as np
import pytest

from video_stabilization_analyzer.tracking.stitching import stitch_track_chunks


def _global_tracks(frames: int = 20, points: int = 9) -> np.ndarray:
    time = np.arange(frames, dtype=np.float64)[:, None, None]
    tracks = np.zeros((frames, points, 2), dtype=np.float64)
    tracks[:, :, 0] = 100.0 + 2.0 * time[:, 0, 0][:, None]
    tracks[:, :, 1] = 50.0 + 0.5 * time[:, 0, 0][:, None]
    return tracks


def test_stitch_chunks_recovers_global_tracks() -> None:
    tracks = _global_tracks()
    visibility = np.ones(tracks.shape[:2], dtype=bool)
    chunks = [(0, 8), (6, 14), (12, 20)]

    chunk_tracks = []
    chunk_visibility = []
    for start, end in chunks:
        chunk_tracks.append(tracks[start:end].copy())
        chunk_visibility.append(visibility[start:end].copy())

    stitched, stitched_visibility = stitch_track_chunks(
        chunk_tracks,
        chunk_visibility,
        overlap=2,
    )

    assert stitched.shape == tracks.shape
    assert stitched_visibility.shape == visibility.shape
    assert np.allclose(stitched, tracks, atol=1e-9)
    assert stitched_visibility.all()


def test_stitch_chunks_uses_visible_points_for_alignment() -> None:
    tracks = _global_tracks()
    visibility = np.ones(tracks.shape[:2], dtype=bool)
    chunk_tracks = [tracks[:10].copy(), tracks[8:].copy()]
    chunk_visibility = [visibility[:10].copy(), visibility[8:].copy()]
    # Corrupt the first point of the second chunk's overlap: it must not
    # dominate the offset estimate.
    chunk_tracks[1][:2, 0, :] += 500.0
    chunk_visibility[1][:2, 0] = False

    stitched, _ = stitch_track_chunks(chunk_tracks, chunk_visibility, overlap=2)

    assert np.allclose(stitched, tracks, atol=1e-9)


def test_stitch_chunks_without_overlap_concatenates() -> None:
    tracks = _global_tracks()
    visibility = np.ones(tracks.shape[:2], dtype=bool)
    chunk_tracks = [tracks[:10].copy(), tracks[10:].copy()]
    chunk_visibility = [visibility[:10].copy(), visibility[10:].copy()]

    stitched, _ = stitch_track_chunks(chunk_tracks, chunk_visibility, overlap=0)

    assert np.allclose(stitched, tracks)


def test_stitch_chunks_rejects_bad_overlap() -> None:
    tracks = _global_tracks(frames=4)
    visibility = np.ones(tracks.shape[:2], dtype=bool)
    with pytest.raises(ValueError):
        stitch_track_chunks(
            [tracks[:2], tracks[2:]],
            [visibility[:2], visibility[2:]],
            overlap=2,
        )
