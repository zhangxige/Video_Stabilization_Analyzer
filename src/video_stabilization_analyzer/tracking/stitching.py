from collections.abc import Sequence

import numpy as np


def stitch_track_chunks(
    chunk_tracks: Sequence[np.ndarray],
    chunk_visibility: Sequence[np.ndarray],
    overlap: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Join per-chunk tracks by aligning each chunk on the overlap frames."""

    if len(chunk_tracks) != len(chunk_visibility):
        raise ValueError("chunk_tracks and chunk_visibility must have equal length")
    if not chunk_tracks:
        raise ValueError("at least one chunk is required")
    if overlap < 0:
        raise ValueError("overlap must be >= 0")

    first_tracks = np.asarray(chunk_tracks[0])
    points = first_tracks.shape[1] if first_tracks.ndim == 3 else 0
    tracks = first_tracks.copy()
    visibility = np.asarray(chunk_visibility[0]).copy()

    for index in range(1, len(chunk_tracks)):
        current_tracks = np.asarray(chunk_tracks[index])
        current_visibility = np.asarray(chunk_visibility[index])
        if current_tracks.ndim != 3 or current_tracks.shape[1] != points:
            raise ValueError("all chunks must track the same number of points")
        if current_tracks.shape[0] <= overlap:
            raise ValueError("overlap must be smaller than every chunk length")

        offset = np.zeros(2, dtype=np.float64)
        if overlap > 0:
            previous_tracks = tracks[-overlap:]
            previous_visibility = visibility[-overlap:]
            per_frame: list[np.ndarray] = []
            for frame in range(overlap):
                usable = (
                    previous_visibility[frame]
                    & current_visibility[frame]
                    & np.isfinite(previous_tracks[frame]).all(axis=1)
                    & np.isfinite(current_tracks[frame]).all(axis=1)
                )
                if usable.any():
                    per_frame.append(
                        np.median(
                            previous_tracks[frame][usable]
                            - current_tracks[frame][usable],
                            axis=0,
                        )
                    )
            if per_frame:
                offset = np.median(np.asarray(per_frame), axis=0)

        tracks = np.vstack((tracks, current_tracks[overlap:] + offset))
        visibility = np.vstack((visibility, current_visibility[overlap:]))

    return tracks, visibility
