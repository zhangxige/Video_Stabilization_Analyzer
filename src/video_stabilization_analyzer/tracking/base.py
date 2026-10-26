from dataclasses import dataclass
from typing import Protocol

import numpy as np


class TrackingError(RuntimeError):
    pass


class TrackerUnavailableError(TrackingError):
    pass


@dataclass(frozen=True)
class TrackResult:
    tracks: np.ndarray
    visibility: np.ndarray
    backend: str

    def __post_init__(self) -> None:
        tracks = np.asarray(self.tracks, dtype=np.float32)
        visibility = np.asarray(self.visibility)
        if visibility.ndim == 3 and visibility.shape[-1] == 1:
            visibility = visibility[..., 0]
        visibility = visibility.astype(bool)

        if tracks.ndim != 3 or tracks.shape[-1] != 2:
            raise TrackingError("tracks must have shape (T, N, 2)")
        if visibility.shape != tracks.shape[:2]:
            raise TrackingError("visibility must have shape (T, N)")
        if tracks.shape[0] == 0 or tracks.shape[1] == 0:
            raise TrackingError("tracks must contain at least one frame and one point")
        if not np.isfinite(tracks).all():
            raise TrackingError("tracks contain non-finite coordinates")

        object.__setattr__(self, "tracks", tracks)
        object.__setattr__(self, "visibility", visibility)

    @property
    def frames(self) -> int:
        return int(self.tracks.shape[0])

    @property
    def points(self) -> int:
        return int(self.tracks.shape[1])


class Tracker(Protocol):
    name: str

    def track(self, frames: np.ndarray) -> TrackResult: ...
