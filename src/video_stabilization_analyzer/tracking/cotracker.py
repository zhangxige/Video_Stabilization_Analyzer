from pathlib import Path
from typing import Any

import numpy as np

from ..points import make_grid_points
from .base import TrackResult, TrackingError, TrackerUnavailableError
from .stitching import stitch_track_chunks

_HUB_REPO = "facebookresearch/co-tracker"
_HUB_MODEL = "cotracker3_offline"


class CoTrackerTracker:
    name = "cotracker3_offline"

    def __init__(
        self,
        grid_size: int = 12,
        grid_margin: float = 0.05,
        device: str = "auto",
        checkpoint: Path | str | None = None,
        repo_dir: Path | str | None = None,
        chunk_frames: int | None = None,
        chunk_overlap: int = 20,
    ) -> None:
        if chunk_frames is not None:
            if chunk_frames < 2:
                raise ValueError("chunk_frames must be >= 2 or None")
            if chunk_overlap < 1:
                raise ValueError("chunk_overlap must be >= 1 when chunking")
            if chunk_overlap >= chunk_frames:
                raise ValueError("chunk_overlap must be smaller than chunk_frames")
        self.grid_size = grid_size
        self.grid_margin = grid_margin
        self.device = device
        self.checkpoint = Path(checkpoint) if checkpoint else None
        self.repo_dir = Path(repo_dir) if repo_dir else None
        self.chunk_frames = chunk_frames
        self.chunk_overlap = chunk_overlap
        self._model: Any = None
        self._torch: Any = None
        self._device = "cpu"

    def _load(self) -> None:
        if self._model is not None:
            return
        try:
            import torch
        except ImportError as exc:
            raise TrackerUnavailableError(
                "CoTracker3 requires PyTorch. Install it with "
                "`uv sync --extra cotracker`."
            ) from exc

        if self.device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            device = self.device
        if device.startswith("cuda") and not torch.cuda.is_available():
            raise TrackerUnavailableError(
                "CUDA was requested but is not available to PyTorch."
            )

        repo = str(self.repo_dir) if self.repo_dir else _HUB_REPO
        use_local_repo = self.repo_dir is not None
        if use_local_repo and not self.repo_dir.is_dir():
            raise TrackingError(f"CoTracker repo not found: {self.repo_dir}")

        try:
            if self.checkpoint is None:
                model = torch.hub.load(
                    repo,
                    _HUB_MODEL,
                    source="local" if use_local_repo else "github",
                    trust_repo=True,
                )
            else:
                if not self.checkpoint.is_file():
                    raise TrackingError(f"checkpoint not found: {self.checkpoint}")
                model = torch.hub.load(
                    repo,
                    _HUB_MODEL,
                    source="local" if use_local_repo else "github",
                    pretrained=False,
                    trust_repo=True,
                )
                state_dict = torch.load(self.checkpoint, map_location="cpu")
                model.model.load_state_dict(state_dict)
        except TrackingError:
            raise
        except Exception as exc:
            raise TrackerUnavailableError(
                "Could not load CoTracker3. The first run downloads the "
                "repository and weights from the network; check connectivity "
                "and the installed PyTorch/torchvision versions."
            ) from exc

        model = model.to(device)
        model.eval()
        self._torch = torch
        self._model = model
        self._device = device

    def track(self, frames: np.ndarray) -> TrackResult:
        frames = np.asarray(frames)
        if frames.ndim != 4 or frames.shape[-1] != 3:
            raise TrackingError("frames must have shape (T, H, W, 3)")
        if frames.dtype != np.uint8:
            raise TrackingError("frames must be uint8 RGB")
        if len(frames) == 0:
            raise TrackingError("no frames to track")

        self._load()
        _, height, width, _ = frames.shape
        grid = make_grid_points(
            width=width,
            height=height,
            grid_size=self.grid_size,
            margin=self.grid_margin,
        )
        if self.chunk_frames is None or len(frames) <= self.chunk_frames:
            return self._track_segment(frames, grid)

        chunk_tracks: list[np.ndarray] = []
        chunk_visibility: list[np.ndarray] = []
        chunk_points = grid
        start = 0
        while True:
            end = min(start + self.chunk_frames, len(frames))
            segment = self._track_segment(frames[start:end], chunk_points)
            chunk_tracks.append(segment.tracks)
            chunk_visibility.append(segment.visibility)
            if end >= len(frames):
                break
            # Continue tracking the same scene points in the next chunk so
            # adjacent chunks stay comparable across the overlap.
            chunk_points = segment.tracks[
                segment.tracks.shape[0] - self.chunk_overlap
            ].copy()
            start = end - self.chunk_overlap

        tracks, visibility = stitch_track_chunks(
            chunk_tracks,
            chunk_visibility,
            overlap=self.chunk_overlap,
        )
        return TrackResult(
            tracks=tracks,
            visibility=visibility,
            backend=self.name,
        )

    def _track_segment(
        self,
        frames: np.ndarray,
        points: np.ndarray,
    ) -> TrackResult:
        torch = self._torch
        _, height, width, _ = frames.shape
        points = np.asarray(points, dtype=np.float32)
        if points.ndim != 2 or points.shape[1] != 2:
            raise TrackingError("points must have shape (P, 2)")

        video = (
            torch.from_numpy(np.ascontiguousarray(frames))
            .permute(0, 3, 1, 2)
            .unsqueeze(0)
            .to(self._device)
            .float()
        )
        queries = torch.zeros(
            (1, len(points), 3),
            dtype=torch.float32,
            device=self._device,
        )
        queries[0, :, 1:] = torch.from_numpy(points).to(self._device)

        with torch.inference_mode():
            tracks, visibility = self._model(video, queries=queries)

        tracks_array = tracks[0].detach().cpu().numpy()
        visibility_array = visibility[0].detach().cpu().numpy()
        return TrackResult(
            tracks=tracks_array,
            visibility=visibility_array,
            backend=self.name,
        )
