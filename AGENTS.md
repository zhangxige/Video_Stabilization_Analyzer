# AGENTS.md

## Project

- Given one input video, evaluate its stabilization/stability quality.
- Pipeline: decode/downscale frames -> CoTracker3 grid tracking -> separate background from moving subjects -> robust similarity camera trajectory (translation + rotation + scale) -> zero-phase Butterworth high-pass -> jitter/score -> JSON report + plot.
- CoTracker3 is only a tracker. Stability is derived from the high-pass residual of the background camera trajectory; do not treat raw track motion as instability. Subject motion is diagnostic only.

## Toolchain

- Python 3.12, managed exclusively with `uv`. Do not add `requirements.txt`, Poetry, conda, or a hand-managed venv.
- `uv sync` installs the core app and dev tools.
- `uv sync --extra cotracker` installs PyTorch/torchvision/einops for the real tracker. The extra is heavy; metrics tests must not require it.
- Run the CLI with `uv run vsa <video> --output-dir outputs/<name>`; batch with `uv run vsa-batch "<glob>" --output-dir outputs/batch --csv outputs/batch/summary.csv`.
- Tests: `uv run pytest`; single test: `uv run pytest tests/test_metrics.py::test_shaky_tracks_score_lower`.
- CI: `.github/workflows/ci.yml` runs `uv sync --frozen`, `uv run --no-sync pytest`, and `uv build`.
- No lint or typecheck command is configured.
- This checkout lives on a 9p mount. If `uv` hardlink fallback makes installs slow or interrupted, use `UV_LINK_MODE=copy uv sync`.

## Architecture

- `src/video_stabilization_analyzer/video.py`: decoding, downscaling (`--max-side`, default 960), sampling (`--max-frames`, default 300), `--start-seconds`/`--duration-seconds` windowing, measured sample FPS, VFR detection, orientation metadata.
- `tracking/`: `Tracker` protocol plus `CoTrackerTracker`; `stitching.py` joins chunked tracks. Torch is imported lazily; never import torch at module import time.
- `analysis.py`: orchestration and user-facing warnings (truncation, VFR, low confidence).
- `metrics.py`: background classification, Huber-IRLS similarity camera model, zero-phase 4th-order Butterworth high-pass, edge trim, confidence, score. Keep this layer numpy-only so it is testable on synthetic trajectories.
- `report.py`, `plotting.py`, `visualization.py`, `cli.py`, `batch.py`: output and entry layers. `visualization.py` draws background/foreground points and trails onto sampled frames.
- `tests/`: synthetic-trajectory, CLI, batch, report, plotting, stitching, visualization, and OpenCV video tests. No CoTracker integration test (network/GPU).

## Gotchas

- First CoTracker run downloads the repo and weights through `torch.hub`; it needs network access. `--cotracker-repo` + `--checkpoint` allow offline/local loading.
- Background classification assumes dominant motion is camera motion. A moving subject filling most of the frame can be selected as background; this is a known limitation, not a bug to hide.
- The score `100 * exp(-rms_jitter_ratio / reference_jitter_ratio)` is a heuristic calibration, not a universal ground truth. `--reference-jitter-ratio` and `--high-pass-cutoff-hz` intentionally change results; `--smooth-window` only affects the smoothed camera-motion diagnostic. Never trim the score using `smooth_window` (regression test: `test_score_is_independent_of_smooth_window`).
- The high-pass is a forward-backward biquad with a `0.8028` design-cutoff factor so the cascade is -3 dB at the requested cutoff. Edge frames (one filter time constant) are excluded from RMS/P95.
- Chunked tracking must seed each chunk with the previous chunk's tracked point positions at the overlap start. Re-gridding breaks point correspondence and produces a spurious rotation/translation jump at every seam.
- Rotation/scale jitter is converted to pixels with a `0.5 * frame_diagonal` lever arm. It is a representative weighting, not per-pixel ground truth.
- Analysis coordinates are in the downscaled frame; pixel metrics are not comparable across different `--max-side` values. Use ratios for comparisons.
- Keep tracking, metric computation, reporting, and visualization separable. The fake-tracker pattern in `tests/test_analysis.py` is the intended way to test pipeline changes without GPU/video.
