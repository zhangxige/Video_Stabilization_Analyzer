# Video Stabilization Analyzer

Evaluate the stabilization quality of one video or a batch of videos by
tracking points with CoTracker3 and analyzing the residual camera motion.

## Demo

Annotated CoTracker3 tracks (green = background/camera points) and the resulting
report card for a deliberately shaky handheld clip. The low score is the
expected detection result, not a stabilized output:

![Annotated tracking and stability metrics for test4.mp4](https://cdn.jsdelivr.net/gh/zhangxige/Video_Stabilization_Analyzer@main/docs/assets/test4-demo.gif)

The GIF lives at `docs/assets/test4-demo.gif`; it is served through jsDelivr
because GitHub's own repo image URLs redirect to `raw.githubusercontent.com`,
which is unreachable on some networks.

| Metric | Value |
|---|---:|
| Stability score | **19.2 / 100** |
| High-pass RMS jitter | 9.91 px (0.826% of frame diagonal) |
| P95 jitter | 19.28 px |
| Rotation jitter | 0.20 deg |
| Scale jitter | 0.28% |
| Background points | 144 / 144 |
| Evaluated frames | 43 / 53 |
| Confidence | high |

The source clip (`test4.mp4`) is a local test asset and is not bundled; the GIF
is generated from the annotated tracking video written by `--output-video`.

## Pipeline

1. Decode sampled frames and downscale them to a configurable maximum side.
   The measured sample rate is recorded, variable-frame-rate videos are flagged,
   and `--start-seconds`/`--duration-seconds` can restrict the analyzed window.
2. Track a regular grid of points with CoTracker3. With `--chunk-frames`, long
   videos are processed in overlapping chunks; each chunk is seeded with the
   previous chunk's tracked positions, so the same scene points continue across
   the seam.
3. Separate background/camera points from moving foreground points using
   dominant-motion per-frame and cumulative residual thresholds. Motion groups
   smaller than `--min-foreground-points` are treated as background outliers.
4. Estimate a robust per-frame **similarity** camera transform (translation +
   rotation + scale) from the background points with Huber-IRLS, accumulate it
   into a camera trajectory, and apply a zero-phase 4th-order Butterworth
   high-pass filter at `--high-pass-cutoff-hz` (default 1.0 Hz). Rotation and
   scale are converted to pixels with a `0.5 * frame_diagonal` lever arm and
   combined with translation into one per-frame jitter magnitude.
5. Exclude one filter time constant of frames at each video edge from the
   RMS/P95 statistics (the filter transient), and report a confidence level from
   frame count, background fraction, visibility, valid camera frames, and
   spatial residual.
6. Report a 0-100 stability score, diagnostic metrics, a JSON report, and an
   optional trajectory/high-pass jitter plot.

## Requirements and dependencies

### Build and runtime environment

- **Python 3.12** (see `.python-version`).
- **[uv](https://docs.astral.sh/uv/)** for environment management, installation,
  locking, and builds. The project does not use `requirements.txt`, Poetry,
  conda, or a manually managed virtualenv.
- Linux, macOS, or Windows. CUDA is optional; CPU inference works but is slower.
- Common video formats are decoded through the OpenCV wheel, so a separate
  system FFmpeg installation is normally not required.

### Python dependencies

Dependencies are declared in `pyproject.toml` and pinned in `uv.lock`.

| Group | Packages | Purpose |
|---|---|---|
| Core runtime | `numpy>=1.26`, `opencv-python-headless>=4.9`, `matplotlib>=3.8` | metrics, video decoding/visualization, plots |
| Optional `cotracker` extra | `torch>=2.2`, `torchvision>=0.17`, `einops>=0.7` | CoTracker3 inference |
| Development | `pytest>=8` | test suite |
| Build backend | `uv_build>=0.11.12,<0.12.0` | building/installing the package |

The core install can run the metric tests, but real video analysis needs the
`cotracker` extra because the CLI uses CoTracker3 for tracking.

### External CoTracker assets

The `cotracker` extra installs only the Python libraries. CoTracker3 source code
and weights are loaded through `torch.hub` on first use, which requires network
access. For offline or reproducible runs, provide local assets explicitly:

```bash
uv run vsa input.mp4 \
  --cotracker-repo /path/to/co-tracker \
  --checkpoint /path/to/scaled_offline.pth \
  --output-dir outputs/input
```

## Install

```bash
uv sync
uv sync --extra cotracker
```

`uv sync` installs the core runtime and development dependencies. Add the
`cotracker` extra for PyTorch-based tracking. On systems without CUDA, use the
CPU-only flow below. To build a wheel or source distribution with the
configured `uv_build` backend:

```bash
uv build
```

For a CPU-only PyTorch wheel, see the
[PyTorch install selector](https://pytorch.org/get-started/locally/) and install
it into the project environment with `uv pip install`.

On this 9p checkout, if uv's hardlink fallback is slow or interrupted, use:

```bash
UV_LINK_MODE=copy uv sync
```

### Windows/WSL note

Do not reuse the same `.venv` from both Windows and WSL. A venv created under
WSL contains Linux executables and a `lib64 -> lib` symlink; Windows `uv` may
then fail while trying to recreate it:

```text
failed to remove file ...\.venv\lib64: access denied (os error 5)
```

Remove the WSL-created environment from a WSL shell:

```bash
cd /mnt/c/Users/zhangxi/Documents/Git_Project/Video_Stabilization_Analyzer
rm -rf .venv
```

Then recreate it from the Windows terminal with `uv sync` (or use the CPU-only
flow below). Switch OS by deleting and recreating `.venv`, or use separate
project copies/`UV_PROJECT_ENVIRONMENT` values for Windows and WSL.

### CPU-only install (no CUDA)

On a machine without a CUDA driver/GPU, install the core environment and
`einops` first, skipping the default PyTorch wheels, then install the CPU-only
PyTorch build:

```bash
uv sync --extra cotracker \
  --no-install-package torch \
  --no-install-package torchvision

uv pip install --torch-backend cpu torch torchvision
```

Verify that CUDA is not being used:

```bash
uv run --no-sync python -c \
  "import torch; print(torch.__version__, torch.cuda.is_available())"
```

The output should show a `+cpu` build and `False`.

Run the analyzer with CPU inference explicitly:

```bash
uv run --no-sync vsa input.mp4 \
  --device cpu \
  --cotracker-repo /path/to/co-tracker \
  --checkpoint /path/to/scaled_offline.pth \
  --output-dir outputs/input
```

`--torch-backend cpu` requires a recent uv. On older versions, use the PyTorch
CPU index instead:

```bash
uv pip install \
  --index-url https://download.pytorch.org/whl/cpu \
  torch torchvision
```

After this CPU-only setup, use `uv run --no-sync ...` for this environment.
A plain `uv sync` or `uv run` may synchronize the lockfile's default PyTorch
wheels and replace the CPU build; rerun the two installation commands if that
happens. CPU inference is slower, so consider lowering `--max-side` or
`--max-frames` for large videos.

## Usage

```bash
uv run vsa input.mp4 --output-dir outputs/input
```

Useful options:

```bash
uv run vsa input.mp4 \
  --max-frames 300 \
  --frame-stride 1 \
  --max-side 960 \
  --start-seconds 0 \
  --grid-size 12 \
  --smooth-window 15 \
  --high-pass-cutoff-hz 1.0 \
  --min-foreground-points 3 \
  --device auto \
  --output-dir outputs/input \
  --output-video outputs/input/tracked.mp4
```

- `--max-frames 0` analyzes all sampled frames.
- `--max-side 0` keeps the original resolution (higher memory and GPU cost).
- `--start-seconds` and `--duration-seconds` analyze only a window of the source
  video; the report records the window and warns when the rest was skipped.
- `--device cuda` or `--device cpu` overrides automatic device selection.
- `--checkpoint path/to/scaled_offline.pth` uses local CoTracker3 weights.
- `--cotracker-repo path/to/co-tracker` loads a local CoTracker source tree
  instead of downloading it through `torch.hub`.
- `--high-pass-cutoff-hz 1.0` sets the jitter band; lower-frequency camera
  motion is treated as intentional. The filter is a forward-backward
  4th-order Butterworth whose -3 dB point is at the requested cutoff, clamped
  below Nyquist.
- `--smooth-window` affects the smoothed camera trajectory/camera-motion
  diagnostic only, not the high-pass jitter score.
- `--min-foreground-points 3` sets the smallest motion group treated as a
  moving subject; smaller groups are folded back into background.
- `--chunk-frames 120 --chunk-overlap 20` tracks long videos in overlapping
  chunks to cap CoTracker3 memory. Chunk seams reuse the previous chunk's point
  positions; results stay close to a whole-video run but are not bit-identical.
- `--output-video` (alias `--video`) writes an annotated tracking video.
  `--video-source-resolution` renders it at source resolution instead of
  `--max-side`.
- `--json` and `--plot` override the default paths under `--output-dir`.
- `--no-plot` skips plot generation.
- `--quiet` suppresses the printed summary (useful in scripts and batch runs).

A fully local/offline run looks like:

```bash
uv run vsa input.mp4 \
  --cotracker-repo /path/to/co-tracker \
  --checkpoint /path/to/scaled_offline.pth \
  --output-dir outputs/offline
```

The command prints the score, jitter metrics, background/foreground point
counts, the confidence level, and any warnings.

### Batch processing

`vsa-batch` shares the sampling, metric, and runtime options with `vsa` and
evaluates several videos in one run:

```bash
uv run vsa-batch "clips/*.mp4" \
  --output-dir outputs/batch \
  --csv outputs/batch/summary.csv
```

- Every input gets its own `report.json` under `<output-dir>/<video-stem>/`.
- `summary.csv` has one row per video with status, score, RMS jitter, rotation
  jitter, scale jitter, confidence, report path, and error text.
- A failing video is recorded with `status=error` and does not stop the run; the
  command exits with code 1 if any input failed.
- Shell-expanded paths and quoted glob patterns are both accepted.

## Output

`report.json` (schema version 2) contains:

- `video`: source resolution, analysis resolution, FPS, measured sample rate,
  window start, variable-frame-rate and orientation-metadata flags, frame counts.
- `tracking`: CoTracker backend and point/frame counts.
- `background_separation`: point counts and method.
- `score`: score definition, high-pass cutoff, reference ratio, camera model
  (`similarity`), rotation/scale lever arm, and scope.
- `metrics`: high-pass RMS/P95/max jitter, normalized jitter, rotation and scale
  jitter (degrees/percent and pixels), camera motion, subject motion, and point
  counts.
- `quality`: confidence (`low`/`medium`/`high`), background fraction, visibility
  rate, spatial residual in pixels, valid camera frames, analyzed frames.
- `warnings`: low-frame-count, low-background, truncated-window,
  variable-frame-rate, orientation-metadata, and similar cautions.

The plot shows the raw and smoothed camera trajectory plus the high-pass jitter
split into translation and rotation+scale components, with RMS/P95 lines and
shaded edge frames that are excluded from the statistics.

With `--output-video`, an annotated video is also written (MP4 by default). Tracked
points are green for background/camera points and red for foreground points;
short trails show their recent motion and a panel shows the score and frame
count.

The score is:

```text
score = 100 * exp(-rms_jitter_ratio / reference_jitter_ratio)
```

`rms_jitter_ratio` is the RMS of the combined high-pass jitter magnitude divided
by the frame diagonal. The magnitude combines:

```text
sqrt(translation_px^2 + rotation_px^2 + scale_px^2)
rotation_px = high_pass(rotation_rad) * 0.5 * frame_diagonal
scale_px    = high_pass(scale)        * 0.5 * frame_diagonal
```

The default reference ratio is `0.005`; it is a calibration choice, not a
universal threshold. Adjust `--reference-jitter-ratio` to match the target
domain.

The high-pass filter is a zero-phase 4th-order Butterworth. Its -3 dB point is at
`--high-pass-cutoff-hz`, and it uses the measured sample rate, so the cutoff
remains meaningful with `--frame-stride`, windowing, and variable-frame-rate
videos. One filter time constant of frames at each edge is excluded from the
RMS/P95 statistics to avoid start/end transients (the count is reported as
`metrics.valid_jitter_frames` vs `metrics.analyzed_frames`).

The score measures **camera/background stability only**. Subject motion is
reported separately and does not lower the score; the report marks this with
`score.scope` and `score.subject_motion_included`. Confidence reflects how much
of the frame and how many frames actually supported the camera estimate; treat
`low` as "do not compare this score with other videos".

Scores are not comparable across different `--max-side` or `--high-pass-cutoff-hz`
settings. Changing `--smooth-window` does not change the score.

## Limitations

- CoTracker3 tracks points; it does not decide what is camera motion. Automatic
  background separation assumes the dominant motion is the camera. Cumulative
  residuals catch slower coherent subject motion, but a subject that fills most
  of the frame can still be treated as background. Foreground detection is
  motion-based, not semantic: a static person is background for scoring
  purposes, and isolated motion outliers are folded back into background.
- The camera model is a similarity transform (translation, rotation, scale).
  Perspective/parallax effects, rolling shutter, and strong lens distortion are
  not modeled and can appear as jitter. Rotation and scale are weighted through
  a fixed half-diagonal lever arm, so they are representative, not per-pixel
  ground truth.
- Intentional fast camera motion can be classified as jitter. Raise
  `--high-pass-cutoff-hz` to exclude slower sway, or lower it to include more
  low-frequency movement.
- Wavelet/gyro-based stabilization cannot be verified from pixels alone; a
  perfectly stabilized video with intentional camera movement can still score
  below 100.
- Downscaling to 960 px by default reduces cost and small-amplitude jitter
  sensitivity. Use `--max-side 0` for full-resolution analysis.
- Long videos are truncated by `--max-frames`; the score only covers the
  sampled segment. Use `--duration-seconds` to choose the segment explicitly,
  or `--chunk-frames` to lower tracking memory.
- Chunked tracking reuses point positions across seams, but tracks restarted at
  each chunk can drift slightly; scores may differ by a few points from an
  unchunked run.
- Variable-frame-rate and rotation-metadata videos are reported but not
  resampled/rotated by the analyzer.

## Development

```bash
uv run pytest
uv run pytest tests/test_metrics.py
uv run pytest tests/test_metrics.py::test_shaky_tracks_score_lower
```

The tracking, metric computation, and reporting layers are separate so metric
tests run on synthetic trajectories without PyTorch, a GPU, or video files.
CI (`.github/workflows/ci.yml`) runs `uv sync --frozen`, the test suite, and
`uv build` on Linux. The CoTracker backend is exercised only by an end-to-end
run with a real video.

## Layout

```text
src/video_stabilization_analyzer/
  analysis.py      orchestration and warnings
  batch.py         vsa-batch multi-video runner and summary CSV
  cli.py           argparse entrypoint and shared option groups
  config.py        AnalysisConfig
  metrics.py       background separation, similarity camera model,
                   high-pass jitter, confidence, metrics
  points.py        grid generation
  plotting.py      diagnostic plot
  report.py        JSON report (schema version 2)
  video.py         decoding, windowing, sample-rate detection
  visualization.py annotated tracking video
  tracking/        Tracker protocol, CoTracker3 adapter, chunk stitching
tests/             synthetic-trajectory, CLI, batch, report, and video tests
```
