import argparse
from pathlib import Path

from . import __version__
from .analysis import evaluate_video
from .config import AnalysisConfig
from .plotting import save_diagnostic_plot
from .report import build_report, write_json
from .tracking.base import TrackingError
from .video import VideoReadError
from .visualization import VisualizationError, render_tracking_video


def add_sampling_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--max-frames",
        type=int,
        default=300,
        help="maximum sampled frames; 0 means all (default: 300)",
    )
    parser.add_argument(
        "--frame-stride",
        type=int,
        default=1,
        help="sample every Nth frame (default: 1)",
    )
    parser.add_argument(
        "--max-side",
        type=int,
        default=960,
        help="downscale the longest side before analysis; 0 disables (default: 960)",
    )
    parser.add_argument(
        "--start-seconds",
        type=float,
        default=0.0,
        help="start analyzing at this source timestamp (default: 0.0)",
    )
    parser.add_argument(
        "--duration-seconds",
        type=float,
        help="analyze at most this many seconds from --start-seconds",
    )


def add_metric_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--grid-size",
        type=int,
        default=12,
        help="CoTracker grid size per axis (default: 12)",
    )
    parser.add_argument(
        "--grid-margin",
        type=float,
        default=0.05,
        help="fractional image margin excluded from the grid (default: 0.05)",
    )
    parser.add_argument(
        "--smooth-window",
        type=int,
        default=15,
        help="trajectory smoothing window for camera motion (default: 15)",
    )
    parser.add_argument(
        "--high-pass-cutoff-hz",
        type=float,
        default=1.0,
        help="high-pass cutoff for jitter measurement in Hz (default: 1.0)",
    )
    parser.add_argument(
        "--reference-jitter-ratio",
        type=float,
        default=0.005,
        help="jitter ratio that maps to a score of about 36.8 (default: 0.005)",
    )
    parser.add_argument(
        "--min-background-points",
        type=int,
        default=10,
        help="minimum points used as background (default: 10)",
    )
    parser.add_argument(
        "--min-foreground-points",
        type=int,
        default=3,
        help="minimum foreground points; smaller groups are treated as background "
        "(default: 3)",
    )


def add_runtime_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--chunk-frames",
        type=int,
        default=0,
        help="track in chunks of this many frames; 0 disables (default: 0)",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=20,
        help="overlap frames used to stitch chunks (default: 20)",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="PyTorch device: auto, cpu, cuda, cuda:0, ... (default: auto)",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        help="local CoTracker3 checkpoint; skips automatic weight download",
    )
    parser.add_argument(
        "--cotracker-repo",
        type=Path,
        help="local co-tracker source tree; avoids downloading it via torch.hub",
    )


def config_from_args(args: argparse.Namespace) -> AnalysisConfig:
    return AnalysisConfig(
        max_frames=None if args.max_frames == 0 else args.max_frames,
        frame_stride=args.frame_stride,
        max_side=None if args.max_side == 0 else args.max_side,
        start_seconds=args.start_seconds,
        duration_seconds=args.duration_seconds,
        grid_size=args.grid_size,
        grid_margin=args.grid_margin,
        smooth_window=args.smooth_window,
        high_pass_cutoff_hz=args.high_pass_cutoff_hz,
        reference_jitter_ratio=args.reference_jitter_ratio,
        min_background_points=args.min_background_points,
        min_foreground_points=args.min_foreground_points,
        chunk_frames=None if args.chunk_frames == 0 else args.chunk_frames,
        chunk_overlap=args.chunk_overlap,
        video_source_resolution=getattr(args, "video_source_resolution", False),
        device=args.device,
        checkpoint=args.checkpoint,
        cotracker_repo=args.cotracker_repo,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="vsa",
        description="Evaluate video stabilization quality from tracked trajectories.",
    )
    parser.add_argument("input", type=Path, help="input video path")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs"),
        help="directory for default report paths (default: outputs)",
    )
    parser.add_argument("--json", type=Path, help="JSON report path")
    parser.add_argument("--plot", type=Path, help="diagnostic plot path")
    parser.add_argument("--no-plot", action="store_true", help="skip the plot")
    parser.add_argument(
        "--output-video",
        "--video",
        dest="output_video",
        type=Path,
        help="annotated tracking video path",
    )
    parser.add_argument(
        "--video-source-resolution",
        action="store_true",
        help="render the annotated video at source resolution instead of --max-side",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="suppress the summary printed after a successful run",
    )
    add_sampling_arguments(parser)
    add_metric_arguments(parser)
    add_runtime_arguments(parser)
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )
    return parser


def _print_summary(
    result,
    json_path: Path,
    plot_path: Path | None,
    video_path: Path | None,
) -> None:
    metrics = result.metrics
    print(f"Stability score: {metrics.stability_score:.1f}/100")
    print(f"High-pass cutoff: {result.config.high_pass_cutoff_hz:.2f} Hz")
    print(
        f"High-pass RMS jitter: {metrics.rms_jitter_px:.3f} px "
        f"({metrics.rms_jitter_ratio * 100:.3f}% of frame diagonal)"
    )
    print(f"High-pass P95 jitter: {metrics.p95_jitter_px:.3f} px")
    print(
        f"Rotation jitter: {metrics.rotation_jitter_deg:.3f} deg "
        f"({metrics.rotation_jitter_px:.3f} px)"
    )
    print(
        f"Scale jitter: {metrics.scale_jitter_ratio * 100:.3f}% "
        f"({metrics.scale_jitter_px:.3f} px)"
    )
    print(
        f"Background points: {metrics.background_points}/{result.track_result.points}"
        f" ({result.metadata.sampled_frames} frames, "
        f"{metrics.valid_jitter_frames} evaluated)"
    )
    print(
        f"Subject motion: {metrics.subject_motion_px:.3f} px "
        f"({metrics.foreground_points} foreground points)"
    )
    print(f"Confidence: {metrics.confidence}")
    for warning in result.warnings:
        print(f"warning: {warning}")
    print(f"Report: {json_path}")
    if plot_path is not None:
        print(f"Plot: {plot_path}")
    if video_path is not None:
        print(f"Video: {video_path}")


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        config = config_from_args(args)
        result = evaluate_video(args.input, config=config)
    except (VideoReadError, TrackingError, ValueError, OSError) as exc:
        parser.exit(1, f"error: {exc}\n")

    report_path = write_json(
        build_report(result),
        args.json or args.output_dir / "report.json",
    )
    plot_path = None
    if not args.no_plot:
        plot_path = save_diagnostic_plot(
            result,
            args.plot or args.output_dir / "report.png",
        )
    video_path = None
    if args.output_video:
        try:
            video_path = render_tracking_video(result, args.output_video)
        except (VisualizationError, OSError) as exc:
            parser.exit(1, f"error: {exc}\n")
    if not args.quiet:
        _print_summary(result, report_path, plot_path, video_path)
    return 0
