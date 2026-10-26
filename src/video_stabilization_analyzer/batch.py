import argparse
import csv
import glob
import sys
from pathlib import Path

from .analysis import evaluate_video
from .cli import (
    add_metric_arguments,
    add_runtime_arguments,
    add_sampling_arguments,
    config_from_args,
)
from .report import build_report, write_json
from .tracking.base import TrackingError
from .video import VideoReadError

_CSV_FIELDS = [
    "video",
    "status",
    "score",
    "rms_jitter_px",
    "rotation_jitter_deg",
    "scale_jitter_px",
    "confidence",
    "report",
    "error",
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="vsa-batch",
        description="Evaluate several videos and write a summary CSV.",
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        type=Path,
        help="input videos or glob patterns",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs"),
        help="parent directory for per-video reports (default: outputs)",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path("outputs/summary.csv"),
        help="summary CSV path (default: outputs/summary.csv)",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="suppress per-video output",
    )
    add_sampling_arguments(parser)
    add_metric_arguments(parser)
    add_runtime_arguments(parser)
    return parser


def _expand_inputs(patterns: list[Path]) -> list[Path]:
    paths: list[Path] = []
    for pattern in patterns:
        text = str(pattern)
        if any(character in text for character in "*?["):
            paths.extend(Path(match) for match in sorted(glob.glob(text)))
        else:
            paths.append(pattern)
    return paths


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        config = config_from_args(args)
    except ValueError as exc:
        parser.exit(1, f"error: {exc}\n")

    paths = _expand_inputs(args.inputs)
    if not paths:
        parser.exit(1, "error: no input videos matched\n")

    rows: list[dict[str, object]] = []
    failures = 0
    for path in paths:
        row: dict[str, object] = {field: "" for field in _CSV_FIELDS}
        row["video"] = str(path)
        try:
            result = evaluate_video(path, config=config)
            report_path = write_json(
                build_report(result),
                args.output_dir / path.stem / "report.json",
            )
            metrics = result.metrics
            row.update(
                status="ok",
                score=round(metrics.stability_score, 2),
                rms_jitter_px=round(metrics.rms_jitter_px, 4),
                rotation_jitter_deg=round(metrics.rotation_jitter_deg, 4),
                scale_jitter_px=round(metrics.scale_jitter_px, 4),
                confidence=metrics.confidence,
                report=str(report_path),
            )
            if not args.quiet:
                print(f"{metrics.stability_score:6.1f}  {path}")
        except (VideoReadError, TrackingError, ValueError, OSError) as exc:
            failures += 1
            row.update(status="error", error=str(exc))
            if not args.quiet:
                print(f"  FAIL  {path}: {exc}", file=sys.stderr)
        rows.append(row)

    csv_path = args.csv
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=_CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    if not args.quiet:
        print(f"Summary: {csv_path}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
