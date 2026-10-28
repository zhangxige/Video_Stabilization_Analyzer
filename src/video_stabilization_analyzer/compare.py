"""Grouped comparison of clips that show the same scene on different devices.

The analyzer's score is only comparable when the clips share the same analysis
settings and cover the same time window. ``vsa-compare`` therefore plans a
common window and sample rate per scene group before running the normal
per-video pipeline, then reports within-group rankings and cross-group product
summaries.

Input can be a manifest::

    path,group,product
    clips/scene-a/phone-a.mp4,scene-a,phone-a
    clips/scene-a/phone-b.mp4,scene-a,phone-b

or a directory layout ``clips/<group>/<product>.mp4``.
"""

from __future__ import annotations

import argparse
import csv
import glob
import io
import json
import math
import sys
from collections import defaultdict
from dataclasses import dataclass, field, replace
from pathlib import Path

import cv2

from .analysis import evaluate_video
from .cli import (
    add_metric_arguments,
    add_runtime_arguments,
    add_sampling_arguments,
    config_from_args,
)
from .config import AnalysisConfig
from .report import build_report, write_json
from .tracking.base import TrackingError
from .video import VideoProbe, VideoReadError, probe_video

VIDEO_SUFFIXES = frozenset(
    {".mp4", ".mov", ".avi", ".mkv", ".m4v", ".webm", ".mpg", ".mpeg"}
)
_CONFIDENCE_RANK = {"low": 0, "medium": 1, "high": 2}


@dataclass(frozen=True)
class ClipSpec:
    path: Path
    group: str
    product: str


@dataclass
class ClipEvaluation:
    spec: ClipSpec
    status: str
    included: bool = False
    score: float | None = None
    rms_jitter_ratio: float | None = None
    rms_jitter_px: float | None = None
    p95_jitter_px: float | None = None
    rotation_jitter_deg: float | None = None
    scale_jitter_ratio: float | None = None
    confidence: str = ""
    analyzed_duration_seconds: float | None = None
    sampled_fps: float | None = None
    evaluated_frames: int | None = None
    report_path: Path | None = None
    warnings: tuple[str, ...] = ()
    error: str = ""
    rank: int | None = None
    relative_jitter: float | None = None


@dataclass
class GroupPlan:
    group: str
    duration_seconds: float | None = None
    target_fps: float | None = None
    notes: list[str] = field(default_factory=list)
    probes: dict[Path, VideoProbe] = field(default_factory=dict)


def load_manifest(path: Path | str) -> list[ClipSpec]:
    """Read a CSV or JSON manifest into clip specifications."""
    manifest_path = Path(path)
    text = manifest_path.read_text(encoding="utf-8-sig")
    if manifest_path.suffix.lower() == ".json":
        try:
            payload = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSON manifest {manifest_path}: {exc}") from exc
        if isinstance(payload, dict):
            entries = payload.get("videos") or payload.get("clips") or payload.get("items")
            if not isinstance(entries, list):
                raise ValueError(
                    "JSON manifest must be a list or contain a 'videos' list"
                )
        elif isinstance(payload, list):
            entries = payload
        else:
            raise ValueError("JSON manifest must be a list or an object with 'videos'")
    else:
        entries = list(csv.DictReader(io.StringIO(text)))

    specs: list[ClipSpec] = []
    for index, entry in enumerate(entries, start=1):
        if not isinstance(entry, dict):
            raise ValueError(f"manifest entry {index} is not an object")
        row = {
            str(key).strip().lower(): value
            for key, value in entry.items()
            if key is not None
        }
        raw_path = row.get("path") or row.get("video") or row.get("file")
        if not raw_path:
            raise ValueError(f"manifest entry {index} has no 'path' column")
        clip_path = Path(str(raw_path).strip())
        if not clip_path.is_absolute():
            clip_path = (manifest_path.parent / clip_path).resolve()
        group = str(row.get("group") or row.get("scene") or "").strip() or "default"
        product = str(row.get("product") or row.get("device") or "").strip()
        product = product or clip_path.stem
        specs.append(ClipSpec(path=clip_path, group=group, product=product))
    _reject_duplicates(specs)
    return specs


def _reject_duplicates(specs: list[ClipSpec]) -> None:
    seen: dict[tuple[str, str], Path] = {}
    for spec in specs:
        key = (spec.group, spec.product)
        if key in seen:
            raise ValueError(
                f"duplicate product {spec.product!r} in group {spec.group!r} "
                f"({seen[key]} and {spec.path}); use a manifest with unique "
                "product names"
            )
        seen[key] = spec.path


def expand_inputs(inputs: list[Path]) -> list[Path]:
    paths: list[Path] = []
    for pattern in inputs:
        text = str(pattern)
        if any(character in text for character in "*?["):
            paths.extend(Path(match) for match in sorted(glob.glob(text)))
        else:
            paths.append(Path(pattern))
    return paths


def _directory_specs(root: Path) -> list[ClipSpec]:
    specs: list[ClipSpec] = []
    group = root.resolve().name or "default"
    for child in sorted(root.iterdir()):
        if child.is_file() and child.suffix.lower() in VIDEO_SUFFIXES:
            specs.append(ClipSpec(path=child, group=group, product=child.stem))
        elif child.is_dir():
            for video in sorted(child.iterdir()):
                if video.is_file() and video.suffix.lower() in VIDEO_SUFFIXES:
                    specs.append(
                        ClipSpec(path=video, group=child.name, product=video.stem)
                    )
    return specs


def discover_specs(inputs: list[Path]) -> list[ClipSpec]:
    """Resolve directories, files and globs into grouped clip specifications."""
    specs: list[ClipSpec] = []
    for path in expand_inputs(inputs):
        if path.is_dir():
            specs.extend(_directory_specs(path))
        else:
            specs.append(
                ClipSpec(
                    path=path,
                    group=path.resolve().parent.name or "default",
                    product=path.stem,
                )
            )
    _reject_duplicates(specs)
    return specs


def group_specs(specs: list[ClipSpec]) -> dict[str, list[ClipSpec]]:
    grouped: dict[str, list[ClipSpec]] = {}
    for spec in specs:
        grouped.setdefault(spec.group, []).append(spec)
    return grouped


def plan_group(
    group: str,
    specs: list[ClipSpec],
    config: AnalysisConfig,
    *,
    align_window: bool,
    target_fps: float,
    max_sample_fps: float,
) -> GroupPlan:
    """Choose a common time window and sample rate for one scene group."""
    plan = GroupPlan(group=group)
    for spec in specs:
        try:
            plan.probes[spec.path] = probe_video(spec.path)
        except (VideoReadError, ValueError, OSError) as exc:
            plan.notes.append(
                f"{spec.path.name}: container metadata unavailable ({exc}); "
                "using the configured sampling window"
            )

    durations: list[float] = []
    for probe in plan.probes.values():
        duration = probe.duration_seconds
        if duration is None:
            continue
        remaining = duration - config.start_seconds
        if remaining > 0.0:
            durations.append(remaining)

    available = min(durations) if durations else None
    if align_window and available is not None:
        plan.duration_seconds = available
        if config.duration_seconds is not None:
            plan.duration_seconds = min(config.duration_seconds, available)
            if available < config.duration_seconds - 0.05:
                plan.notes.append(
                    f"window shortened to {available:.2f}s by the shortest clip"
                )
    elif config.duration_seconds is not None:
        plan.duration_seconds = config.duration_seconds
    if align_window and available is None:
        plan.notes.append(
            "clip durations could not be measured; window alignment is disabled"
        )

    fps_values = [probe.fps for probe in plan.probes.values() if probe.fps > 0.0]
    if fps_values:
        target = min(fps_values)
        if target_fps and target_fps > 0.0:
            target = min(target, target_fps)
        if max_sample_fps and max_sample_fps > 0.0:
            target = min(target, max_sample_fps)
        plan.target_fps = target
        if target < 0.5 * config.high_pass_cutoff_hz:
            plan.notes.append(
                f"target sample rate {target:.2f} Hz is below the high-pass cutoff; "
                "the filter will be clamped"
            )
    else:
        plan.notes.append(
            "clip FPS could not be measured; the configured stride is used as-is"
        )

    if (
        not align_window
        and len(durations) > 1
        and max(durations) / min(durations) > 1.1
    ):
        plan.notes.append(
            "clips have different lengths and window alignment is off; the scores "
            "may not be comparable"
        )

    if plan.target_fps:
        for spec in specs:
            probe = plan.probes.get(spec.path)
            if probe is None or probe.fps <= 0.0:
                continue
            sampled = probe.fps / _frame_stride(config, probe, plan)
            if abs(sampled - plan.target_fps) / plan.target_fps > 0.1:
                plan.notes.append(
                    f"{spec.path.name}: sampled at {sampled:.1f} Hz; the group "
                    f"target is {plan.target_fps:.1f} Hz"
                )

    if config.max_frames is not None and plan.duration_seconds is not None:
        for spec in specs:
            probe = plan.probes.get(spec.path)
            if probe is None or probe.fps <= 0.0:
                continue
            stride = _frame_stride(config, probe, plan)
            needed = math.ceil(plan.duration_seconds * probe.fps / stride) + 2
            if config.max_frames < needed:
                plan.notes.append(
                    f"{spec.path.name}: --max-frames {config.max_frames} covers "
                    f"only part of the {plan.duration_seconds:.2f}s window"
                )
    return plan


def _frame_stride(
    config: AnalysisConfig, probe: VideoProbe | None, plan: GroupPlan
) -> int:
    """Pick the stride whose sampled rate is closest to the group target."""
    stride = config.frame_stride
    if probe is None or probe.fps <= 0.0 or not plan.target_fps:
        return stride
    target = plan.target_fps
    candidates = {
        max(stride, 1),
        max(stride, int(math.floor(probe.fps / target))),
        max(stride, int(math.ceil(probe.fps / target))),
    }
    best = min(
        candidates,
        key=lambda value: (abs(probe.fps / value - target), value),
    )
    return max(stride, best)


def clip_config(
    config: AnalysisConfig, probe: VideoProbe | None, plan: GroupPlan
) -> AnalysisConfig:
    """Derive a per-clip config with the group's aligned window and stride."""
    stride = _frame_stride(config, probe, plan)
    updates: dict[str, object] = {"frame_stride": stride}
    if plan.duration_seconds is not None:
        updates["duration_seconds"] = plan.duration_seconds
        if config.max_frames is None and probe is not None and probe.fps > 0.0:
            sampled_fps = probe.fps / stride
            updates["max_frames"] = (
                max(2, math.ceil(plan.duration_seconds * sampled_fps)) + 2
            )
    return replace(config, **updates)


def _safe_name(value: str) -> str:
    cleaned = "".join(
        character if character.isalnum() or character in "-._" else "_"
        for character in value
    )
    return cleaned.strip("._") or "clip"


def _evaluate_clip(
    spec: ClipSpec,
    config: AnalysisConfig,
    plan: GroupPlan,
    *,
    output_dir: Path,
    min_confidence: str,
) -> ClipEvaluation:
    evaluation = ClipEvaluation(spec=spec, status="error")
    try:
        result = evaluate_video(spec.path, config=config)
    except (VideoReadError, TrackingError, ValueError, OSError, cv2.error) as exc:
        evaluation.error = str(exc)
        return evaluation

    metrics = result.metrics
    report = build_report(result)
    report["comparison"] = {
        "group": spec.group,
        "product": spec.product,
        "aligned_duration_seconds": plan.duration_seconds,
        "target_fps": plan.target_fps,
        "frame_stride": config.frame_stride,
    }
    report_path = write_json(
        report,
        output_dir
        / "reports"
        / _safe_name(spec.group)
        / _safe_name(spec.product)
        / "report.json",
    )
    evaluation.status = "ok"
    evaluation.included = _CONFIDENCE_RANK.get(metrics.confidence, 0) >= (
        _CONFIDENCE_RANK[min_confidence]
    )
    evaluation.score = metrics.stability_score
    evaluation.rms_jitter_ratio = metrics.rms_jitter_ratio
    evaluation.rms_jitter_px = metrics.rms_jitter_px
    evaluation.p95_jitter_px = metrics.p95_jitter_px
    evaluation.rotation_jitter_deg = metrics.rotation_jitter_deg
    evaluation.scale_jitter_ratio = metrics.scale_jitter_ratio
    evaluation.confidence = metrics.confidence
    evaluation.analyzed_duration_seconds = result.metadata.analyzed_duration_seconds
    evaluation.sampled_fps = result.metadata.sampled_fps
    evaluation.evaluated_frames = metrics.valid_jitter_frames
    evaluation.report_path = report_path
    evaluation.warnings = result.warnings
    return evaluation


def run_comparison(
    specs: list[ClipSpec],
    config: AnalysisConfig,
    *,
    output_dir: Path,
    align_window: bool = True,
    target_fps: float = 0.0,
    max_sample_fps: float = 30.0,
    min_confidence: str = "low",
    quiet: bool = False,
) -> tuple[list[ClipEvaluation], dict[str, GroupPlan], int]:
    grouped = group_specs(specs)
    plans: dict[str, GroupPlan] = {}
    evaluations: list[ClipEvaluation] = []
    failures = 0

    for group, group_specs_list in grouped.items():
        plan = plan_group(
            group,
            group_specs_list,
            config,
            align_window=align_window,
            target_fps=target_fps,
            max_sample_fps=max_sample_fps,
        )
        plans[group] = plan
        if not quiet and plan.notes:
            for note in plan.notes:
                print(f"note [{group}] {note}", file=sys.stderr)
        for spec in group_specs_list:
            clip = _evaluate_clip(
                spec,
                clip_config(config, plan.probes.get(spec.path), plan),
                plan,
                output_dir=output_dir,
                min_confidence=min_confidence,
            )
            evaluations.append(clip)
            if clip.status != "ok":
                failures += 1
                if not quiet:
                    print(
                        f"  FAIL  {group}/{spec.product}: {clip.error}",
                        file=sys.stderr,
                    )
            elif not quiet:
                print(f"{clip.score:6.1f}  {group}/{spec.product}")
    return evaluations, plans, failures


def build_comparison(
    evaluations: list[ClipEvaluation],
    plans: dict[str, GroupPlan],
    settings: dict[str, object],
    *,
    tie_threshold: float = 0.03,
) -> dict[str, object]:
    groups: list[dict[str, object]] = []
    group_names = sorted({evaluation.spec.group for evaluation in evaluations})

    for group in group_names:
        items = [item for item in evaluations if item.spec.group == group]
        usable = [
            item
            for item in items
            if item.included and item.rms_jitter_ratio is not None
        ]
        usable.sort(key=lambda item: float(item.rms_jitter_ratio))
        best = float(usable[0].rms_jitter_ratio) if usable else None
        ranking: list[dict[str, object]] = []
        for item in usable:
            ratio = float(item.rms_jitter_ratio)
            item.rank = 1 + sum(
                1 for other in usable if float(other.rms_jitter_ratio) < ratio - 1e-12
            )
            item.relative_jitter = ratio / best if best and best > 0.0 else None
            ranking.append(
                {
                    "rank": item.rank,
                    "product": item.spec.product,
                    "video": str(item.spec.path),
                    "score": item.score,
                    "rms_jitter_ratio": ratio,
                    "relative_jitter": item.relative_jitter,
                    "confidence": item.confidence,
                }
            )
        second = float(usable[1].rms_jitter_ratio) if len(usable) > 1 else None
        gap = (second / best - 1.0) if second is not None and best else None
        plan = plans.get(group)
        groups.append(
            {
                "group": group,
                "clips": len(items),
                "evaluated": len(usable),
                "failed": sum(1 for item in items if item.status != "ok"),
                "excluded": sum(
                    1 for item in items if item.status == "ok" and not item.included
                ),
                "best_product": usable[0].spec.product if usable else None,
                "best_rms_jitter_ratio": best,
                "spread_ratio": (
                    float(usable[-1].rms_jitter_ratio) / best
                    if usable and best
                    else None
                ),
                "best_gap": gap,
                "within_noise": bool(gap is not None and gap < tie_threshold),
                "aligned_duration_seconds": plan.duration_seconds if plan else None,
                "target_fps": plan.target_fps if plan else None,
                "ranking": ranking,
                "notes": list(plan.notes) if plan else [],
            }
        )

    by_product: dict[str, list[ClipEvaluation]] = defaultdict(list)
    for evaluation in evaluations:
        if evaluation.included and evaluation.relative_jitter is not None:
            by_product[evaluation.spec.product].append(evaluation)

    products: list[dict[str, object]] = []
    for product, items in by_product.items():
        relatives = [float(item.relative_jitter) for item in items]
        ranks = [int(item.rank) for item in items if item.rank is not None]
        scores = [float(item.score) for item in items if item.score is not None]
        geometric = math.exp(sum(math.log(value) for value in relatives) / len(relatives))
        products.append(
            {
                "product": product,
                "groups": len({item.spec.group for item in items}),
                "clips": len(items),
                "wins": sum(1 for rank in ranks if rank == 1),
                "mean_rank": sum(ranks) / len(ranks) if ranks else None,
                "best_rank": min(ranks) if ranks else None,
                "mean_relative_jitter": sum(relatives) / len(relatives),
                "geometric_mean_relative_jitter": geometric,
                "worst_relative_jitter": max(relatives),
                "mean_score": sum(scores) / len(scores) if scores else None,
                "min_confidence": min(
                    (item.confidence for item in items),
                    key=lambda value: _CONFIDENCE_RANK.get(value, 0),
                ),
            }
        )
    def product_key(product: dict[str, object]) -> tuple[float, float]:
        mean_rank = product["mean_rank"]
        return (
            float(mean_rank) if mean_rank is not None else math.inf,
            float(product["geometric_mean_relative_jitter"]),
        )

    products.sort(key=product_key)
    for product in products:
        key = product_key(product)
        product["rank"] = 1 + sum(
            1
            for other in products
            if product_key(other)[0] < key[0] - 1e-9
            or (
                abs(product_key(other)[0] - key[0]) <= 1e-9
                and product_key(other)[1] < key[1] - 1e-9
            )
        )

    caveats = [
        "Scores are descriptive: each group usually has 2-4 clips and no "
        "significance test is applied.",
        "Jitter is normalized by the frame diagonal, so clips with different "
        "resolutions are comparable, but different crops/fields of view are not "
        "fully corrected.",
        "Compare scores only between runs that share --max-side, "
        "--high-pass-cutoff-hz and --reference-jitter-ratio.",
    ]
    if any(group["within_noise"] for group in groups):
        caveats.append(
            f"Groups marked within_noise have a best/second-best gap below "
            f"{tie_threshold * 100:.1f}%; treat the winner as undecided."
        )

    return {
        "schema_version": 1,
        "generator": "vsa-compare",
        "settings": settings,
        "groups": groups,
        "products": products,
        "clips": [
            {
                "group": item.spec.group,
                "product": item.spec.product,
                "video": str(item.spec.path),
                "status": item.status,
                "included": item.included,
                "rank": item.rank,
                "score": item.score,
                "rms_jitter_ratio": item.rms_jitter_ratio,
                "rms_jitter_px": item.rms_jitter_px,
                "relative_jitter": item.relative_jitter,
                "p95_jitter_px": item.p95_jitter_px,
                "rotation_jitter_deg": item.rotation_jitter_deg,
                "scale_jitter_ratio": item.scale_jitter_ratio,
                "confidence": item.confidence,
                "analyzed_duration_seconds": item.analyzed_duration_seconds,
                "sampled_fps": item.sampled_fps,
                "evaluated_frames": item.evaluated_frames,
                "report": str(item.report_path) if item.report_path else None,
                "warnings": list(item.warnings),
                "error": item.error,
            }
            for item in evaluations
        ],
        "caveats": caveats,
    }


def _write_csv(path: Path, fields: list[str], rows: list[dict[str, object]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    return path


def write_outputs(
    comparison: dict[str, object],
    output_dir: Path,
    *,
    plot: bool = True,
) -> dict[str, Path]:
    """Write JSON, CSV, Markdown, the dashboard and an optional heatmap."""
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}

    paths["json"] = write_json(comparison, output_dir / "comparison.json")

    paths["clips_csv"] = _write_csv(
        output_dir / "clips.csv",
        [
            "group",
            "product",
            "video",
            "status",
            "included",
            "rank",
            "score",
            "rms_jitter_ratio",
            "relative_jitter",
            "confidence",
            "analyzed_duration_seconds",
            "sampled_fps",
            "evaluated_frames",
            "report",
            "error",
        ],
        list(comparison["clips"]),
    )
    paths["products_csv"] = _write_csv(
        output_dir / "products.csv",
        [
            "rank",
            "product",
            "groups",
            "clips",
            "wins",
            "mean_rank",
            "mean_relative_jitter",
            "geometric_mean_relative_jitter",
            "worst_relative_jitter",
            "mean_score",
            "min_confidence",
        ],
        list(comparison["products"]),
    )
    heatmap_requested = plot and len(comparison["groups"]) > 1
    paths["markdown"] = _write_markdown(
        comparison,
        output_dir / "comparison.md",
        dashboard=plot,
        heatmap=heatmap_requested,
    )
    if plot:
        plot_path = save_comparison_dashboard(comparison, output_dir / "comparison.png")
        if plot_path is not None:
            paths["plot"] = plot_path
        if heatmap_requested:
            heatmap_path = save_comparison_heatmap(
                comparison, output_dir / "comparison-heatmap.png"
            )
            if heatmap_path is not None:
                paths["heatmap"] = heatmap_path
    return paths


def _format_float(value: object, digits: int = 2) -> str:
    if value is None:
        return "-"
    return f"{float(value):.{digits}f}"


def _jitter_marker(relative: object) -> str:
    """Traffic-light marker for a clip's jitter relative to the group best."""
    if relative is None:
        return ""
    value = float(relative)
    if value <= 1.05:
        return "🟢"
    if value <= 2.0:
        return "🟠"
    return "🔴"


def _write_markdown(
    comparison: dict[str, object],
    path: Path,
    *,
    dashboard: bool = True,
    heatmap: bool = False,
) -> Path:
    settings = comparison["settings"]
    if settings.get("target_fps"):
        sample_text = f"{float(settings['target_fps']):g} Hz"
    elif settings.get("max_sample_fps"):
        sample_text = (
            "auto (lowest clip FPS, capped at "
            f"{float(settings['max_sample_fps']):g} Hz)"
        )
    else:
        sample_text = "auto (lowest clip FPS)"
    lines = [
        "# Stabilization comparison",
        "",
        "Within each group the clips show the same scene, so the lowest high-pass "
        "jitter is the most stable capture. `relative` is the RMS jitter divided "
        "by the best clip in that group (1.00 = best).",
        "",
        f"- High-pass cutoff: {_format_float(settings.get('high_pass_cutoff_hz'))} Hz",
        f"- Analysis max side: {settings.get('max_side') or 'source'}",
        f"- Window alignment: {'on' if settings.get('align_window') else 'off'}",
        f"- Target sample rate: {sample_text}",
        "",
        "Markers: 🟢 within 5% of the group best, 🟠 up to 2x, 🔴 above 2x.",
        "",
    ]

    if dashboard:
        lines.extend(["![Comparison dashboard](comparison.png)", ""])
    if heatmap:
        lines.extend(
            ["![Relative jitter heatmap](comparison-heatmap.png)", ""]
        )

    for group in comparison["groups"]:
        lines.extend(
            [
                f"## Group `{group['group']}`",
                "",
                f"{group['evaluated']} evaluated of {group['clips']} clips"
                + (
                    f", {group['failed']} failed"
                    if group["failed"]
                    else ""
                )
                + (
                    f", {group['excluded']} below the confidence threshold"
                    if group["excluded"]
                    else ""
                )
                + ".",
                "",
            ]
        )
        if group["within_noise"]:
            lines.extend(
                [
                    "> The best two clips are within the tie threshold; the winner "
                    "is not statistically separated.",
                    "",
                ]
            )
        if group["ranking"]:
            leader = group["ranking"][0]
            lines.extend(
                [
                    "**Best: {product}** — score {score}, jitter {relative} vs the "
                    "group best.".format(
                        product=leader["product"],
                        score=_format_float(leader["score"], 1),
                        relative=f"{float(leader['relative_jitter']):.2f}x",
                    ),
                    "",
                    "| Rank | Product | Score | RMS jitter (diag.) | Relative | Confidence |",
                    "|---:|---|---:|---:|---:|---|",
                ]
            )
            for row in group["ranking"]:
                lines.append(
                    "| {rank} | {product} | {score} | {ratio} | {relative} | {confidence} |".format(
                        rank=row["rank"],
                        product=row["product"],
                        score=_format_float(row["score"], 1),
                        ratio=f"{float(row['rms_jitter_ratio']) * 100:.3f}%",
                        relative=(
                            f"{_jitter_marker(row['relative_jitter'])} "
                            f"{float(row['relative_jitter']):.2f}x"
                        ),
                        confidence=row["confidence"],
                    )
                )
            lines.append("")
        else:
            lines.extend(["No comparable clips in this group.", ""])
        for note in group["notes"]:
            lines.append(f"- {note}")
        if group["notes"]:
            lines.append("")

    if comparison["products"]:
        lines.extend(
            [
                "## Product summary",
                "",
                "| Rank | Product | Groups | Wins | Mean rank | Mean score | Jitter vs best (geo.) |",
                "|---:|---|---:|---:|---:|---:|---:|",
            ]
        )
        for product in comparison["products"]:
            lines.append(
                "| {rank} | {product} | {groups} | {wins} | {mean_rank} | "
                "{mean_score} | {marker} {geo} |".format(
                    rank=product["rank"],
                    product=product["product"],
                    groups=product["groups"],
                    wins=product["wins"],
                    mean_rank=_format_float(product["mean_rank"]),
                    mean_score=_format_float(product["mean_score"], 1),
                    geo=f"{float(product['geometric_mean_relative_jitter']):.2f}x",
                    marker=_jitter_marker(product["geometric_mean_relative_jitter"]),
                )
            )
        lines.append("")

    lines.extend(["## Caveats", ""])
    for caveat in comparison["caveats"]:
        lines.append(f"- {caveat}")
    lines.append("")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def save_comparison_dashboard(
    comparison: dict[str, object], path: Path | str
) -> Path | None:
    """Draw the readable overview: product ranking plus per-group jitter bars."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    groups = comparison["groups"]
    products = comparison["products"]
    if not groups or not products:
        return None

    palette = plt.get_cmap("tab10")
    colors = {
        str(product["product"]): palette(index % 10)
        for index, product in enumerate(products)
    }

    has_summary = len(groups) > 1
    columns = 1 if not has_summary else (2 if len(groups) <= 4 else 3)
    rows = math.ceil(len(groups) / columns)
    height = (3.6 + 2.7 * rows) if has_summary else (2.4 + 2.6 * rows)
    figure = plt.figure(figsize=(max(9.5, 4.8 * columns), height))
    height_ratios = ([1.35] if has_summary else []) + [1.0] * rows
    grid = figure.add_gridspec(
        rows + (1 if has_summary else 0),
        columns,
        height_ratios=height_ratios,
        hspace=0.55,
        wspace=0.35,
    )
    figure.suptitle(
        "Stabilization comparison — lower high-pass jitter is better",
        fontsize=13,
        fontweight="bold",
    )

    if has_summary:
        summary_ax = figure.add_subplot(grid[0, :])
    else:
        summary_ax = None
    summary = sorted(comparison["products"], key=lambda item: int(item["rank"]))
    summary_values = []
    if summary_ax is not None:
        summary_values = [
            float(product["geometric_mean_relative_jitter"]) for product in summary
        ]
        positions = np.arange(len(summary))
        summary_ax.barh(
            positions,
            summary_values,
            color=[colors[str(product["product"])] for product in summary],
            height=0.62,
        )
        summary_ax.set_yticks(
            positions, [str(product["product"]) for product in summary]
        )
        summary_ax.invert_yaxis()
        limit = max(summary_values) * 1.45 if summary_values else 1.5
        summary_ax.set_xlim(0.0, limit)
        summary_ax.set_xlabel(
            "geometric-mean jitter vs group best (1.00 = wins every group)",
            fontsize=8.5,
        )
        summary_ax.set_title(
            "Product ranking", loc="left", fontsize=11, fontweight="bold"
        )
        summary_ax.grid(axis="x", alpha=0.25)
        summary_ax.set_axisbelow(True)
        for spine in ("top", "right"):
            summary_ax.spines[spine].set_visible(False)
        for position, product in zip(positions, summary):
            value = float(product["geometric_mean_relative_jitter"])
            label = (
                f"{value:.2f}x · mean rank {_format_float(product['mean_rank'])} · "
                f"{product['wins']}/{product['groups']} wins"
            )
            if int(product["rank"]) == 1:
                label = f"★ {label}"
            summary_ax.text(
                value + limit * 0.015,
                position,
                label,
                va="center",
                fontsize=8.5,
            )

    row_offset = 1 if has_summary else 0
    for index, group in enumerate(groups):
        axis = figure.add_subplot(
            grid[row_offset + index // columns, index % columns]
        )
        ranking = list(group["ranking"])
        title = f"Group {group['group']} · {group['evaluated']} clips"
        if group["within_noise"]:
            title += " · tie"
        axis.set_title(title, loc="left", fontsize=10, fontweight="bold")
        if not ranking:
            axis.text(0.5, 0.5, "no comparable clips", ha="center", va="center")
            axis.set_axis_off()
            continue

        values = [float(row["relative_jitter"]) for row in ranking]
        positions = np.arange(len(ranking))
        edges = [
            "#1b1b1b" if int(row["rank"]) == 1 else "none" for row in ranking
        ]
        widths = [1.8 if int(row["rank"]) == 1 else 0.8 for row in ranking]
        axis.barh(
            positions,
            values,
            color=[colors[str(row["product"])] for row in ranking],
            height=0.62,
            edgecolor=edges,
            linewidth=widths,
        )
        axis.set_yticks(positions, [str(row["product"]) for row in ranking])
        axis.invert_yaxis()
        upper = max(values) * 1.9 if values else 1.5
        axis.set_xlim(0.0, max(upper, 1.15))
        axis.set_xlabel("RMS jitter vs group best", fontsize=8)
        axis.tick_params(axis="y", labelsize=8.5)
        axis.tick_params(axis="x", labelsize=8)
        axis.grid(axis="x", alpha=0.25)
        axis.set_axisbelow(True)
        for spine in ("top", "right"):
            axis.spines[spine].set_visible(False)
        for position, row in zip(positions, ranking):
            label = (
                f"{_format_float(row['score'], 1)} pts · "
                f"{float(row['rms_jitter_ratio']) * 100:.3f}% · "
                f"{float(row['relative_jitter']):.2f}x"
            )
            if int(row["rank"]) == 1 and group["within_noise"]:
                label += " · tie"
            axis.text(
                float(row["relative_jitter"]) + upper * 0.025,
                position,
                label,
                va="center",
                fontsize=8,
            )

    settings = comparison["settings"]
    footer = (
        f"max side {settings.get('max_side') or 'source'} · "
        f"high-pass {_format_float(settings.get('high_pass_cutoff_hz'))} Hz · "
        f"window {'aligned' if settings.get('align_window') else 'per clip'} · "
        f"sample rate "
        + (
            f"{float(settings['target_fps']):g} Hz"
            if settings.get("target_fps")
            else f"auto (cap {_format_float(settings.get('max_sample_fps'), 0)} Hz)"
        )
        + f" · tie threshold {100 * float(settings.get('tie_threshold', 0.03)):.0f}%"
    )
    figure.text(0.01, 0.012, footer, fontsize=8, color="#555555", ha="left")

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    bottom = 0.09 if has_summary else 0.115
    figure.subplots_adjust(left=0.09, right=0.98, top=0.90, bottom=bottom)
    figure.savefig(output_path, dpi=150)
    plt.close(figure)
    return output_path


def save_comparison_heatmap(
    comparison: dict[str, object], path: Path | str
) -> Path | None:
    """Draw the product-by-group matrix of relative jitter."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    groups = [group["group"] for group in comparison["groups"]]
    products = [product["product"] for product in comparison["products"]]
    if not groups or not products:
        return None

    lookup = {
        (clip["group"], clip["product"]): clip for clip in comparison["clips"]
    }
    matrix = np.full((len(products), len(groups)), np.nan)
    for row, product in enumerate(products):
        for column, group in enumerate(groups):
            clip = lookup.get((group, product))
            if clip is not None and clip.get("relative_jitter") is not None:
                matrix[row, column] = float(clip["relative_jitter"])

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(
        figsize=(max(6.0, 2.5 + 1.1 * len(groups)), max(3.0, 1.2 + 0.55 * len(products)))
    )
    colormap = matplotlib.colormaps["RdYlGn_r"].with_extremes(bad="#f0f0f0")
    finite = matrix[np.isfinite(matrix)]
    upper = max(1.35, float(finite.max())) if finite.size else 1.5
    image = axes.imshow(
        np.ma.masked_invalid(matrix),
        cmap=colormap,
        vmin=1.0,
        vmax=upper,
        aspect="auto",
    )
    axes.set_xticks(range(len(groups)), groups, rotation=30, ha="right")
    axes.set_yticks(range(len(products)), products)
    for row in range(len(products)):
        for column in range(len(groups)):
            value = matrix[row, column]
            label = "n/a" if not np.isfinite(value) else f"{value:.2f}x"
            axes.text(
                column,
                row,
                label,
                ha="center",
                va="center",
                fontsize=9,
                color="black",
                fontweight="bold",
            )
    axes.set_title(
        "High-pass RMS jitter relative to the best clip in each group",
        fontsize=11,
        fontweight="bold",
    )
    figure.colorbar(image, ax=axes, label="relative jitter (1.00 = best)")
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)
    return output_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="vsa-compare",
        description=(
            "Compare clips of the same scenes across products and report "
            "within-group rankings."
        ),
    )
    parser.add_argument(
        "inputs",
        nargs="*",
        type=Path,
        help="directories, videos or glob patterns; a directory may contain "
        "<group>/<product>.mp4 subdirectories",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        help="CSV/JSON manifest with path, group and product columns",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/compare"),
        help="directory for comparison.json, CSVs, Markdown and plots "
        "(default: outputs/compare)",
    )
    parser.add_argument(
        "--align-window",
        dest="align_window",
        action="store_true",
        default=True,
        help="analyze the same time window (shortest clip) per group (default)",
    )
    parser.add_argument(
        "--no-align-window",
        dest="align_window",
        action="store_false",
        help="analyze each clip's configured window independently",
    )
    parser.add_argument(
        "--target-fps",
        type=float,
        default=0.0,
        help="override the per-group target sample rate (0 = use the lowest clip "
        "FPS, capped by --max-sample-fps)",
    )
    parser.add_argument(
        "--max-sample-fps",
        type=float,
        default=30.0,
        help="cap the per-group target sample rate (0 = no cap, default: 30)",
    )
    parser.add_argument(
        "--tie-threshold",
        type=float,
        default=0.03,
        help="relative jitter gap below which the top two clips are called a tie "
        "(default: 0.03 = 3%%)",
    )
    parser.add_argument(
        "--min-confidence",
        choices=["low", "medium", "high"],
        default="low",
        help="exclude clips below this confidence from rankings (default: low)",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="skip the dashboard and heatmap images",
    )
    parser.add_argument(
        "--quiet", action="store_true", help="suppress per-clip output"
    )
    add_sampling_arguments(parser)
    add_metric_arguments(parser)
    add_runtime_arguments(parser)
    parser.set_defaults(max_frames=0)
    return parser


def _print_summary(comparison: dict[str, object], paths: dict[str, Path]) -> None:
    for group in comparison["groups"]:
        print(f"\n{group['group']}:")
        if not group["ranking"]:
            print("  no comparable clips")
            continue
        for row in group["ranking"]:
            flag = "  (within noise)" if group["within_noise"] and row["rank"] == 1 else ""
            print(
                f"  {row['rank']}. {row['product']:<24} "
                f"score {float(row['score']):5.1f}  "
                f"jitter {float(row['relative_jitter']):.2f}x{flag}"
            )
    if comparison["products"]:
        print("\nProducts:")
        for product in comparison["products"]:
            print(
                f"  {product['rank']}. {product['product']:<24} "
                f"mean rank {float(product['mean_rank']):.2f}  "
                f"wins {product['wins']}/{product['groups']}  "
                f"jitter {float(product['geometric_mean_relative_jitter']):.2f}x"
            )
    print()
    for name, path in paths.items():
        print(f"{name.replace('_', ' ').capitalize()}: {path}")


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        config = config_from_args(args)
    except ValueError as exc:
        parser.exit(1, f"error: {exc}\n")

    if args.tie_threshold < 0.0:
        parser.exit(1, "error: --tie-threshold must be >= 0\n")
    if args.manifest is not None and args.inputs:
        parser.exit(1, "error: pass either --manifest or input paths, not both\n")

    try:
        if args.manifest is not None:
            specs = load_manifest(args.manifest)
        else:
            if not args.inputs:
                parser.exit(1, "error: provide input paths or --manifest\n")
            specs = discover_specs(args.inputs)
    except (OSError, ValueError) as exc:
        parser.exit(1, f"error: {exc}\n")

    if not specs:
        parser.exit(1, "error: no input videos found\n")

    try:
        evaluations, plans, failures = run_comparison(
            specs,
            config,
            output_dir=args.output_dir,
            align_window=args.align_window,
            target_fps=args.target_fps,
            max_sample_fps=args.max_sample_fps,
            min_confidence=args.min_confidence,
            quiet=args.quiet,
        )
    except OSError as exc:
        parser.exit(1, f"error: {exc}\n")

    settings = {
        "align_window": args.align_window,
        "target_fps": args.target_fps or None,
        "max_sample_fps": args.max_sample_fps or None,
        "tie_threshold": args.tie_threshold,
        "min_confidence": args.min_confidence,
        "max_side": config.max_side,
        "high_pass_cutoff_hz": config.high_pass_cutoff_hz,
        "reference_jitter_ratio": config.reference_jitter_ratio,
        "grid_size": config.grid_size,
        "max_frames": config.max_frames,
    }
    comparison = build_comparison(
        evaluations,
        plans,
        settings,
        tie_threshold=args.tie_threshold,
    )
    paths = write_outputs(comparison, args.output_dir, plot=not args.no_plot)

    if not args.quiet:
        _print_summary(comparison, paths)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
