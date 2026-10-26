import json
from pathlib import Path

from .analysis import AnalysisResult


def build_report(result: AnalysisResult) -> dict[str, object]:
    config = result.config
    metrics = result.metrics
    return {
        "schema_version": 2,
        "video": result.metadata.to_dict(),
        "config": config.to_dict(),
        "tracking": {
            "backend": result.track_result.backend,
            "points": result.track_result.points,
            "frames": result.track_result.frames,
        },
        "background_separation": {
            "background_points": int(result.background_mask.sum()),
            "foreground_points": int((~result.background_mask).sum()),
            "method": "dominant-motion per-frame and cumulative residuals",
        },
        "score": {
            "stability_score": metrics.stability_score,
            "scale": "0-100",
            "higher_is_better": True,
            "model": "100 * exp(-rms_jitter_ratio / reference_jitter_ratio)",
            "reference_jitter_ratio": config.reference_jitter_ratio,
            "high_pass_cutoff_hz": config.high_pass_cutoff_hz,
            "camera_model": "similarity (translation + rotation + scale)",
            "rotation_scale_lever_arm": "half frame diagonal",
            "scope": "camera/background stability",
            "subject_motion_included": False,
        },
        "metrics": metrics.to_dict(),
        "quality": {
            "confidence": metrics.confidence,
            "background_fraction": metrics.background_fraction,
            "visibility_rate": metrics.visibility_rate,
            "spatial_residual_px": metrics.spatial_residual_px,
            "valid_camera_frames": metrics.valid_camera_frames,
            "analyzed_frames": metrics.analyzed_frames,
        },
        "warnings": list(result.warnings),
    }


def write_json(report: dict[str, object], path: Path | str) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return output_path
