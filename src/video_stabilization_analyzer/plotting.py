from pathlib import Path

import numpy as np

from .analysis import AnalysisResult


def save_diagnostic_plot(result: AnalysisResult, path: Path | str) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    frames = np.arange(result.metadata.sampled_frames)
    figure, axes = plt.subplots(2, 1, figsize=(10, 6.5), sharex=True)

    axes[0].plot(frames, result.global_trajectory[:, 0], label="x raw", linewidth=1.2)
    axes[0].plot(
        frames,
        result.smoothed_trajectory[:, 0],
        "--",
        label="x smoothed",
        linewidth=1.0,
    )
    axes[0].plot(frames, result.global_trajectory[:, 1], label="y raw", linewidth=1.2)
    axes[0].plot(
        frames,
        result.smoothed_trajectory[:, 1],
        "--",
        label="y smoothed",
        linewidth=1.0,
    )
    axes[0].set_ylabel("Global position (px)")
    axes[0].set_title("Camera trajectory")
    axes[0].legend(loc="best", fontsize=8)
    axes[0].grid(alpha=0.25)

    series = result.jitter_series
    translation_magnitude = np.linalg.norm(series.translation, axis=1)
    angular_magnitude = np.sqrt(series.rotation_px**2 + series.scale_px**2)
    axes[1].plot(frames, series.magnitude, color="tab:red", linewidth=1.2, label="total")
    axes[1].plot(
        frames,
        translation_magnitude,
        color="tab:blue",
        linewidth=0.9,
        alpha=0.7,
        label="translation",
    )
    axes[1].plot(
        frames,
        angular_magnitude,
        color="tab:green",
        linewidth=0.9,
        alpha=0.7,
        label="rotation + scale",
    )
    trim = series.edge_trim_frames
    if trim > 0 and trim < len(frames) // 2:
        axes[1].axvspan(
            0,
            trim,
            color="0.85",
            alpha=0.6,
            linewidth=0,
            label="excluded edge",
        )
        axes[1].axvspan(
            len(frames) - trim,
            len(frames) - 1,
            color="0.85",
            alpha=0.6,
            linewidth=0,
        )
    axes[1].axhline(
        result.metrics.rms_jitter_px,
        color="black",
        linestyle="--",
        linewidth=1.0,
        label=f"RMS = {result.metrics.rms_jitter_px:.2f} px",
    )
    axes[1].axhline(
        result.metrics.p95_jitter_px,
        color="tab:orange",
        linestyle=":",
        linewidth=1.0,
        label=f"P95 = {result.metrics.p95_jitter_px:.2f} px",
    )
    axes[1].set_xlabel("Sampled frame")
    axes[1].set_ylabel("High-pass jitter (px)")
    axes[1].set_title(
        f"High-pass jitter | {result.config.high_pass_cutoff_hz:.2f} Hz cutoff "
        f"| RMS {result.metrics.rms_jitter_px:.2f} px "
        f"| score {result.metrics.stability_score:.1f}/100"
    )
    axes[1].legend(loc="best", fontsize=8)
    axes[1].grid(alpha=0.25)

    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)
    return output_path
