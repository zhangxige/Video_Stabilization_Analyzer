import csv
import json
import math
from dataclasses import replace
from pathlib import Path

import pytest

from video_stabilization_analyzer import compare
from video_stabilization_analyzer.compare import (
    ClipEvaluation,
    ClipSpec,
    build_comparison,
    clip_config,
    discover_specs,
    load_manifest,
    plan_group,
)
from video_stabilization_analyzer.config import AnalysisConfig
from video_stabilization_analyzer.video import VideoProbe, VideoReadError


def _score(ratio: float) -> float:
    return 100.0 * math.exp(-ratio / 0.005)


def _result(base, path, ratio, confidence="high"):
    return replace(
        base,
        metrics=replace(
            base.metrics,
            rms_jitter_ratio=ratio,
            stability_score=_score(ratio),
            confidence=confidence,
        ),
        metadata=replace(
            base.metadata,
            path=Path(path),
            sampled_fps=30.0,
            source_total_frames=150,
            sampled_frames=150,
        ),
        warnings=(),
    )


def _probe(path, fps=30.0, frames=150):
    return VideoProbe(
        path=Path(path), width=1920, height=1080, fps=fps, frame_count=frames
    )


def _make_videos(root: Path, layout: dict[str, list[str]]) -> None:
    for group, names in layout.items():
        directory = root / group
        directory.mkdir(parents=True, exist_ok=True)
        for name in names:
            (directory / f"{name}.mp4").write_bytes(b"")


def _patch_pipeline(monkeypatch, base, ratios: dict[tuple[str, str], float]) -> None:
    def fake_evaluate(path, config=None):
        video = Path(path)
        key = (video.parent.name, video.stem)
        return _result(base, video, ratios[key])

    monkeypatch.setattr(compare, "evaluate_video", fake_evaluate)
    monkeypatch.setattr(compare, "probe_video", lambda path: _probe(path))


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def test_load_manifest_resolves_paths_and_defaults(tmp_path: Path) -> None:
    (tmp_path / "clips").mkdir()
    (tmp_path / "clips" / "a.mp4").write_bytes(b"")
    (tmp_path / "clips" / "b.mp4").write_bytes(b"")
    manifest = tmp_path / "manifest.csv"
    manifest.write_text(
        "path,group,product\nclips/a.mp4,scene,phone-a\nclips/b.mp4,scene,\n",
        encoding="utf-8",
    )

    specs = load_manifest(manifest)

    assert [spec.group for spec in specs] == ["scene", "scene"]
    assert [spec.product for spec in specs] == ["phone-a", "b"]
    assert specs[0].path == (tmp_path / "clips" / "a.mp4").resolve()


def test_load_manifest_supports_json_and_rejects_duplicates(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "videos": [
                    {"path": "a.mp4", "group": "g", "product": "x"},
                    {"path": "b.mp4", "group": "g", "product": "x"},
                ]
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="duplicate product"):
        load_manifest(manifest)


def test_discover_specs_uses_group_directories(tmp_path: Path) -> None:
    _make_videos(tmp_path, {"scene-a": ["phone-a", "phone-b"], "scene-b": ["phone-a"]})

    specs = discover_specs([tmp_path])

    assert {(spec.group, spec.product) for spec in specs} == {
        ("scene-a", "phone-a"),
        ("scene-a", "phone-b"),
        ("scene-b", "phone-a"),
    }


def test_plan_group_aligns_window_and_sample_rate(monkeypatch) -> None:
    specs = [
        ClipSpec(Path("a.mp4"), "g", "a"),
        ClipSpec(Path("b.mp4"), "g", "b"),
    ]
    probes = {
        Path("a.mp4"): _probe("a.mp4", fps=30.0, frames=300),
        Path("b.mp4"): _probe("b.mp4", fps=60.0, frames=480),
    }
    monkeypatch.setattr(compare, "probe_video", lambda path: probes[Path(path)])

    plan = plan_group(
        "g",
        specs,
        AnalysisConfig(max_frames=None),
        align_window=True,
        target_fps=0.0,
        max_sample_fps=30.0,
    )

    assert plan.duration_seconds == pytest.approx(8.0)
    assert plan.target_fps == pytest.approx(30.0)

    config = clip_config(AnalysisConfig(max_frames=None), probes[Path("b.mp4")], plan)
    assert config.frame_stride == 2
    assert config.duration_seconds == pytest.approx(8.0)
    assert config.max_frames == math.ceil(8.0 * 30.0) + 2


def test_clip_config_picks_closest_sampled_rate(monkeypatch) -> None:
    probes = {
        Path("a.mp4"): _probe("a.mp4", fps=24.0, frames=240),
        Path("b.mp4"): _probe("b.mp4", fps=60.0, frames=600),
    }
    monkeypatch.setattr(compare, "probe_video", lambda path: probes[Path(path)])
    specs = [ClipSpec(Path("a.mp4"), "g", "a"), ClipSpec(Path("b.mp4"), "g", "b")]

    plan = plan_group(
        "g",
        specs,
        AnalysisConfig(max_frames=None),
        align_window=True,
        target_fps=0.0,
        max_sample_fps=60.0,
    )
    config = clip_config(AnalysisConfig(max_frames=None), probes[Path("b.mp4")], plan)

    assert plan.target_fps == pytest.approx(24.0)
    assert config.frame_stride == 3
    assert any("group target" in note for note in plan.notes)


def test_clip_config_respects_explicit_max_frames(monkeypatch) -> None:
    monkeypatch.setattr(
        compare, "probe_video", lambda path: _probe(path, fps=30.0, frames=300)
    )
    plan = plan_group(
        "g",
        [ClipSpec(Path("a.mp4"), "g", "a")],
        AnalysisConfig(max_frames=100),
        align_window=True,
        target_fps=0.0,
        max_sample_fps=30.0,
    )

    assert plan.duration_seconds == pytest.approx(10.0)
    assert any("--max-frames" in note for note in plan.notes)

    config = clip_config(AnalysisConfig(max_frames=100), plan.probes[Path("a.mp4")], plan)

    assert config.max_frames == 100


def _evaluation(group, product, ratio, confidence="high", status="ok"):
    item = ClipEvaluation(
        spec=ClipSpec(Path(f"{group}-{product}.mp4"), group, product),
        status=status,
    )
    if status == "ok":
        item.included = confidence != "low"
        item.rms_jitter_ratio = ratio
        item.score = _score(ratio)
        item.confidence = confidence
    return item


def test_build_comparison_ranks_groups_and_products() -> None:
    evaluations = [
        _evaluation("g1", "a", 0.0010),
        _evaluation("g1", "b", 0.0020),
        _evaluation("g1", "c", 0.0015),
        _evaluation("g2", "a", 0.0030),
        _evaluation("g2", "b", 0.0035),
    ]

    comparison = build_comparison(evaluations, {}, {})

    g1 = next(group for group in comparison["groups"] if group["group"] == "g1")
    assert [row["product"] for row in g1["ranking"]] == ["a", "c", "b"]
    assert [row["rank"] for row in g1["ranking"]] == [1, 2, 3]
    assert g1["ranking"][1]["relative_jitter"] == pytest.approx(1.5)
    assert g1["within_noise"] is False

    products = {product["product"]: product for product in comparison["products"]}
    assert products["a"]["rank"] == 1
    assert products["a"]["wins"] == 2
    assert products["a"]["mean_rank"] == pytest.approx(1.0)
    assert products["b"]["mean_rank"] == pytest.approx(2.5)
    assert products["b"]["geometric_mean_relative_jitter"] == pytest.approx(
        math.sqrt(2.0 * (0.0035 / 0.0030))
    )


def test_build_comparison_ties_product_ranks() -> None:
    evaluations = [
        _evaluation("g1", "a", 0.0010),
        _evaluation("g1", "b", 0.0010),
        _evaluation("g2", "a", 0.0020),
        _evaluation("g2", "b", 0.0020),
    ]

    comparison = build_comparison(evaluations, {}, {})

    assert {product["rank"] for product in comparison["products"]} == {1}


def test_build_comparison_flags_close_results_as_within_noise() -> None:
    evaluations = [
        _evaluation("g", "a", 0.0010),
        _evaluation("g", "b", 0.00102),
    ]

    comparison = build_comparison(evaluations, {}, {}, tie_threshold=0.03)

    assert comparison["groups"][0]["within_noise"] is True
    assert any("within_noise" in caveat for caveat in comparison["caveats"])


def test_build_comparison_excludes_low_confidence_clips() -> None:
    evaluations = [
        _evaluation("g", "a", 0.0010),
        _evaluation("g", "b", 0.0005, confidence="low"),
    ]

    comparison = build_comparison(evaluations, {}, {})

    assert comparison["groups"][0]["evaluated"] == 1
    assert comparison["groups"][0]["excluded"] == 1
    assert [product["product"] for product in comparison["products"]] == ["a"]


def test_compare_cli_writes_outputs(
    tmp_path: Path, monkeypatch, synthetic_result
) -> None:
    root = tmp_path / "clips"
    _make_videos(root, {"g1": ["alpha", "beta"], "g2": ["alpha", "beta"]})
    ratios = {
        ("g1", "alpha"): 0.0010,
        ("g1", "beta"): 0.0020,
        ("g2", "alpha"): 0.0015,
        ("g2", "beta"): 0.0030,
    }
    _patch_pipeline(monkeypatch, synthetic_result, ratios)
    output = tmp_path / "out"

    code = compare.main(
        [str(root), "--output-dir", str(output), "--no-plot", "--quiet"]
    )

    assert code == 0
    comparison = json.loads((output / "comparison.json").read_text(encoding="utf-8"))
    g1 = comparison["groups"][0]
    assert g1["group"] == "g1"
    assert [row["product"] for row in g1["ranking"]] == ["alpha", "beta"]
    assert g1["ranking"][1]["relative_jitter"] == pytest.approx(2.0)
    assert comparison["products"][0]["product"] == "alpha"

    clips = _read_csv(output / "clips.csv")
    assert len(clips) == 4
    assert all(row["status"] == "ok" for row in clips)
    assert _read_csv(output / "products.csv")[0]["product"] == "alpha"

    markdown = (output / "comparison.md").read_text(encoding="utf-8")
    assert "## Group `g1`" in markdown
    assert "## Product summary" in markdown

    report = json.loads(
        (output / "reports" / "g1" / "alpha" / "report.json").read_text(
            encoding="utf-8"
        )
    )
    assert report["comparison"]["group"] == "g1"
    assert report["comparison"]["product"] == "alpha"


def test_compare_cli_continues_after_failure(
    tmp_path: Path, monkeypatch, synthetic_result
) -> None:
    root = tmp_path / "clips"
    _make_videos(root, {"g": ["a", "b"]})

    def fake_evaluate(path, config=None):
        if Path(path).stem == "b":
            raise VideoReadError("could not open video")
        return _result(synthetic_result, path, 0.001)

    monkeypatch.setattr(compare, "evaluate_video", fake_evaluate)
    monkeypatch.setattr(compare, "probe_video", lambda path: _probe(path))
    output = tmp_path / "out"

    code = compare.main(
        [str(root), "--output-dir", str(output), "--no-plot", "--quiet"]
    )

    assert code == 1
    comparison = json.loads((output / "comparison.json").read_text(encoding="utf-8"))
    g = comparison["groups"][0]
    assert g["evaluated"] == 1
    assert g["failed"] == 1
    clips = _read_csv(output / "clips.csv")
    failed = next(row for row in clips if row["product"] == "b")
    assert failed["status"] == "error"
    assert "could not open video" in failed["error"]


def test_compare_cli_min_confidence_excludes_clip(
    tmp_path: Path, monkeypatch, synthetic_result
) -> None:
    root = tmp_path / "clips"
    _make_videos(root, {"g": ["a", "b"]})

    def fake_evaluate(path, config=None):
        confidence = "low" if Path(path).stem == "b" else "high"
        return _result(synthetic_result, path, 0.002, confidence=confidence)

    monkeypatch.setattr(compare, "evaluate_video", fake_evaluate)
    monkeypatch.setattr(compare, "probe_video", lambda path: _probe(path))
    output = tmp_path / "out"

    code = compare.main(
        [
            str(root),
            "--output-dir",
            str(output),
            "--min-confidence",
            "medium",
            "--no-plot",
            "--quiet",
        ]
    )

    assert code == 0
    comparison = json.loads((output / "comparison.json").read_text(encoding="utf-8"))
    assert comparison["groups"][0]["excluded"] == 1
    clips = {row["product"]: row for row in _read_csv(output / "clips.csv")}
    assert clips["b"]["included"] == "False"
    assert clips["b"]["rank"] == ""


def test_compare_cli_rejects_manifest_and_inputs(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.csv"
    manifest.write_text("path,group,product\n", encoding="utf-8")

    with pytest.raises(SystemExit) as info:
        compare.main(
            [
                "some.mp4",
                "--manifest",
                str(manifest),
                "--output-dir",
                str(tmp_path / "out"),
            ]
        )

    assert info.value.code == 1
