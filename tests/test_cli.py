from pathlib import Path

import pytest

from video_stabilization_analyzer import cli
from video_stabilization_analyzer.video import VideoReadError

def test_cli_writes_report_and_plot(
    tmp_path, monkeypatch, synthetic_result
) -> None:
    monkeypatch.setattr(cli, "evaluate_video", lambda *a, **k: synthetic_result)

    code = cli.main(["input.mp4", "--output-dir", str(tmp_path)])

    assert code == 0
    assert (tmp_path / "report.json").is_file()
    assert (tmp_path / "report.png").is_file()


def test_cli_no_plot_skips_plot(tmp_path, monkeypatch, synthetic_result) -> None:
    monkeypatch.setattr(cli, "evaluate_video", lambda *a, **k: synthetic_result)

    cli.main(["input.mp4", "--output-dir", str(tmp_path), "--no-plot"])

    assert (tmp_path / "report.json").is_file()
    assert not (tmp_path / "report.png").exists()


def test_cli_quiet_prints_nothing(
    tmp_path, monkeypatch, capsys, synthetic_result
) -> None:
    monkeypatch.setattr(cli, "evaluate_video", lambda *a, **k: synthetic_result)

    cli.main(["input.mp4", "--output-dir", str(tmp_path), "--quiet"])

    captured = capsys.readouterr()
    assert captured.out == ""


def test_cli_reports_errors_with_exit_code(
    tmp_path, monkeypatch, capsys
) -> None:
    def fail(*args, **kwargs):
        raise VideoReadError("could not open video")

    monkeypatch.setattr(cli, "evaluate_video", fail)

    with pytest.raises(SystemExit) as info:
        cli.main(["missing.mp4", "--output-dir", str(tmp_path)])

    assert info.value.code == 1
    assert "could not open video" in capsys.readouterr().err


def test_cli_version_exits_cleanly() -> None:
    with pytest.raises(SystemExit) as info:
        cli.main(["--version"])

    assert info.value.code == 0


def test_cli_passes_analysis_options(monkeypatch, synthetic_result) -> None:
    captured = {}

    def fake_evaluate(path, config=None):
        captured["path"] = path
        captured["config"] = config
        return synthetic_result

    monkeypatch.setattr(cli, "evaluate_video", fake_evaluate)
    monkeypatch.setattr(cli, "write_json", lambda report, path: path)
    monkeypatch.setattr(cli, "save_diagnostic_plot", lambda result, path: path)
    monkeypatch.setattr(cli, "_print_summary", lambda *args: None)

    cli.main(
        [
            "input.mp4",
            "--no-plot",
            "--start-seconds",
            "1.5",
            "--duration-seconds",
            "2.5",
            "--chunk-frames",
            "120",
            "--chunk-overlap",
            "10",
            "--video-source-resolution",
        ]
    )

    config = captured["config"]
    assert captured["path"] == Path("input.mp4")
    assert config.start_seconds == 1.5
    assert config.duration_seconds == 2.5
    assert config.chunk_frames == 120
    assert config.chunk_overlap == 10
    assert config.video_source_resolution is True
