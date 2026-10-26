import csv
from pathlib import Path

import pytest

from video_stabilization_analyzer import batch
from video_stabilization_analyzer.video import VideoReadError


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def test_batch_writes_summary_and_reports(
    tmp_path, monkeypatch, synthetic_result
) -> None:
    monkeypatch.setattr(
        batch, "evaluate_video", lambda path, config=None: synthetic_result
    )

    code = batch.main(
        [
            "a.mp4",
            "b.mp4",
            "--output-dir",
            str(tmp_path),
            "--csv",
            str(tmp_path / "summary.csv"),
            "--quiet",
        ]
    )

    assert code == 0
    rows = _read_rows(tmp_path / "summary.csv")
    assert [row["video"] for row in rows] == ["a.mp4", "b.mp4"]
    assert all(row["status"] == "ok" for row in rows)
    assert (tmp_path / "a" / "report.json").is_file()
    assert (tmp_path / "b" / "report.json").is_file()


def test_batch_continues_after_failure(
    tmp_path, monkeypatch, synthetic_result
) -> None:
    def fake_evaluate(path, config=None):
        if str(path) == "bad.mp4":
            raise VideoReadError("could not open video")
        return synthetic_result

    monkeypatch.setattr(batch, "evaluate_video", fake_evaluate)

    code = batch.main(
        [
            "bad.mp4",
            "good.mp4",
            "--output-dir",
            str(tmp_path),
            "--csv",
            str(tmp_path / "summary.csv"),
            "--quiet",
        ]
    )

    assert code == 1
    rows = _read_rows(tmp_path / "summary.csv")
    assert rows[0]["status"] == "error"
    assert "could not open video" in rows[0]["error"]
    assert rows[1]["status"] == "ok"
    assert rows[1]["score"]


def test_batch_expands_glob_patterns(
    tmp_path, monkeypatch, synthetic_result
) -> None:
    (tmp_path / "one.mp4").write_bytes(b"")
    (tmp_path / "two.mp4").write_bytes(b"")
    monkeypatch.setattr(
        batch, "evaluate_video", lambda path, config=None: synthetic_result
    )

    code = batch.main(
        [
            str(tmp_path / "*.mp4"),
            "--output-dir",
            str(tmp_path / "out"),
            "--csv",
            str(tmp_path / "summary.csv"),
            "--quiet",
        ]
    )

    assert code == 0
    assert len(_read_rows(tmp_path / "summary.csv")) == 2


def test_batch_rejects_bad_configuration(tmp_path) -> None:
    with pytest.raises(SystemExit) as info:
        batch.main(
            [
                "a.mp4",
                "--chunk-frames",
                "10",
                "--chunk-overlap",
                "10",
                "--csv",
                str(tmp_path / "summary.csv"),
            ]
        )

    assert info.value.code == 1
