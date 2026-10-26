from video_stabilization_analyzer.plotting import save_diagnostic_plot


def test_diagnostic_plot_is_written(tmp_path, synthetic_result) -> None:
    path = save_diagnostic_plot(synthetic_result, tmp_path / "report.png")

    assert path.is_file()
    assert path.stat().st_size > 0
    assert path.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
