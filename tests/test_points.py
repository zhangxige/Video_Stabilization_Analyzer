import numpy as np
import pytest

from video_stabilization_analyzer.points import make_grid_points


def test_grid_shape_and_bounds() -> None:
    points = make_grid_points(640, 480, 4, margin=0.0)
    assert points.shape == (16, 2)
    assert points[:, 0].min() == pytest.approx(0.0)
    assert points[:, 0].max() == pytest.approx(639.0)
    assert points[:, 1].min() == pytest.approx(0.0)
    assert points[:, 1].max() == pytest.approx(479.0)


def test_grid_margin_keeps_points_inside() -> None:
    points = make_grid_points(100, 50, 5, margin=0.1)
    assert points[:, 0].min() == pytest.approx(9.9, abs=1e-4)
    assert points[:, 0].max() == pytest.approx(89.1, abs=1e-4)
    assert points[:, 1].min() == pytest.approx(4.9, abs=1e-4)
    assert points[:, 1].max() == pytest.approx(44.1, abs=1e-4)


def test_grid_validation() -> None:
    with pytest.raises(ValueError):
        make_grid_points(10, 10, 1)
    with pytest.raises(ValueError):
        make_grid_points(10, 10, 4, margin=0.5)
    with pytest.raises(ValueError):
        make_grid_points(1, 10, 4)
