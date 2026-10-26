import numpy as np


def make_grid_points(
    width: int,
    height: int,
    grid_size: int,
    margin: float = 0.05,
) -> np.ndarray:
    if width < 2 or height < 2:
        raise ValueError("width and height must be >= 2")
    if grid_size < 2:
        raise ValueError("grid_size must be >= 2")
    if not 0.0 <= margin < 0.5:
        raise ValueError("margin must be in [0, 0.5)")

    x_min = margin * (width - 1)
    x_max = (1.0 - margin) * (width - 1)
    y_min = margin * (height - 1)
    y_max = (1.0 - margin) * (height - 1)

    xs = np.linspace(x_min, x_max, grid_size, dtype=np.float32)
    ys = np.linspace(y_min, y_max, grid_size, dtype=np.float32)
    grid_x, grid_y = np.meshgrid(xs, ys)
    return np.column_stack((grid_x.ravel(), grid_y.ravel())).astype(np.float32)
