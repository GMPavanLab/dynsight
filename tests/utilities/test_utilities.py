"""Pytest for dynsight.utilities."""

import numpy as np

from dynsight.utilities import (
    find_extrema_points,
)


def test_find_extrema_points() -> None:
    x_coords = np.array([float(x) for x in np.linspace(-5, 5, 400)])
    y_coords = (x_coords + 3) * (x_coords - 2) ** 2 * (x_coords + 1) ** 3

    min_points = find_extrema_points(
        x_axis=x_coords,
        y_axis=y_coords,
        extrema_type="min",
        prominence=0.5,
    )
    max_points = find_extrema_points(
        x_axis=x_coords,
        y_axis=y_coords,
        extrema_type="max",
        prominence=0.5,
    )
    expected_max = np.array([[0.91478697, 32.36678208]])
    expected_min = np.array(
        [[-2.56892231e00, -3.47526086e01], [1.99248120e00, 7.56323248e-03]]
    )

    assert np.allclose(min_points, expected_min, atol=1e-2, rtol=1e-2)
    assert np.allclose(max_points, expected_max, atol=1e-2, rtol=1e-2)
