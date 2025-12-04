"""Pytest for dynsight.utilities."""

from pathlib import Path

import numpy as np
import pytest

from dynsight.utilities import (
    find_extrema_points,
    save_xyz_from_ndarray,
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


def test_save_xyz_from_array(tmp_path: Path) -> None:
    output_path = tmp_path / "tmp.xyz"
    rng = np.random.default_rng(42)
    coords = rng.random((10, 10, 3))
    save_xyz_from_ndarray(
        output_path=output_path,
        coords=coords,
    )
    coords = rng.random((10, 10, 2))
    with pytest.raises(
        ValueError,
        match=r"coords array must have shape \(n_frames, n_atoms, 3\)\.",
    ):
        save_xyz_from_ndarray(
            output_path=output_path,
            coords=coords,
        )
