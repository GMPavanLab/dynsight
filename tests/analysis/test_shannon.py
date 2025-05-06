"""Pytest for dynsight.analysis.compute_entropy_gain."""

import numpy as np
import pytest

import dynsight


# Define the actual test
def test_output_files() -> None:
    rng = np.random.default_rng(12345)

    data_shape = (100, 100)
    random_data = rng.random(data_shape)
    random_labels = rng.integers(0, 5, (100,))
    wrong_labels = rng.integers(0, 5, (200, 50))

    # Test the use of the function computing entropy
    # This is necessary because of type checking:
    data_min = float(np.min(random_data))
    data_max = float(np.max(random_data))
    data_entropy = dynsight.analysis.compute_shannon(
        random_data,
        data_range=(data_min, data_max),
        n_bins=20,
    )

    expected_entropy = 0.9995963122117133004

    if isinstance(data_entropy, float):
        assert np.isclose(data_entropy, expected_entropy)

    # Test the case of empty dataset
    with pytest.raises(ValueError, match="data is empty"):
        _ = dynsight.analysis.compute_shannon(np.array([]), (0.0, 1.0), 20)

    # Test the case where labels have the wrong shape
    with pytest.raises(RuntimeError):
        _ = dynsight.analysis.compute_entropy_gain(
            random_data,
            wrong_labels,
            n_bins=20,
        )

    # Test the case where it works
    _, clustering_gain, *_ = dynsight.analysis.compute_entropy_gain(
        random_data,
        random_labels,
        n_bins=20,
    )

    expected_gain = 0.0010842808402454819

    if isinstance(clustering_gain, float):
        assert np.isclose(clustering_gain, expected_gain)
