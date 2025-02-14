"""Pytest for dynsight.analysis.compute_entropy_gain."""

import os
import tempfile
from pathlib import Path
from typing import Generator

import numpy as np
import pytest

import dynsight

THRESHOLD = 1e-6


@pytest.fixture
def original_wd() -> Generator[Path, None, None]:
    original_dir = Path.cwd()

    # Ensure the original working directory is restored after the test
    yield original_dir

    os.chdir(original_dir)


# Define the actual test
def test_output_files(original_wd: Path) -> None:  # noqa: ARG001
    rng = np.random.default_rng(12345)

    data_shape = (100, 100)
    random_data = rng.random(data_shape)
    random_labels = rng.integers(0, 5, (100,))
    wrong_labels = rng.integers(0, 5, (200, 50))

    with tempfile.TemporaryDirectory() as temp_dir:
        os.chdir(temp_dir)

        # Test the use of the function computing entropy

        # This is necessary because of type checking:
        data_min = float(np.min(random_data))
        data_max = float(np.max(random_data))
        data_entropy = dynsight.analysis.compute_data_entropy(
            random_data,
            data_range=(data_min, data_max),
            n_bins=20,
        )

        expected_entropy = 0.9995963122117133004

        if isinstance(data_entropy, float):
            assert data_entropy - expected_entropy < THRESHOLD

        # Test the case where labels have the wrong shape
        with pytest.raises(RuntimeError):
            _ = dynsight.analysis.compute_entropy_gain(
                random_data,
                wrong_labels,
                n_bins=20,
            )

        # Test the case where it works
        clustering_gain = dynsight.analysis.compute_entropy_gain(
            random_data,
            random_labels,
            n_bins=20,
        )

        expected_gain = 0.0012652602795437606

        if isinstance(clustering_gain, float):
            assert clustering_gain - expected_gain < THRESHOLD
