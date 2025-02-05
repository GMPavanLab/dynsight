"""Pytest for dynsight.analysis.compute_sample_entropy."""

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

    with tempfile.TemporaryDirectory() as temp_dir:
        os.chdir(temp_dir)

        # Test the use of the function computing entropy
        data_sample_entropy = dynsight.analysis.compute_sample_entropy(
            random_data,
        )

        expected_entropy = 2.3133123011773167

        if isinstance(data_sample_entropy, float):
            assert np.isclose(data_sample_entropy, expected_entropy)

        # Test the case where time-series are too short
        with pytest.raises(ValueError, match="Time-series too short"):
            _ = dynsight.analysis.compute_sample_entropy(
                random_data,
                m_par=105,
            )

        # Test the case where threshold is too strict
        random_data = rng.random(100)
        with pytest.raises(ValueError, match="Distance threshold too strict"):
            _ = dynsight.analysis.sample_entropy(
                random_data,
                r_factor=0.001,
            )
