"""Pytest for dynsight.analysis.time_correlations."""

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
def test_output_files(original_wd: Path) -> None:
    rng = np.random.default_rng(12345)

    # Generate random walks
    n_part = 5
    n_frames = 100
    random_walk = np.zeros((n_part, n_frames))
    x_pos = 0.0
    for i in range(n_part):
        for j in range(n_frames):
            displ = rng.random(1)
            x_pos += displ[0]
            random_walk[i][j] = x_pos

    with tempfile.TemporaryDirectory() as temp_dir:
        os.chdir(temp_dir)

        # Test the self-correlation function
        t_corr, std_dev = dynsight.analysis.self_time_correlation(random_walk)

        exp_t_corr = np.load(original_wd / "tests/analysis/tcorr/t_corr.npy")
        exp_std_dev = np.load(original_wd / "tests/analysis/tcorr/std_dev.npy")

        assert np.allclose(t_corr, exp_t_corr)
        assert np.allclose(std_dev, exp_std_dev)

        # Test the cross-correlation function
        t_corr, std_dev = dynsight.analysis.cross_time_correlation(random_walk)

        exp_t_corr = np.load(original_wd / "tests/analysis/tcorr/c_corr.npy")
        exp_std_dev = np.load(original_wd / "tests/analysis/tcorr/ctd_dev.npy")

        assert np.allclose(t_corr, exp_t_corr)
        assert np.allclose(std_dev, exp_std_dev)
