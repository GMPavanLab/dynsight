"""Pytest for dynsight.onion.onion_multi."""

import os
import tempfile
from pathlib import Path
from typing import Generator

import numpy as np
import pytest

from dynsight.trajectory import Insight


@pytest.fixture
def original_wd() -> Generator[Path, None, None]:
    original_dir = Path.cwd()

    # Ensure the original working directory is restored after the test
    yield original_dir

    os.chdir(original_dir)


# Define the actual test
def test_output_files(original_wd: Path) -> None:
    with tempfile.TemporaryDirectory() as _:
        ## Create the input data ###
        n_particles = 5
        n_steps = 1000
        rng = np.random.default_rng(12345)
        random_walk = np.zeros((n_particles, n_steps, 2))
        for i in range(n_particles):
            for j in range(1, n_steps):
                d_x, d_y = rng.normal(), rng.normal()
                random_walk[i][j][0] = random_walk[i][j - 1][0] + d_x
                random_walk[i][j][1] = random_walk[i][j - 1][1] + d_y

        data = Insight(random_walk)
        clustering = data.get_onion(delta_t=10)

        # Define the paths to the expected output files
        results_dir = original_wd / "tests/onion/"
        expected_output_path = results_dir / "output_multi/labels.npy"

        # Compare the contents of the expected and actual output
        expected_output = np.load(expected_output_path)
        assert np.allclose(expected_output, clustering.labels)
