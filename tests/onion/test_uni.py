"""Pytest for dynsight.onion.onion_uni."""

import os
import tempfile
from pathlib import Path
from typing import Generator

import numpy as np
import pytest

import dynsight


@pytest.fixture
def original_wd() -> Generator[Path, None, None]:
    original_dir = Path.cwd()

    # Ensure the original working directory is restored after the test
    yield original_dir

    os.chdir(original_dir)


# Define the actual test
def test_output_files(original_wd: Path) -> None:
    ### Set all the analysis parameters ###
    n_particles = 5
    n_steps = 1000
    tau_window = 10

    ### Create the input data ###
    rng = np.random.default_rng(12345)
    random_walk = []
    for _ in range(n_particles):
        tmp = [0.0]
        for _ in range(n_steps - 1):
            d_x = rng.normal()
            x_new = tmp[-1] + d_x
            tmp.append(x_new)
        random_walk.append(tmp)

    n_windows = int(n_steps / tau_window)
    reshaped_input_data = np.reshape(
        np.array(random_walk), (n_particles * n_windows, -1)
    )

    with tempfile.TemporaryDirectory() as _:
        ### Test the clustering class ###
        tmp_clusterer = dynsight.onion.OnionUni()
        tmp_clusterer.fit_predict(reshaped_input_data)
        _ = tmp_clusterer.get_params()
        tmp_clusterer.set_params()

        ### Test the clustering function ###
        state_list, labels = dynsight.onion.onion_uni(reshaped_input_data)

        _ = state_list[0].get_attributes()

        ### Define the paths to the expected output ###
        results_dir = original_wd / "tests/onion/"
        expected_output_path = results_dir / "output_uni/labels.npy"

        ### Compare the contents of the expected and actual output ###
        expected_output = np.load(expected_output_path)
        assert np.allclose(expected_output, labels)
