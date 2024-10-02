"""Pytest for dynsight.onion.onion_multi."""

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
    random_walk_x = []
    random_walk_y = []
    for _ in range(n_particles):
        tmp_x = [0.0]
        tmp_y = [0.0]
        for _ in range(n_steps - 1):
            d_x = rng.normal()
            d_y = rng.normal()
            x_new = tmp_x[-1] + d_x
            y_new = tmp_y[-1] + d_y
            tmp_x.append(x_new)
            tmp_y.append(y_new)
        random_walk_x.append(tmp_x)
        random_walk_y.append(tmp_y)

    n_windows = int(n_steps / tau_window)
    reshaped_input_data_x = np.reshape(
        np.array(random_walk_x), (n_particles * n_windows, -1)
    )
    reshaped_input_data_y = np.reshape(
        np.array(random_walk_y), (n_particles * n_windows, -1)
    )
    reshaped_input_data = np.array(
        [
            np.concatenate((tmp, reshaped_input_data_y[i]))
            for i, tmp in enumerate(reshaped_input_data_x)
        ]
    )

    with tempfile.TemporaryDirectory() as _:
        ### Test the clustering class ###
        tmp = dynsight.onion.OnionMulti()
        tmp.fit_predict(reshaped_input_data)
        _ = tmp.get_params()
        tmp.set_params()

        ### Test the clustering function ###
        state_list, labels = dynsight.onion.onion_multi(reshaped_input_data)

        _ = state_list[0].get_attributes()

        ### Define the paths to the expected output ###
        results_dir = original_wd / "tests/onion/"
        expected_output_path = results_dir / "output_multi/labels.npy"

        np.save(expected_output_path, labels)

        ### Compare the contents of the expected and actual output ###
        expected_output = np.load(expected_output_path)
        assert np.allclose(expected_output, labels, atol=1e-07)
