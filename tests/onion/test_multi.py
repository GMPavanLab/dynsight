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
    ### Load the input data (already in the correct shape) ###
    results_dir = original_wd / "tests/onion/"
    input_data = np.load(results_dir / "output_multi/2D_trajectory.npy")

    with tempfile.TemporaryDirectory() as _:
        ### Test the clustering class ###
        tmp = dynsight.onion.OnionMulti()
        tmp.fit_predict(input_data)
        _ = tmp.get_params()
        tmp.set_params()

        ### Test the clustering function ###
        state_list, labels = dynsight.onion.onion_multi(input_data)

        _ = state_list[0].get_attributes()

        ### Define the paths to the expected output ###
        expected_output_path = results_dir / "output_multi/labels.npy"

        ### Compare the contents of the expected and actual output ###
        expected_output = np.load(expected_output_path)
        assert np.allclose(expected_output, labels)
