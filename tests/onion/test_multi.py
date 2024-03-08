"""Pytest for dynsight.onion.main_2d."""

import os
import tempfile
from pathlib import Path
from typing import Generator

import dynsight
import numpy as np
import pytest


@pytest.fixture()
def original_wd() -> Generator[Path, None, None]:
    original_dir = Path.cwd()

    # Ensure the original working directory is restored after the test
    yield original_dir

    os.chdir(original_dir)


# Define the actual test
def test_output_files(original_wd: Path) -> None:
    ### Create the input data ###
    rng = np.random.default_rng(12345)
    random_walk_x = []
    random_walk_y = []
    for _ in range(5):
        tmp_x = [0.0]
        tmp_y = [0.0]
        for _ in range(1000):
            d_x = rng.normal()
            d_y = rng.normal()
            x_new = tmp_x[-1] + d_x
            y_new = tmp_y[-1] + d_y
            tmp_x.append(x_new)
            tmp_y.append(y_new)
        random_walk_x.append(tmp_x)
        random_walk_y.append(tmp_y)
    random_walk_arr = np.array([random_walk_x, random_walk_y])

    with tempfile.TemporaryDirectory() as temp_dir:
        os.chdir(temp_dir)
        data_directory = "input_data.npy"
        np.save(data_directory, random_walk_arr)

        onion_cl = dynsight.onion.OnionMulti(
            path_to_input="../input_data.npy",
            tau_w=10,
            num_tau_w=1,
            min_tau_w=10,
            max_tau_w=10,
            max_t_smooth=1,
        )

        onion_cl.run()

        # Define the paths to the expected and actual output files
        results_dir = original_wd / "tests/onion/"
        expected_output_path_1 = results_dir / "output_multi/final_states.txt"
        expected_output_path_2 = (
            results_dir / "output_multi/number_of_states.txt"
        )
        expected_output_path_3 = results_dir / "output_multi/fraction_0.txt"
        actual_output_path_1 = "onion_output/final_states.txt"
        actual_output_path_2 = "onion_output/number_of_states.txt"
        actual_output_path_3 = "onion_output/fraction_0.txt"

        # Compare "final_states.txt"
        exp_file = Path(expected_output_path_1)
        act_file = Path(actual_output_path_1)

        with exp_file.open(mode="r") as file:
            lines = file.readlines()
        tmp_data_1 = []
        for line in lines[1:]:
            line1 = line.replace("[", "").replace("]", "").replace(",", "")
            elements = [float(x) for x in line1.split()]
            tmp_data_1.append(elements)

        with act_file.open(mode="r") as file:
            lines = file.readlines()
        tmp_data_2 = []
        for line in lines[1:]:
            line1 = line.replace("[", "").replace("]", "").replace(",", "")
            elements = [float(x) for x in line1.split()]
            tmp_data_2.append(elements)

        assert np.allclose(tmp_data_1, tmp_data_2, atol=1e-07)

        # Compare "number_of_states.txt"
        exp_file = Path(expected_output_path_2)
        act_file = Path(actual_output_path_2)
        with exp_file.open(mode="r") as file_0, act_file.open(
            mode="r"
        ) as file_1:
            assert file_0.read() == file_1.read()

        # Compare "fraction_0.txt"
        exp_array = np.loadtxt(expected_output_path_3)
        act_array = np.loadtxt(actual_output_path_3)
        assert np.allclose(exp_array, act_array, atol=1e-07)
