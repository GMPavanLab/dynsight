"""Pytest for dynsight.onion.main."""

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
    random_walk = []
    for _ in range(5):
        tmp = [0.0]
        for _ in range(1000):
            d_x = rng.normal()
            x_new = tmp[-1] + d_x
            tmp.append(x_new)
        random_walk.append(tmp)
    random_walk_arr = np.array(random_walk)

    with tempfile.TemporaryDirectory() as temp_dir:
        os.chdir(temp_dir)
        data_directory = "input_data_uni.npy"
        np.save(data_directory, random_walk_arr)

        onion_cl = dynsight.onion.OnionUni(
            path_to_input="../" + data_directory,
            tau_w=10,
            num_tau_w=1,
            min_tau_w=10,
            max_tau_w=10,
            max_t_smooth=1,
        )

        onion_cl.run()

        # Define the paths to the expected and actual output files
        results_dir = original_wd / "tests/onion/"
        expected_output_path_1 = results_dir / "output_uni/final_states.txt"
        expected_output_path_2 = (
            results_dir / "output_uni/number_of_states.txt"
        )
        expected_output_path_3 = results_dir / "output_uni/fraction_0.txt"
        actual_output_path_1 = "onion_output/final_states.txt"
        actual_output_path_2 = "onion_output/number_of_states.txt"
        actual_output_path_3 = "onion_output/fraction_0.txt"

        exp_file = Path(expected_output_path_1)
        act_file = Path(actual_output_path_1)
        with exp_file.open(mode="r") as file_0, act_file.open(
            mode="r"
        ) as file_1:
            assert file_0.read() == file_1.read()
        exp_file = Path(expected_output_path_2)
        act_file = Path(actual_output_path_2)
        with exp_file.open(mode="r") as file_0, act_file.open(
            mode="r"
        ) as file_1:
            assert file_0.read() == file_1.read()
        exp_file = Path(expected_output_path_3)
        act_file = Path(actual_output_path_3)
        with exp_file.open(mode="r") as file_0, act_file.open(
            mode="r"
        ) as file_1:
            assert file_0.read() == file_1.read()
