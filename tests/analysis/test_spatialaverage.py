"""Pytest for dynsight.analysis.stapialaverage."""

import os
import tempfile
from pathlib import Path
from typing import Generator

import numpy as np
import pytest

from dynsight.trajectory import Insight, Trj


@pytest.fixture
def original_wd() -> Generator[Path, None, None]:
    original_dir = Path.cwd()

    # Ensure the original working directory is restored after the test
    yield original_dir

    os.chdir(original_dir)


def test_spatialaverage(original_wd: Path) -> None:
    with tempfile.TemporaryDirectory() as _:
        topology_file = original_wd / "tests/systems/coex/test_coex.gro"
        trajectory_file = original_wd / "tests/systems/coex/test_coex.xtc"
        expected_results = original_wd / "tests/analysis/spavg/test_spavg.npy"

        example_trj = Trj.init_from_xtc(trajectory_file, topology_file)
        descriptor = example_trj.get_coordinates("type O")[:, :, 0].T
        example_data = Insight(descriptor.astype(np.float64))

        aver_data = example_data.spatial_average(
            example_trj,
            r_cut=5.0,
            selection="type O",
            num_processes=1,
        )

        # Load expected results and compare
        expected_arr = np.load(expected_results)
        assert np.allclose(aver_data.dataset, expected_arr)
