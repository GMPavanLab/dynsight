import os
from pathlib import Path
from typing import Generator

import MDAnalysis
import numpy as np
import pytest

from dynsight.trajectory import Insight, Trj


@pytest.fixture
def original_wd() -> Generator[Path, None, None]:
    original_dir = Path.cwd()

    # Ensure the original working directory is restored after the test
    yield original_dir

    os.chdir(original_dir)


def test_spatialaverage() -> None:
    original_dir = Path(__file__).resolve().parent
    topology_file = original_dir / "../systems/coex/test_coex.gro"
    trajectory_file = original_dir / "../systems/coex/test_coex.xtc"
    expected_results = original_dir / "spavg/test_spavg.npy"

    universe = MDAnalysis.Universe(topology_file, trajectory_file)
    example_trj = Trj(universe)
    example_trj = Trj.init_from_xtc(trajectory_file, topology_file)
    atoms = universe.select_atoms("type O")

    descriptor = np.zeros((2048, 6))
    for ts in universe.trajectory:
        descriptor[:, ts.frame] = atoms.positions[:, 0]

    example_data = Insight(descriptor)

    aver_data = example_data.spatial_average(
        example_trj,
        r_cut=5.0,
        selection="type O",
        num_processes=1,
    )

    # Load expected results and compare
    expected_arr = np.load(expected_results)
    assert np.allclose(aver_data.dataset, expected_arr)
