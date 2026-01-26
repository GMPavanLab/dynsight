"""Pytest for dynsight.Trj.get_coord_number."""

from pathlib import Path

import MDAnalysis
import numpy as np
import pytest

from dynsight.trajectory import Trj

from .case_data import NNCaseData


def test_nn(case_data: NNCaseData) -> None:
    original_dir = Path(__file__).resolve().parent
    topology_file = original_dir / "../systems/balls_7_nvt.gro"
    trajectory_file = original_dir / "../systems/balls_7_nvt.xtc"
    expected_nn = original_dir / "test_nn" / case_data.expected_nn
    universe = MDAnalysis.Universe(topology_file, trajectory_file)

    example_trj = Trj(universe)

    _, test_nn = example_trj.get_coord_number(
        r_cut=case_data.r_cut,
        centers=case_data.centers,
        selection=case_data.selection,
        n_jobs=case_data.n_jobs,
    )

    if not expected_nn.exists():
        np.save(expected_nn, test_nn.dataset)
        pytest.fail("NN test files were not present. They have been created.")
    exp_nn = np.load(expected_nn)
    assert np.allclose(exp_nn, test_nn.dataset, atol=1e-6)
