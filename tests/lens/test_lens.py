"""Pytest for dynsight.lens.compute_lens."""

from pathlib import Path

import MDAnalysis
import numpy as np
import pytest

from dynsight.trajectory import Trj

from .case_data import LENSCaseData


def test_lens(case_data: LENSCaseData) -> None:
    original_dir = Path(__file__).resolve().parent
    topology_file = original_dir / "../systems/balls_7_nvt.gro"
    trajectory_file = original_dir / "../systems/balls_7_nvt.xtc"
    expected_lens = original_dir / "test_lens" / case_data.expected_lens
    universe = MDAnalysis.Universe(topology_file, trajectory_file)

    example_trj = Trj(universe)

    test_lens = example_trj.get_lens(
        r_cut=case_data.r_cut,
        delay=case_data.delay,
        centers=case_data.centers,
        selection=case_data.selection,
        n_jobs=case_data.n_jobs,
    )

    if not expected_lens.exists():
        np.save(expected_lens, test_lens.dataset)
        pytest.fail(
            "LENS test files were not present. They have been created."
        )
    exp_lens = np.load(expected_lens)
    assert np.allclose(exp_lens, test_lens.dataset, atol=1e-6)
