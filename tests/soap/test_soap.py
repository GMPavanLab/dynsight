"""Pytest for dynsight.soap.saponify_trajectory."""

from pathlib import Path

import MDAnalysis
import numpy as np
import pytest

from dynsight.trajectory import Trj

from .case_data import SOAPCaseData


def test_soap(case_data: SOAPCaseData) -> None:
    original_dir = Path(__file__).resolve().parent
    topology_file = original_dir / "../systems/balls_7_nvt.gro"
    trajectory_file = original_dir / "../systems/balls_7_nvt.xtc"
    expected_soap = original_dir / "test_soap" / case_data.expected_soap
    universe = MDAnalysis.Universe(topology_file, trajectory_file)

    example_trj = Trj(universe)

    test_soap = example_trj.get_soap(
        r_cut=case_data.r_c,
        n_max=case_data.n_max,
        l_max=case_data.l_max,
        respect_pbc=case_data.respect_pbc,
        centers=case_data.centers,
    )

    if not expected_soap.exists():
        np.save(expected_soap, test_soap.dataset)
        pytest.fail(
            "SOAP test files were not present. They have been created."
        )
    exp_soap = np.load(expected_soap)
    assert np.allclose(exp_soap, test_soap.dataset)
