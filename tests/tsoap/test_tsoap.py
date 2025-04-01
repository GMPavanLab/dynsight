"""Pytest for dynsight.soap.timesoap."""

from pathlib import Path

import MDAnalysis
import numpy as np
import pytest

from dynsight.soap import saponify_trajectory, timesoap
from tests.tsoap.case_data import TimeSOAPCaseData


def test_tsoap(case_data: TimeSOAPCaseData) -> None:
    oritinal_dir = Path(__file__).resolve().parent
    topology_file = oritinal_dir / "../systems/balls_7_nvt.gro"
    trajectory_file = oritinal_dir / "../systems/balls_7_nvt.xtc"
    expected_tsoap = oritinal_dir / "test_tsoap" / case_data.expected_tsoap
    u = MDAnalysis.Universe(topology_file, trajectory_file)
    soap_traj = saponify_trajectory(
        universe=u,
        soaprcut=case_data.r_c,
        soaplmax=4,
        soapnmax=4,
        soap_respectpbc=True,
        centers="all",
    )
    test_tsoap = timesoap(soaptrajectory=soap_traj, delay=case_data.delay)
    if not expected_tsoap.exists():
        np.save(expected_tsoap, test_tsoap)
        pytest.fail(
            "tSOAP test files were not present. They have been created."
        )
    exp_tsoap = np.load(expected_tsoap)
    assert np.allclose(exp_tsoap, test_tsoap, atol=1e-6)
