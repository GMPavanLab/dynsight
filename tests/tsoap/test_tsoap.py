"""Pytest for dynsight.soap.timesoap."""

import os
from pathlib import Path
from typing import Generator

import MDAnalysis
import numpy as np
import pytest

from dynsight.trajectory import Trj

from .case_data import TimeSOAPCaseData


@pytest.fixture
def original_wd() -> Generator[Path, None, None]:
    original_dir = Path.cwd()

    # Ensure the original working directory is restored after the test
    yield original_dir

    os.chdir(original_dir)


def test_tsoap(case_data: TimeSOAPCaseData) -> None:
    oritinal_dir = Path(__file__).resolve().parent
    topology_file = oritinal_dir / "../systems/balls_7_nvt.gro"
    trajectory_file = oritinal_dir / "../systems/balls_7_nvt.xtc"
    expected_tsoap = oritinal_dir / "test_tsoap" / case_data.expected_tsoap
    universe = MDAnalysis.Universe(topology_file, trajectory_file)

    example_trj = Trj(universe)

    test_tsoap = example_trj.get_timesoap(
        r_cut=case_data.r_c,
        n_max=4,
        l_max=4,
        delay=case_data.delay,
    )

    if not expected_tsoap.exists():
        np.save(expected_tsoap, test_tsoap.dataset)
        pytest.fail(
            "tSOAP test files were not present. They have been created."
        )
    exp_tsoap = np.load(expected_tsoap)
    assert np.allclose(exp_tsoap, test_tsoap.dataset, atol=1e-6)
