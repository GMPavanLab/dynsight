"""Pytest for dynsight.trajectory.Trj.init methods."""

import os
import tempfile
from pathlib import Path
from typing import Generator

import MDAnalysis
import pytest

from dynsight.trajectory import Trj


@pytest.fixture
def original_wd() -> Generator[Path, None, None]:
    original_dir = Path.cwd()

    # Ensure the original working directory is restored after the test
    yield original_dir

    os.chdir(original_dir)


# Define the actual test
def test_output_files() -> None:
    with tempfile.TemporaryDirectory() as _:
        input_file_xyz = Path("tests/systems/2_particles.xyz")
        input_file_gro = Path("tests/systems/balls_7_nvt.gro")
        input_file_xtc = Path("tests/systems/balls_7_nvt.xtc")
        universe = MDAnalysis.Universe(input_file_xyz, dt=1)
        n_frames_xyz = 21
        n_frames_xtc = 201

        # Test default init
        trj_1 = Trj(universe)
        assert len(trj_1.universe.trajectory) == n_frames_xyz

        # Test init_from_universe
        trj_2 = Trj.init_from_universe(universe)
        assert len(trj_2.universe.trajectory) == n_frames_xyz

        # Test init_from_xyz
        trj_3 = Trj.init_from_xyz(input_file_xyz, dt=1)
        assert len(trj_3.universe.trajectory) == n_frames_xyz

        # Test init_from_xtc
        trj_4 = Trj.init_from_xtc(input_file_xtc, input_file_gro)
        assert len(trj_4.universe.trajectory) == n_frames_xtc
