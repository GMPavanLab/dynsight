"""Pytest for dynsight.trajectory methods."""

from pathlib import Path

import MDAnalysis
import numpy as np

from dynsight.trajectory import ClusterInsight, Insight, Trj


# Define the actual test
def test_output_files() -> None:
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

    # Test dump and load of Insight
    ins_1 = trj_1.get_lens(10.0)
    ins_1.dump_to_json(Path("tests/systems/_tmp.json"))
    _ = Insight.load_from_json(Path("tests/systems/ins_1_test.json"))

    # Test dump and load of ClusterInsight
    fake_labels = np.zeros((5, 5), dtype=int)
    cl_ins = ClusterInsight(fake_labels)
    cl_ins.dump_to_json(Path("tests/systems/_tmp.json"))
    _ = ClusterInsight.load_from_json(Path("tests/systems/cl_ins_test.json"))
