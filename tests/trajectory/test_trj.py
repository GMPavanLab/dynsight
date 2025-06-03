"""Pytest for dynsight.trajectory methods."""

from pathlib import Path

import MDAnalysis
import numpy as np
import pytest

from dynsight.trajectory import (
    ClusterInsight,
    Insight,
    OnionInsight,
    OnionSmoothInsight,
    Trj,
)


# Define the actual test
def test_output_files(tmp_path: Path) -> None:
    here = Path(__file__).parent
    input_file_xyz = Path(here / "../systems/2_particles.xyz")
    input_file_gro = Path(here / "../systems/balls_7_nvt.gro")
    input_file_xtc = Path(here / "../systems/balls_7_nvt.xtc")
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

    files_path = Path(here / "files")
    # Test dump and load of Insight
    ins_1 = trj_1.get_lens(10.0)
    ins_1.dump_to_json(tmp_path / "_tmp.json")
    _ = Insight.load_from_json(files_path / "ins_1_test.json")

    # Test dump and load of ClusterInsight
    fake_labels = np.zeros((5, 5), dtype=int)
    cl_ins = ClusterInsight(fake_labels)
    cl_ins.dump_to_json(tmp_path / "_tmp.json")
    _ = ClusterInsight.load_from_json(files_path / "cl_ins_test.json")

    # Test dump and load of OnionInsight
    on_ins = ins_1.get_onion(delta_t=5)
    on_ins.dump_to_json(tmp_path / "_tmp.json")
    _ = OnionInsight.load_from_json(files_path / "on_ins_test.json")
    (tmp_path / "_tmp.json").unlink()
    on_smooth_ins = ins_1.get_onion_smooth(delta_t=5)
    on_smooth_ins.dump_to_json(tmp_path / "_tmp.json")
    _ = OnionSmoothInsight.load_from_json(tmp_path / "_tmp.json")

    # Test onion analysis
    _, _, _ = ins_1.get_onion_analysis()

    with pytest.raises(
        ValueError, match="'dataset' key not found in JSON file."
    ):
        _ = Insight.load_from_json(files_path / "empty.json")

    with pytest.raises(
        ValueError, match="'labels' key not found in JSON file."
    ):
        _ = ClusterInsight.load_from_json(files_path / "ins_1_test.json")

    with pytest.raises(
        ValueError, match="'state_list' key not found in JSON file."
    ):
        _ = OnionInsight.load_from_json(files_path / "cl_ins_test.json")
