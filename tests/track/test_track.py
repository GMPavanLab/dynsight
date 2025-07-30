from __future__ import annotations

from pathlib import Path

import numpy as np

from dynsight.track import track_xyz
from dynsight.trajectory import Trj


def test_track_xyz(tmp_path: Path) -> None:
    original_dir = Path(__file__).resolve().parent
    filename = original_dir / "../systems/lj_noid.xyz"
    file_with_id = original_dir / "../systems/lj_id.xyz"
    output = tmp_path / "trajectory.xyz"

    ref_trajectory = Trj.init_from_xyz(traj_file=file_with_id, dt=1)

    trajectory = track_xyz(input_xyz=filename, output_xyz=output)

    ref_pos = ref_trajectory.get_coordinates(selection="all")
    tst_pos = trajectory.get_coordinates(selection="all")

    r_sorted = np.array([frame[np.argsort(frame[:, 0])] for frame in ref_pos])
    t_sorted = np.array([frame[np.argsort(frame[:, 0])] for frame in tst_pos])

    assert r_sorted.shape == t_sorted.shape
    assert np.allclose(r_sorted[:, :, :], t_sorted[:, :, :])
