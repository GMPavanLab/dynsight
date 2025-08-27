from __future__ import annotations

from pathlib import Path

import numpy as np

from dynsight.track import track_xyz
from dynsight.utilities import read_xyz


def test_track_xyz(tmp_path: Path) -> None:
    original_dir = Path(__file__).resolve().parent
    filename = original_dir / "../systems/lj_noid.xyz"
    file_with_id = original_dir / "../systems/lj_id.xyz"
    output = tmp_path / "trajectory.xyz"
    track_xyz(input_xyz=filename, output_xyz=output, search_range=10)
    n_atoms = 5
    for _ in range(n_atoms):
        arr1 = read_xyz(
            input_xyz=output, cols_order=["name", "x", "y", "z", "ID"]
        ).to_numpy()
        arr2 = read_xyz(
            input_xyz=file_with_id, cols_order=["name", "x", "y", "z", "ID"]
        ).to_numpy()
        assert arr1.shape == arr2.shape
        assert np.array_equal(arr1, arr2)
