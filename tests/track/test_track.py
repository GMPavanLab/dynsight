from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pytest

from dynsight._internal.track.track import _DEFAULT_NAME
from dynsight.track import track_xyz
from dynsight.trajectory import Trj
from dynsight.utilities import read_xyz

if TYPE_CHECKING:
    from dynsight._internal.utilities.utilities import Col


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


NAMED_COLS = 4


def strip_names(input_xyz: Path, output_xyz: Path) -> None:
    """Write a copy of an .xyz file without its name column."""
    lines = []
    for line in input_xyz.read_text().splitlines():
        parts = line.split()
        if len(parts) == NAMED_COLS:
            parts = parts[1:]
        lines.append(" ".join(parts))
    output_xyz.write_text("\n".join(lines) + "\n")


def test_track_xyz_without_names(tmp_path: Path) -> None:
    # The name column is optional in the input file.
    original_dir = Path(__file__).resolve().parent
    file_with_id = original_dir / "../systems/lj_id.xyz"

    nameless = tmp_path / "nameless.xyz"
    strip_names(original_dir / "../systems/lj_noid.xyz", nameless)

    output = tmp_path / "trajectory.xyz"
    track_xyz(input_xyz=nameless, output_xyz=output, search_range=10)

    cols_order: list[Col] = ["name", "x", "y", "z", "ID"]
    tracked = read_xyz(input_xyz=output, cols_order=cols_order)
    expected = read_xyz(input_xyz=file_with_id, cols_order=cols_order)

    # Positions and IDs match the run on the file with names, and the
    # output is a valid .xyz: the missing names get a placeholder.
    compared = ["frame", "x", "y", "z", "ID"]
    assert tracked.shape == expected.shape
    assert np.array_equal(
        tracked[compared].to_numpy(), expected[compared].to_numpy()
    )
    assert set(tracked["name"]) == {_DEFAULT_NAME}


def test_track_xyz_invalid_format(tmp_path: Path) -> None:
    invalid = tmp_path / "invalid.xyz"
    invalid.write_text("2\ncomment\n1.0 2.0\n3.0 4.0\n")
    with pytest.raises(ValueError, match=r"Error in the \.xyz format"):
        track_xyz(
            input_xyz=invalid,
            output_xyz=tmp_path / "out.xyz",
            search_range=10,
        )


def test_track_xyz_only_writes_the_tracked_file(tmp_path: Path) -> None:
    """No Trj is built: a tracked file is not necessarily a trajectory."""
    original_dir = Path(__file__).resolve().parent
    output = tmp_path / "tracked.xyz"
    track_xyz(
        input_xyz=original_dir / "../systems/lj_noid.xyz",
        output_xyz=output,
        search_range=10,
    )
    # A clean file can still be turned into a Trj by the caller.
    n_particles = 5
    trj = Trj.init_from_xyz(traj_file=output, dt=1)
    assert trj.n_atoms == n_particles


def test_track_xyz_writes_a_ragged_file(tmp_path: Path) -> None:
    """A variable object count is written out as it is, without raising."""
    ragged = tmp_path / "ragged.xyz"
    ragged.write_text(
        "3\nf0\n1.0 1.0 0.0\n5.0 1.0 0.0\n9.0 1.0 0.0\n"
        "2\nf1\n1.2 1.0 0.0\n5.2 1.0 0.0\n"
        "3\nf2\n1.4 1.0 0.0\n5.4 1.0 0.0\n9.4 1.0 0.0\n"
    )
    output = tmp_path / "tracked.xyz"

    track_xyz(
        input_xyz=ragged,
        output_xyz=output,
        search_range=3,
        memory=0,
    )
    # The file is written: it is a faithful record of the detections.
    assert output.exists()
