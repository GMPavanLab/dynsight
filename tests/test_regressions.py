"""Regression tests for three bugs found while writing the tutorials.

1. Descriptors built on a neighbor list ignored the ``Trj`` slice and
   walked off the end of it (``IndexError``).
2. ``spatialaverage`` indexed the descriptor with the trajectory's frame
   count, so a descriptor defined on frame pairs (LENS, timeSOAP) raised a
   bare ``IndexError`` from inside a worker process.
3. ``track_xyz`` returned a ``Trj`` built on a file whose frames may hold
   different numbers of objects, which raises ``EOFError`` as soon as any
   descriptor is computed on it. It now only writes the tracked file.
   Planar tracked data also made ``compute_lens`` infer a zero-thickness
   box and divide by zero.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from dynsight.analysis import spatialaverage
from dynsight.track import track_xyz
from dynsight.trajectory import Trj

SYSTEMS = Path(__file__).resolve().parent / "systems"
_LJ_N_PARTICLES = 5
TRJ_2D = (
    Path(__file__).resolve().parent.parent
    / "docs/source/_static/ex_test_files/trajectory.xyz"
)


def _trj_2d() -> Trj:
    return Trj.init_from_xyz(traj_file=TRJ_2D, dt=1.0)


def test_orientational_op_on_a_sliced_trj() -> None:
    """The descriptor must follow the slice, not the whole trajectory."""
    trj = _trj_2d()
    _, psi_full = trj.get_orientational_op(r_cut=3.0, order=6)

    n_frames = 10
    sliced = trj.with_slice(slice(0, n_frames, 1))
    neigcounts, _ = sliced.get_coord_number(r_cut=3.0)
    _, psi = sliced.get_orientational_op(
        r_cut=3.0, order=6, neigcounts=neigcounts
    )

    assert psi.dataset.shape == (trj.n_atoms, n_frames)
    assert np.allclose(psi.dataset, psi_full.dataset[:, :n_frames])


def test_velocity_alignment_on_a_sliced_trj() -> None:
    trj = _trj_2d()
    n_frames = 10
    sliced = trj.with_slice(slice(0, n_frames, 1))
    neigcounts, _ = sliced.get_coord_number(r_cut=3.0)
    _, phi = sliced.get_velocity_alignment(r_cut=3.0, neigcounts=neigcounts)

    # No velocities in an .xyz: displacements are used, hence n_frames - 1.
    assert phi.dataset.shape == (trj.n_atoms, n_frames - 1)


def test_mismatched_neighbour_list_raises_clearly() -> None:
    """A neighbor list from another slice must fail loudly, not silently."""
    trj = _trj_2d()
    neigcounts, _ = trj.get_coord_number(r_cut=3.0)
    sliced = trj.with_slice(slice(0, 10, 1))

    with pytest.raises(ValueError, match="neigh_list_per_frame covers"):
        sliced.get_orientational_op(r_cut=3.0, order=6, neigcounts=neigcounts)


def test_spatial_average_frame_mismatch_raises_clearly() -> None:
    """LENS is one frame shorter than its trajectory: say so, don't crash."""
    trj = _trj_2d()
    descriptor = np.zeros((trj.n_atoms, trj.n_frames - 1))

    with pytest.raises(ValueError, match="Descriptor covers"):
        spatialaverage(
            universe=trj.universe,
            descriptor_array=descriptor,
            selection="all",
            r_cut=3.0,
        )


def test_lens_on_planar_data_without_a_box(tmp_path: Path) -> None:
    """A flat system has zero extent along z; the box must still be usable.

    This is exactly the shape of the data ``dynsight.vision`` produces:
    pixel coordinates in a plane, with no simulation box.
    """
    rng = np.random.default_rng(42)
    n_atoms, n_frames = 30, 6
    xy = rng.uniform(0.0, 40.0, size=(n_frames, n_atoms, 2))

    planar = tmp_path / "planar.xyz"
    with planar.open("w") as file:
        for frame in range(n_frames):
            file.write(f"{n_atoms}\nFrame {frame}\n")
            for x, y in xy[frame]:
                file.write(f"P {x:.5f} {y:.5f} 0.00000\n")

    trj = Trj.init_from_xyz(traj_file=planar, dt=1.0)
    assert trj.universe.trajectory[0].dimensions is None
    assert np.allclose(trj.get_coordinates("all")[:, :, 2], 0.0)

    lens = trj.get_lens(r_cut=10.0)

    assert lens.dataset.shape == (n_atoms, n_frames - 1)
    assert np.all(np.isfinite(lens.dataset))


def test_track_xyz_only_writes_the_tracked_file(tmp_path: Path) -> None:
    """No Trj is built: a tracked file is not necessarily a trajectory."""
    output = tmp_path / "tracked.xyz"
    track_xyz(
        input_xyz=SYSTEMS / "lj_noid.xyz",
        output_xyz=output,
        search_range=10,
    )
    # A clean file can still be turned into a Trj by the caller.
    trj = Trj.init_from_xyz(traj_file=output, dt=1)
    assert trj.n_atoms == _LJ_N_PARTICLES


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
