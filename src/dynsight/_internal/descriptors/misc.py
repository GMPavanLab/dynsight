"""Miscellaneous descriptors."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from MDAnalysis import AtomGroup, Universe
    from numpy.typing import NDArray

import numpy as np
from scipy.spatial.distance import cosine


def velocity_alignment(
    universe: Universe,
    neigh_list_per_frame: list[list[AtomGroup]],
    velocities: NDArray[np.float64] | None = None,
) -> NDArray[np.float64]:
    """Compute average velocity alignment phi.

    Parameters:
        neigh_list_per_frame: A frame-by-frame list of the neighbors of each
            atom, output of :func:`listNeighboursAlongTrajectory`.

        coords: shape (n_frames, n_atoms, n_dims), the particles' traj.

        velocities: shape (n_frames - 1, n_atoms, n_dims), the particles'
            velocities. If not passed, velocities are computed as dispacements.

    Returns:
        NDArray[np.float64]:
            An array of shape (n_atoms, n_frames - 1), with the values of phi.

    Example:

        .. testsetup:: phi-test

            import pathlib

            path = pathlib.Path('source/_static/ex_test_files')

        .. testcode:: phi-test

            import numpy as np
            import MDAnalysis
            from dynsight.lens import list_neighbours_along_trajectory
            from dynsight.descriptors import velocity_alignment

            univ = MDAnalysis.Universe(path / "trajectory.xyz")
            cutoff = 3.0

            neig_counts = list_neighbours_along_trajectory(
                input_universe=univ,
                cutoff=cutoff,
            )

            phi = velocity_alignment(
                universe=univ,
                neigh_list_per_frame=neig_counts,
            )

        .. testcode:: phi-test
            :hide:

            assert np.isclose(phi[0][1], 0.15532779089361093)
    """
    n_frames = len(universe.trajectory)
    n_atoms = universe.atoms.n_atoms

    coords = np.empty((n_frames, n_atoms, 3), dtype=float)
    for t, _ in enumerate(universe.trajectory):
        coords[t] = universe.atoms.positions

    phi = np.zeros((n_atoms, n_frames - 1))

    vel = coords[1:,] - coords[:-1,] if velocities is None else velocities

    for t, frame in enumerate(vel):
        for i, atom_i in enumerate(frame):
            tmp = 0.0
            for j in neigh_list_per_frame[t][i]:
                tmp += 1 - cosine(atom_i, frame[j])
            if len(neigh_list_per_frame) > 0:
                tmp /= len(neigh_list_per_frame)
            phi[i][t] = tmp

    return phi
