"""Miscellaneous descriptors."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from MDAnalysis import AtomGroup, Universe
    from numpy.typing import NDArray

import numpy as np
from scipy.spatial.distance import cosine


def orientational_order_param(
    universe: Universe,
    neigh_list_per_frame: list[list[AtomGroup]],
    order: int = 6,
) -> NDArray[np.float64]:
    r"""Compute the magnitude of the orientational order parameter.

    .. math::

        | \psi^{(n)}_i | = \frac{1}{N_i} \sum_{j=1}^{N_{i}} e^{i n \theta_{ij}}

    where n is the symmetry order.

    .. warning::

        Particles are considered as laying in the (x, y) plane. z-coordinates
        are ignored.

    Parameters:
        universe: contains the coordinates at each frame.

        neigh_list_per_frame: A frame-by-frame list of the neighbors of each
            atom, output of :func:`listNeighboursAlongTrajectory`.

        order: the order of the symmetry measured by the descriptor. Default
            is 6, corresponding to the hexatic order parameter.

    Returns:
        NDArray[np.float64]:
            An array of shape (n_atoms, n_frames), with the values of psi.

    Example:

        .. testsetup:: psi-test

            import pathlib

            path = pathlib.Path('source/_static/ex_test_files')

        .. testcode:: psi-test

            import numpy as np
            import MDAnalysis
            from dynsight.lens import list_neighbours_along_trajectory
            from dynsight.descriptors import orientational_order_param

            univ = MDAnalysis.Universe(path / "trajectory.xyz")

            cutoff = 3.0
            neig_counts = list_neighbours_along_trajectory(
                input_universe=univ,
                cutoff=cutoff,
            )

            psi = orientational_order_param(
                universe=univ,
                neigh_list_per_frame=neig_counts,
            )

        .. testcode:: psi-test
            :hide:

            assert np.isclose(psi[0][0], 0.095872301262402)
    """
    n_frames = len(universe.trajectory)
    n_atoms = universe.atoms.n_atoms

    coords = np.empty((n_frames, n_atoms, 2), dtype=float)
    for t, _ in enumerate(universe.trajectory):
        coords[t] = universe.atoms.positions[..., :2]

    psi = np.zeros((n_atoms, n_frames))

    for t, frame in enumerate(coords):
        for i, atom_i in enumerate(frame):
            tmp = 0.0
            for j in neigh_list_per_frame[t][i]:
                x, y = frame[j] - atom_i
                if x != 0:
                    theta = np.arctan(y / x)
                    if x < 0.0:
                        theta += np.pi
                else:
                    theta = np.pi / 2
                tmp += np.exp(1j * order * theta)
            if len(neigh_list_per_frame) > 0:
                tmp /= len(neigh_list_per_frame)
            psi[i][t] = np.abs(tmp)

    return psi


def velocity_alignment(
    universe: Universe,
    neigh_list_per_frame: list[list[AtomGroup]],
    velocities: NDArray[np.float64] | None = None,
) -> NDArray[np.float64]:
    """Compute average velocity alignment phi.

    Parameters:
        universe: contains the coordinates at each frame.

        neigh_list_per_frame: A frame-by-frame list of the neighbors of each
            atom, output of :func:`listNeighboursAlongTrajectory`.

        velocities: shape (n_frames - 1, n_atoms, n_dims), the particles'
            velocities. If is None, velocities are computed as dispacements.

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
