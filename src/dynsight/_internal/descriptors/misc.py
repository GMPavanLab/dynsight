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
        An array of shape (n_atoms, n_frames), with the values of psi.

    Example:

        .. testsetup:: psi-test

            import pathlib

            path = pathlib.Path('source/_static/ex_test_files')

        .. testcode:: psi-test

            import numpy as np
            from dynsight.trajectory import Trj
            from dynsight.descriptors import orientational_order_param

            trj = Trj.init_from_xyz(path / "trajectory.xyz", dt=1.0)
            neig_counts, _ = trj.get_coord_number(r_cut=3.0)

            psi = orientational_order_param(
                universe=trj.universe,
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
        An array of shape (n_atoms, n_frames - 1), with the values of phi.

    Example:

        .. testsetup:: phi-test

            import pathlib

            path = pathlib.Path('source/_static/ex_test_files')

        .. testcode:: phi-test

            import numpy as np
            from dynsight.trajectory import Trj
            from dynsight.descriptors import velocity_alignment

            trj = Trj.init_from_xyz(path / "trajectory.xyz", dt=1.0)
            neig_counts, _ = trj.get_coord_number(r_cut=3.0)

            phi = velocity_alignment(
                universe=trj.universe,
                neigh_list_per_frame=neig_counts,
            )

        .. testcode:: phi-test
            :hide:

            assert np.isclose(phi[0][1], 0.4948548674583435)

    """
    n_atoms = universe.atoms.n_atoms
    n_frames = len(universe.trajectory)

    phi = np.zeros((n_atoms, n_frames - 1))

    r_0 = None

    for t, _ in enumerate(universe.trajectory):
        r_1 = universe.atoms.positions.copy()

        if t == 0:
            r_0 = r_1
            continue

        frame_vel = (r_1 - r_0) if velocities is None else velocities[t - 1]

        for i, atom_i in enumerate(frame_vel):
            tmp = 0.0
            neighbors = neigh_list_per_frame[t - 1][i]
            for j in neighbors:
                tmp += 1 - cosine(atom_i, frame_vel[j])
            if len(neighbors) > 0:
                tmp /= len(neighbors)
            phi[i, t - 1] = tmp

    return phi
