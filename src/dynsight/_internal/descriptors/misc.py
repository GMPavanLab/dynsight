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

            assert np.isclose(psi[0][0], 0.09556616097688675)

    """
    n_atoms = universe.atoms.n_atoms
    n_frames = len(universe.trajectory)

    psi = np.zeros((n_atoms, n_frames))

    for t, _ in enumerate(universe.trajectory):
        frame = universe.atoms.positions[:, :2].copy()

        for i, atom_i in enumerate(frame):
            neighbors = neigh_list_per_frame[t][i]
            if len(neighbors) <= 2:  # noqa: PLR2004
                # if neighbors are none or just 1, psi = 0
                continue
            tmp = 0.0
            for j in neighbors:
                if j != i:
                    x, y = frame[j] - atom_i
                    theta = np.mod(np.arctan2(y, x), 2 * np.pi)
                    tmp += np.exp(1j * order * theta)
            tmp /= len(neighbors) - 1
            psi[i][t] = np.abs(tmp)

    return psi


def _compute_aver_align(
    neigh_list_t: list[AtomGroup],
    frame_vel: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Computes the average alignment for all the atoms in a frame."""
    phi_t = np.zeros(len(frame_vel))
    for i, atom_i in enumerate(frame_vel):
        if not np.any(atom_i):  # skip if zero velocity vector
            continue
        neighbors = neigh_list_t[i]
        if len(neighbors) <= 1:
            continue  # no meaningful averaging if 0 neighbors
        valid_neighbors = [
            j for j in neighbors if j != i and np.any(frame_vel[j])
        ]
        if not valid_neighbors:
            continue  # no self-counting, no neighbors with v = 0.0

        alignments = np.array(
            [1 - cosine(atom_i, frame_vel[j]) for j in valid_neighbors]
        )
        phi_t[i] = np.mean(alignments)
    return phi_t


def velocity_alignment(
    universe: Universe,
    neigh_list_per_frame: list[list[AtomGroup]],
) -> NDArray[np.float64]:
    """Compute average velocity alignment phi.

    If the Universe includes velocities, those are used. Otherwise, the
    displacements are used.

    Parameters:
        universe: contains the coordinates at each frame.

        neigh_list_per_frame: A frame-by-frame list of the neighbors of each
            atom, output of :func:`listNeighboursAlongTrajectory`.

    Returns:
        If the Universe inclused velocities, the output has shape
        (n_atoms, n_frames), otherwise it has shape (n_atoms, n_frames - 1).

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

            assert np.isclose(phi[0][0], 0.38268664479255676)

    """
    n_atoms = universe.atoms.n_atoms
    n_frames = len(universe.trajectory)

    if (
        hasattr(universe.atoms, "velocities")
        and universe.atoms.velocities is not None
    ):  # If the Universe has velocities, use them
        phi = np.zeros((n_frames, n_atoms))
        for t, _ in enumerate(universe.trajectory):
            phi[t] = _compute_aver_align(
                neigh_list_per_frame[t],
                frame_vel=universe.atoms.velocities,
            )
        return phi.T

    # If the Universe does not has velocities, use the displacements
    r_0 = None
    phi = np.zeros((n_frames - 1, n_atoms))
    for t, _ in enumerate(universe.trajectory):
        r_1 = universe.atoms.positions.copy()
        if t == 0:
            r_0 = r_1
            continue
        frame_vel = r_1 - r_0
        phi[t - 1] = _compute_aver_align(
            neigh_list_per_frame[t - 1],
            frame_vel,
        )
    return phi.T
