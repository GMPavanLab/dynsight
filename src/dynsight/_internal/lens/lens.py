"""Compute LENS for each atom in a trajectory."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from MDAnalysis import AtomGroup, Universe
    from numpy.typing import NDArray

import numpy as np
from MDAnalysis.lib.NeighborSearch import AtomNeighborSearch


def list_neighbours_along_trajectory(
    input_universe: Universe,
    cutoff: float,
    selection: str = "all",
    trajslice: slice | None = None,
) -> list[list[AtomGroup]]:
    """Produce a per-frame list of the neighbors, atom by atom.

    * Original author: Martina Crippa

    Parameters:
        input_universe:
            The universe, or the atom group containing the trajectory.
        cutoff:
            The maximum neighbor distance.
        selection:
            Selection of atoms taken from the Universe for the computation.
            More information concerning the selection language can be found
            `here <https://userguide.mdanalysis.org/stable/selections.html>`_
        trajslice:
            The slice of the trajectory to consider. Defaults to slice(None).

    Returns:
        list[list[AtomGroup]]:
            List of AtomGroups with the neighbors of each atom for each frame.

    Example:

        .. testsetup:: lens1-test

            import pathlib

            path = pathlib.Path('source/_static/ex_test_files')

        .. testcode:: lens1-test

            import numpy as np
            import MDAnalysis
            from dynsight.lens import list_neighbours_along_trajectory

            univ = MDAnalysis.Universe(path / "trajectory.xyz")
            cutoff = 2.0

            neigh_counts = list_neighbours_along_trajectory(
                input_universe=univ,
                cutoff=cutoff,
            )

        .. testcode:: lens1-test
            :hide:

            assert neigh_counts[0][0][3] == 17

        All supported input file formats by MDAnalysis can be used
        to set up the Universe.
    """
    if trajslice is None:
        trajslice = slice(None)
    neigh_list_per_frame = []
    selected_atoms = input_universe.select_atoms(selection)
    for _ in input_universe.universe.trajectory[trajslice]:
        neigh_search = AtomNeighborSearch(
            input_universe.atoms, box=input_universe.dimensions
        )

        neigh_list_per_atom = [
            neigh_search.search(atom, cutoff) for atom in selected_atoms
        ]
        neigh_list_per_frame.append([at.ix for at in neigh_list_per_atom])
    return neigh_list_per_frame


def neighbour_change_in_time(
    neigh_list_per_frame: list[list[AtomGroup]],
    delay: int = 1,
) -> tuple[
    NDArray[np.float64],
    NDArray[np.int64],
    NDArray[np.float64],
    NDArray[np.float64],
]:
    """Return, listed per atom, the LENS values at each frame.

    * Original author: Martina Crippa

    Parameters:
        neigh_list_per_frame:
            A frame-by-frame list of the neighbors of each atom, output
            of :func:`listNeighboursAlongTrajecøtory`.

        delay:
            The delay between frames on which LENS is computed. Default is 1.

    Returns:
        tuple:
            - lens_array: The calculated LENS parameter.
                It's a numpy.array of shape (n_particles, n_frames - 1)
            - number_of_neighs: The count of neighbors per frame.
                It's a numpy.array of shape (n_particles, n_frames)
            - lens_numerators: The numerators used for calculating LENS.
                It's a numpy.array of shape (n_particles, n_frames - 1)
            - lens_denominators: The denominators used for calculating LENS.
                It's a numpy.array of shape (n_particles, n_frames - 1)

    Example:

        .. testsetup:: lens2-test

            import pathlib

            path = pathlib.Path('source/_static/ex_test_files')

        .. testcode:: lens2-test

            import numpy as np
            import MDAnalysis
            import dynsight.lens as lens

            univ = MDAnalysis.Universe(path / "trajectory.xyz")
            cutoff = 3.0

            neig_counts = lens.list_neighbours_along_trajectory(
                input_universe=univ,
                cutoff=cutoff,
            )

            lens, n_neigh, *_ = lens.neighbour_change_in_time(neig_counts)

        .. testcode:: lens2-test
            :hide:

            assert lens[0][4] == 0.75

        All supported input file formats by MDAnalysis can be used
        to set up the Universe.
    """
    n_atoms = np.asarray(neigh_list_per_frame, dtype=object).shape[1]
    n_frames = np.asarray(neigh_list_per_frame, dtype=object).shape[0]

    lens_array = np.zeros((n_atoms, n_frames))  # The LENS values
    number_of_neighs = np.zeros((n_atoms, n_frames), dtype=int)  # The NN
    lens_numerators = np.zeros((n_atoms, n_frames))  # LENS numerator
    lens_denominators = np.zeros((n_atoms, n_frames))  # LENS denominator

    # each nnlist contains also the atom that generates them,
    # so 0 nn is a 1 element list
    for atom_id in range(n_atoms):
        number_of_neighs[atom_id, 0] = (
            neigh_list_per_frame[0][atom_id].shape[0] - 1
        )
        # let's calculate the numerators and the denominators
        for frame in range(delay, n_frames - delay + 1):
            number_of_neighs[atom_id, frame] = (
                neigh_list_per_frame[frame][atom_id].shape[0] - 1
            )
            lens_denominators[atom_id, frame] = (
                neigh_list_per_frame[frame][atom_id].shape[0]
                + neigh_list_per_frame[frame - delay][atom_id].shape[0]
                - 2
            )
            lens_numerators[atom_id, frame] = np.setxor1d(
                neigh_list_per_frame[frame][atom_id],
                neigh_list_per_frame[frame - delay][atom_id],
            ).shape[0]

    den_not_0 = lens_denominators != 0

    lens_array[den_not_0] = (
        lens_numerators[den_not_0] / lens_denominators[den_not_0]
    )

    return lens_array, number_of_neighs, lens_numerators, lens_denominators
