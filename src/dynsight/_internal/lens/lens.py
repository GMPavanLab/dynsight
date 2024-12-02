from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from MDAnalysis import AtomGroup, Universe

import numpy as np
from MDAnalysis.lib.NeighborSearch import AtomNeighborSearch


def list_neighbours_along_trajectory(
    input_universe: Universe,
    cutoff: float,
    trajslice: slice | None = None,
) -> list[list[AtomGroup]]:
    """Produce a per-frame list of the neighbors, atom by atom.

    * Original author: Martina Crippa
    * Maintainer: Matteo Becchi

    Parameters:
        input_universe (Universe):
            The universe, or the atom group containing the trajectory.
        cutoff (float):
            The maximum neighbor distance.
        trajslice (slice, optional):
            The slice of the trajectory to consider. Defaults to slice(None).

    Returns:
        list[list[AtomGroup]]:
            List of AtomGroups with the neighbors of each atom for each frame.
    """
    if trajslice is None:
        trajslice = slice(None)
    neigh_list_per_frame = []
    for _ in input_universe.universe.trajectory[trajslice]:
        neigh_search = AtomNeighborSearch(
            input_universe.atoms, box=input_universe.dimensions
        )

        neigh_list_per_atom = [
            neigh_search.search(atom, cutoff) for atom in input_universe.atoms
        ]
        neigh_list_per_frame.append([at.ix for at in neigh_list_per_atom])
    return neigh_list_per_frame


def neighbour_change_in_time(
    neigh_list_per_frame: list[list[AtomGroup]],
) -> tuple[
    np.ndarray[float, Any],
    np.ndarray[int, Any],
    np.ndarray[float, Any],
    np.ndarray[float, Any],
]:
    """Return, listed per atom, the LENS values at each frame.

    * Original author: Martina Crippa
    * Mantainer: Matteo Becchi

    Parameters:
        neigh_list_per_frame:
            A frame-by-frame list of the neighbors of each atom, output
            of :func:`listNeighboursAlongTrajectory`.

    Returns:
        tuple:
            A tuple of the following elements:
                - lensArray: The calculated LENS parameter.
                - numberOfNeighs: The count of neighbors per frame.
                - lensNumerators: The numerators used for calculating LENS.
                - lensDenominators: The denominators used for calculating LENS.
    """
    nat = np.asarray(neigh_list_per_frame, dtype=object).shape[1]
    nframes = np.asarray(neigh_list_per_frame, dtype=object).shape[0]
    # this is the number of common NN between frames
    lensarray = np.zeros((nat, nframes))
    # this is the number of NN at that frame
    numberofneighs = np.zeros((nat, nframes))
    # this is the numerator of LENS
    lensnumerators = np.zeros((nat, nframes))
    # this is the denominator of lens
    lensdenominators = np.zeros((nat, nframes))
    # each nnlist contains also the atom that generates them,
    # so 0 nn is a 1 element list
    for atom_id in range(nat):
        numberofneighs[atom_id, 0] = (
            neigh_list_per_frame[0][atom_id].shape[0] - 1
        )
        # let's calculate the numerators and the denominators
        for frame in range(1, nframes):
            numberofneighs[atom_id, frame] = (
                neigh_list_per_frame[frame][atom_id].shape[0] - 1
            )
            lensdenominators[atom_id, frame] = (
                neigh_list_per_frame[frame][atom_id].shape[0]
                + neigh_list_per_frame[frame - 1][atom_id].shape[0]
                - 2
            )
            lensnumerators[atom_id, frame] = np.setxor1d(
                neigh_list_per_frame[frame][atom_id],
                neigh_list_per_frame[frame - 1][atom_id],
            ).shape[0]

    den_not_0 = lensdenominators != 0
    # lens
    lensarray[den_not_0] = (
        lensnumerators[den_not_0] / lensdenominators[den_not_0]
    )
    return lensarray, numberofneighs, lensnumerators, lensdenominators
