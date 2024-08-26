from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from MDAnalysis import AtomGroup, Universe

import numpy as np
from MDAnalysis.lib.nsgrid import FastNS


def list_neighbours_along_trajectory(
    input_universe: Universe,
    cutoff: float,
    trajslice: slice | None = None,
) -> list[list[AtomGroup]]:
    """Produce a per frame list of the neighbours, atom per atom.

    * Original author: Martina Crippa
    * Mantainer: Daniele Rapetti

    Parameters:
        input_universe (Universe):
            the universe, or the atomgroup containing the trajectory.
        cutoff (float):
            the maximum neighbour distance.
        trajslice (slice, optional):
            the slice of the trajectory to consider. Defaults to slice(None).

    Returns:
        list[list[AtomGroup]]:
            list of AtomGroup wint the neighbours of each atom for each frame
    """
    if trajslice is None:
        trajslice = slice(None)
    neigh_list_per_frame = []

    for _ in input_universe.universe.trajectory[trajslice]:
        atom_pos = input_universe.atoms.positions
        box_dim = input_universe.dimensions
        gridsearch = FastNS(cutoff, atom_pos, box=box_dim, pbc=True)
        max_cutoff = gridsearch._prepare_box(box=box_dim, pbc=True)
        if cutoff > max_cutoff:
            cutoff = max_cutoff
        fastns_results = gridsearch.self_search()
        pairs = fastns_results.get_pairs()
        neigh_list_per_atom = [[] for _ in range(len(input_universe.atoms))]
        for x, y in pairs:
            neigh_list_per_atom[x].append(y)
            neigh_list_per_atom[y].append(x)
        neigh_list_per_frame.append(neigh_list_per_atom)

    return neigh_list_per_frame


def neighbour_change_in_time(
    neigh_list_per_frame: list[list[AtomGroup]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:  # type: ignore[type-arg]
    """return, listed per each atoms the parameters used in the LENS analysis.

    * Original author: Martina Crippa
    * Mantainer: Daniele Rapetti

    Parameters:
        neigh_list_per_frame:
             a frame by frame list of the neighbours of each atom output
             of :func:`listNeighboursAlongTrajectory`.

    Returns:
        - **lensArray** The calculated LENS parameter
        - **numberOfNeighs** the count of neighbours per frame
        - **lensNumerators** the numerators used for calculating LENS
            parameter
        - **lensDenominators** the denominators used for calculating LENS
            parameter
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
