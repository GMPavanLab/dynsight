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
        fastns_results = gridsearch.self_search()
        pairs = fastns_results.get_pairs()
        neigh_list_per_atom = [[] for _ in range(len(input_universe.atoms))]
        for (x,y) in pairs:
            neigh_list_per_atom[x].append(y)
            neigh_list_per_atom[y].append(x)
        neigh_list_per_frame.append(results)
        
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
    lensarray = np.zeros((nat, nframes))
    numberofneighs = np.zeros((nat, nframes))
    lensnumerators = np.zeros((nat, nframes))
    lensdenominators = np.zeros((nat, nframes))
    for atom_id in range(nat):
        #no need to subtract 1 as FastNS does not count the atom itself
        numberofneighs[atom_id, 0] = len(neigh_list_per_frame[0][atom_id])
        for frame in range(1, nframes):
            numberofneighs[atom_id, frame] = len(neigh_list_per_frame[frame][atom_id])
            lensdenominators[atom_id, frame] = len(neigh_list_per_frame[frame][atom_id]) + len(neigh_list_per_frame[frame-1][atom_id])
            lensnumerators[atom_id, frame] = np.setxor1d(
                np.array(neigh_list_per_frame[frame][atom_id]),
                np.array(neigh_list_per_frame[frame - 1][atom_id]),
            ).shape[0]

    den_not_0 = lensdenominators != 0
    # lens
    lensarray[den_not_0] = (
        lensnumerators[den_not_0] / lensdenominators[den_not_0]
    )
    return lensarray, numberofneighs, lensnumerators, lensdenominators
