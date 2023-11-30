from __future__ import annotations

from typing import TYPE_CHECKING

import dynsight
import MDAnalysis
import numpy as np
from numpy.testing import assert_array_equal

from .utilities import is_sorted

if TYPE_CHECKING:
    import pathlib


def test_count_neigh_for_lens(
    hdf5_file: tuple[pathlib.Path, MDAnalysis.Universe], input1_2: int
) -> None:
    """Test :class:`.list_neighbours_along_trajectory`.

    Parameters:

        hdf5_file:
            A test case.

        input1_2:
            Something.

    """
    # this is the original version by Martina Crippa

    inputuniverse: MDAnalysis.Universe = hdf5_file[1]
    wantedslice = slice(0, len(inputuniverse.trajectory) // input1_2, 1)
    coff = 10.0
    init = wantedslice.start
    end = wantedslice.stop
    stride = wantedslice.step

    beads = inputuniverse.select_atoms("all")
    cont_list = []
    # loop over traj
    for _, _ in enumerate(inputuniverse.trajectory[init:end:stride]):
        nsearch = MDAnalysis.lib.NeighborSearch.AtomNeighborSearch(
            beads,
            box=inputuniverse.dimensions,
        )
        cont_list.append([nsearch.search(i, coff, level="A") for i in beads])
    for selection in [inputuniverse, beads]:
        neigh_list_per_frame = dynsight.lens.list_neighbours_along_trajectory(
            input_universe=selection,
            cutoff=coff,
            trajslice=wantedslice,
        )

        assert len(neigh_list_per_frame) == len(cont_list)
        for nnlistorig, mynnlist in zip(cont_list, neigh_list_per_frame):
            assert len(nnlistorig) == len(mynnlist)
            for atomgroupnn, myatomsid in zip(nnlistorig, mynnlist):
                atomsid = np.sort([at.ix for at in atomgroupnn])
                assert is_sorted(myatomsid)
                assert_array_equal(atomsid, myatomsid)
