from __future__ import annotations

from typing import TYPE_CHECKING

import dynsight
import numpy as np
from numpy.testing import assert_array_almost_equal

if TYPE_CHECKING:
    import pathlib

    import MDAnalysis


def test_emulate_lens(
    hdf5_file: tuple[pathlib.Path, MDAnalysis.Universe],
    input1_2: int,
) -> None:
    """Test :class:`.` SOMETHING.

    Parameters:

        hdf5_file:
            A test case.

        input1_2:
            Something.

    """
    inputuniverse = hdf5_file[1]
    wantedslice = slice(0, len(inputuniverse.trajectory) // input1_2, 1)
    coff = 4.0
    nnlistperframe = dynsight.lens.list_neighbours_along_trajectory(
        input_universe=inputuniverse,
        cutoff=coff,
        trajslice=wantedslice,
    )
    # this is the original version by Martina Crippa
    # def local_dynamics(list_sum):
    particle = list(range(np.shape(nnlistperframe)[1]))
    ncont_tot = []
    nn_tot = []
    num_tot = []
    den_tot = []
    for p in particle:
        ncont = []
        nn = []
        num = []
        den = []
        for frame in range(len(nnlistperframe)):
            if frame == 0:
                ncont.append(0)
                # modifications by Daniele:
                # needed to give the nn counts on the first nn
                nn.append(len(nnlistperframe[frame][p]) - 1)
                # needed to give same lenght the all on the lists
                num.append(0)
                den.append(0)
                # END modification
                # ORIGINAL:nn.append(0)  # noqa: ERA001
            else:  # noqa: PLR5501
                # if the nn set chacne totally set LENS to 1: the nn
                # list contains the atom, hence the  various ==1 and -1

                # se il set di primi vicini cambia totalmente,
                # l'intersezione è lunga 1 ovvero la bead self
                # vale anche se il numero di primi vicini prima e dopo cambia
                if (
                    len(
                        list(
                            set(nnlistperframe[frame - 1][p])
                            & set(nnlistperframe[frame][p])
                        )
                    )
                    == 1
                ):
                    # se non ho NN lens è 0
                    if (
                        len(list(set(nnlistperframe[frame - 1][p]))) == 1
                        and len(set(nnlistperframe[frame][p])) == 1
                    ):
                        ncont.append(0)
                        nn.append(0)
                        num.append(0)
                        den.append(0)
                    # se ho NN lo metto 1
                    else:
                        ncont.append(1)
                        nn.append(len(nnlistperframe[frame][p]) - 1)
                        # changed by daniele
                        # needed to make num/den=1
                        num.append(
                            len(nnlistperframe[frame - 1][p])
                            - 1
                            + len(nnlistperframe[frame][p])
                            - 1
                        )
                        # END modification
                        # ORGINAL: num.append(1)  # noqa: ERA001
                        den.append(
                            len(nnlistperframe[frame - 1][p])
                            - 1
                            + len(nnlistperframe[frame][p])
                            - 1
                        )
                else:
                    # contrario dell'intersezione fra vicini al frame f-1 e al
                    # frame f
                    c_diff = set(
                        nnlistperframe[frame - 1][p]
                    ).symmetric_difference(set(nnlistperframe[frame][p]))
                    ncont.append(  # type:ignore[arg-type]
                        len(c_diff)  # type:ignore[arg-type]
                        / (
                            len(nnlistperframe[frame - 1][p])
                            - 1
                            + len(nnlistperframe[frame][p])
                            - 1
                        )
                    )
                    nn.append(len(nnlistperframe[frame][p]) - 1)
                    num.append(len(c_diff))
                    den.append(
                        len(nnlistperframe[frame - 1][p])
                        - 1
                        + len(nnlistperframe[frame][p])
                        - 1
                    )
        num_tot.append(num)
        den_tot.append(den)
        ncont_tot.append(ncont)
        nn_tot.append(nn)
    # return ncont_tot, nn_tot, num_tot, den_tot  # noqa: ERA001
    (
        mynconttot,
        mynntot,
        mynumtot,
        mydentot,
    ) = dynsight.lens.neighbour_change_in_time(nnlistperframe)

    assert len(mynconttot) == len(ncont_tot)
    assert len(mynntot) == len(nn_tot)
    assert len(mynumtot) == len(num_tot)
    assert len(mydentot) == len(den_tot)

    # lens Value
    for atomdata, wantedatomdata in zip(mynconttot, ncont_tot):
        assert_array_almost_equal(atomdata, wantedatomdata)
    # NN count
    for atomdata, wantedatomdata in zip(mynntot, nn_tot):
        assert_array_almost_equal(atomdata, wantedatomdata)
    # LENS numerator
    for atomdata, wantedatomdata in zip(mynumtot, num_tot):
        assert_array_almost_equal(atomdata, wantedatomdata)
    # LENS denominator
    for atomdata, wantedatomdata in zip(mydentot, den_tot):
        assert_array_almost_equal(atomdata, wantedatomdata)
