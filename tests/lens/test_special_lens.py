import dynsight
import MDAnalysis
import numpy as np
from numpy.testing import assert_array_equal


def test_special_lens(lensfixtures: MDAnalysis.Universe) -> None:
    """Test :class:`.` SOMETHING.

    Parameters:

        hdf5_file:
            A test case.

        input1_2:
            Something.

    """
    expected = lensfixtures[1]
    universe = lensfixtures[0]
    coff = 1.1
    nnlistperframe = dynsight.lens.list_neighbours_along_trajectory(
        input_universe=universe,
        cutoff=coff,
    )
    (
        mynconttot,
        mynntot,
        mynumtot,
        mydentot,
    ) = dynsight.lens.neighbour_change_in_time(nnlistperframe)

    assert_array_equal(mynconttot[:, 0], [0] * mynconttot.shape[0])
    assert_array_equal(mynconttot[:, 1], expected)

    for frame in [0, 1]:
        for atom in universe.atoms:
            atomid = atom.ix
            assert (
                mynntot[atomid, frame]
                == len(nnlistperframe[frame][atomid]) - 1
            )
    for frame in [1]:
        for atom in universe.atoms:
            atomid = atom.ix

            assert (
                mydentot[atomid, frame]
                == len(nnlistperframe[frame][atomid])
                + len(nnlistperframe[frame - 1][atomid])
                - 2
            )
            assert (
                mynumtot[atomid, frame]
                == np.setxor1d(
                    nnlistperframe[frame][atomid],
                    nnlistperframe[frame - 1][atomid],
                ).shape[0]
            )
