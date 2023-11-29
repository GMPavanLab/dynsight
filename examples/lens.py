import logging
import pathlib
import sys

import dynsight
import h5py
import matplotlib.pyplot as plt

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)


def main() -> None:
    """Run the example.

    Example available for cpctools here:
    https://github.com/GMPavanLab/cpctools/blob/main/Examples/LENS.ipynb

    An example for a LENS analysis as in the original paper:
    https://arxiv.org/abs/2212.12694
    """
    # Let's get the data:
    # wget https://github.com/GMPavanLab/dynNP/releases/download/V1.0-trajectories/ico309.hdf5
    # We'll start by caclulating the neighbours and the LENS parameters.
    # using cutoff=2.88*1.1 that is 10% more than the Au radius.
    first_line = f"Usage: {__file__}.py data_directory"
    num_args = 2
    if len(sys.argv) != num_args:
        logging.info(first_line)
        sys.exit()
    else:
        data_directory = sys.argv[1]

    data_path = pathlib.Path(data_directory)

    trajfilename = data_path / "ico309.hdf5"
    cutoff = 2.88 * 1.1

    wantedtrajectory = slice(0, None, 10)
    trajaddress = "/Trajectories/ico309-SV_18631-SL_31922-T_500"
    with h5py.File(trajfilename, "r") as trajfile:
        tgroup = trajfile[trajaddress]
        universe = dynsight.hdf5er.create_universe_from_slice(
            tgroup, wantedtrajectory
        )

    natoms = len(universe.atoms)
    logging.info(natoms)

    neigcounts = dynsight.lens.list_neighbours_along_trajectory(
        input_universe=universe,
        cutoff=cutoff,
    )
    lens, nn, *_ = dynsight.lens.neighbour_change_in_time(neigcounts)

    fig, axes = plt.subplots(2, sharey=True)
    for i in range(4):
        axes[0].plot(lens[i], label=f"Atom {i}")
        axes[1].plot(lens[natoms - 1 - i], label=f"Atom {natoms-1-i}")

    for ax in axes:
        ax.legend()
    fig.tight_layout()
    fig.savefig(
        "lens.png",
        dpi=360,
        bbox_inches="tight",
    )
    plt.close()


if __name__ == "__main__":
    main()
