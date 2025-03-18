import argparse
import logging
import pathlib

try:
    import h5py
except ImportError:
    h5py = None

import matplotlib.pyplot as plt

import dynsight

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("data_path", type=str, help="path with test files")
    return parser.parse_args()


def main() -> None:
    """Run the example.

    * Original author: Martina Crippa
    * Mantainer: Matteo Becchi

    Example available for cpctools here:
    https://github.com/GMPavanLab/cpctools/blob/main/Examples/LENS.ipynb

    An example for a LENS analysis as in the original paper:
    https://arxiv.org/abs/2212.12694
    """
    # Let's get the data:
    # wget https://github.com/GMPavanLab/dynNP/releases/download/V1.0-trajectories/ico309.hdf5
    # We'll start by caclulating the neighbours and the LENS parameters.
    # using cutoff=2.88*1.1 that is 10% more than the Au radius.
    if h5py is None:
        msg = "Please install SOAPify|h5py with cpctools."
        raise ModuleNotFoundError(msg)

    args = _parse_args()

    data_path = pathlib.Path(args.data_path)

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
    logger = logging.getLogger(__name__)
    logger.info(natoms)

    neigcounts = dynsight.lens.list_neighbours_along_trajectory(
        input_universe=universe,
        cutoff=cutoff,
    )
    lens, nn, *_ = dynsight.lens.neighbour_change_in_time(neigcounts)

    fig, axes = plt.subplots(2, sharey=True)
    for i in range(4):
        axes[0].plot(lens[i], label=f"Atom {i}")
        axes[1].plot(lens[natoms - 1 - i], label=f"Atom {natoms - 1 - i}")

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
