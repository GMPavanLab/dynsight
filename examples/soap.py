from __future__ import annotations

import argparse
import logging
import pathlib

import matplotlib.pyplot as plt
import MDAnalysis

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

    Paper: https://doi.org/10.1063/5.0147025

    """
    # trajectory (it is a 55 gold atoms nanoparticle NVT at few temperatures):
    # wget https://github.com/GMPavanLab/SOAPify/releases/download/0.1.0rc0/SmallExample.zip
    # We'll start by caclulating the SOAP fingerprints of the simulation
    # using lMax=8, nMax=8, and cutoff=4.48023312 that is 10% more than the Au
    # cell

    args = _parse_args()

    data_path = pathlib.Path(args.data_path)

    universe = MDAnalysis.Universe(
        data_path / "ih55.data",
        [data_path / "ih55-T_100.lammpsdump"],
        atom_style="id type x y z",
    )
    universe.atoms.types = ["Au"] * len(universe.atoms)

    soap = dynsight.soap.saponify_trajectory(
        universe,
        soaprcut=4.48023312,
        soapnmax=8,
        soaplmax=8,
        n_core=4,
    )

    tsoap = dynsight.soap.timesoap(soap)

    natoms = tsoap.shape[0]
    fig, axes = plt.subplots(2, sharey=True)
    for i in range(4):
        axes[0].plot(tsoap[i], label=f"Atom {i}")
        axes[1].plot(tsoap[-1 - i], label=f"Atom {natoms - 1 - i}")

    for ax in axes:
        ax.legend()
    fig.tight_layout()
    fig.savefig(
        "tsoap.png",
        dpi=360,
        bbox_inches="tight",
    )
    plt.show()


if __name__ == "__main__":
    main()
