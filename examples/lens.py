"""An example for the lens module.

Details on LENS and additional examples can be found in the paper:
M. Crippa, A. Cardellini, C. Caruso,  & G.M. Pavan,
Detecting dynamic domains and local fluctuations in complex molecular
systems via timelapse neighbors shuffling, PNAS 120 (30) e2300565120,
https://doi.org/10.1073/pnas.2300565120 (2023).
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from MDAnalysis import Universe

from dynsight.lens import compute_lens, list_neighbours_along_trajectory


def example_for_lens(universe: Universe, r_cut: float) -> None:
    """Compute LENS on a trajectory."""
    lens = compute_lens(
        universe=universe,
        r_cut=r_cut,
        delay=1,
        centers="all",
        selection="all",
        respect_pbc=True,
        n_jobs=4,
    )

    # Plot the LENS time-series and the cumulative distribution
    fig, ax = plt.subplots(1, 2, sharey=True)
    fig.suptitle(f"LENS computed with cutoff radius {r_cut} Å")
    for particle in lens[::200]:
        ax[0].plot(particle, c="k", lw=1.0, alpha=0.5)
    ax[1].hist(lens.ravel(), bins=50, density=True, orientation="horizontal")
    ax[0].set_xlabel("Time")
    ax[0].set_ylabel("LENS")
    ax[1].set_xlabel("Probability density")
    plt.show()
    plt.close()


def example_for_number_of_neighbors(universe: Universe, r_cut: float) -> None:
    """Compute number of neighbors on a trajectory."""
    neighlist = list_neighbours_along_trajectory(
        universe=universe,
        r_cut=r_cut,
        centers="all",
        selection="all",
        respect_pbc=True,
        n_jobs=4,
    )  # list[list[AtomGroup]]

    # Count the neighbors for each molecule at each frame
    n_neigh = np.array(
        [[len(atom_group) for atom_group in frame] for frame in neighlist]
    ).T

    # Plot the n_neigh time-series and the cumulative distribution
    fig, ax = plt.subplots(1, 2, sharey=True)
    fig.suptitle(f"Number of neighbors within a cutoff radius {r_cut} Å")
    for particle in n_neigh[::200]:
        ax[0].plot(particle, c="k", lw=1.0, alpha=0.5)
    ax[1].hist(
        n_neigh.ravel(),
        bins=np.unique(n_neigh).size,
        density=True,
        orientation="horizontal",
    )
    ax[0].set_xlabel("Time")
    ax[0].set_ylabel(r"$n_{neigh}$")
    ax[1].set_xlabel("Probability density")
    plt.show()
    plt.close()


def main() -> None:
    """An example for the lens module."""
    # Get the trajectory data (as an MDAnalysis.Universe)
    files_path = Path("analysis_workflow")
    universe = Universe(
        files_path / "oxygens.gro",
        files_path / "oxygens.xtc",
    )

    r_cut = 7.5  # Angstrom
    example_for_lens(universe, r_cut)
    example_for_number_of_neighbors(universe, r_cut)


if __name__ == "__main__":
    main()
