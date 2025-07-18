"""Copiable code from Recipe #1."""

from pathlib import Path

from dynsight.trajectory import Trj


def main() -> None:
    """Code from the Recipe #1."""
    # Loading an example trajectory
    files_path = Path("../tests/systems/")
    trj = Trj.init_from_xtc(
        traj_file=files_path / "balls_7_nvt.xtc",
        topo_file=files_path / "balls_7_nvt.gro",
    )

    _ = trj.get_soap(
        r_cut=2.0,  # cutoff radius for neighbors list
        n_max=4,  # n_max SOAP parameter
        l_max=4,  # l_max SOAP parameter
        selection="all",  # compute on a selection of particles
        centers="all",  # compute for a selection of centers
        respect_pbc=False,  # consider PBC
        n_core=1,  # use multiprocessing to speed up calculations
    )

    # Loading an example trajectory
    files_path = Path("../tests/systems/")
    trj = Trj.init_from_xtc(
        traj_file=files_path / "balls_7_nvt.xtc",
        topo_file=files_path / "balls_7_nvt.gro",
    )

    # Computing number of neighbors from scratch
    neigcounts, n_neig = trj.get_coord_number(
        r_cut=2.0,  # cutoff radius for neighbors list
        selection="all",  # compute on a selection of particles
        neigcounts=None,  # it will be computed and returned
    )

    # Now for LENS we already have neigcounts
    _, lens = trj.get_lens(
        r_cut=2.0,  # cutoff radius for neighbors list
        selection="all",  # compute on a selection of particles
        neigcounts=neigcounts,  # no need to compute it again
    )


if __name__ == "__main__":
    main()
