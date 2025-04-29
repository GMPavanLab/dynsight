"""Creating an example for the new trajectory module."""

from pathlib import Path

import MDAnalysis

from dynsight.trajectory import Trj


def main() -> None:
    """Creating an example for the new trajectory module."""
    files_path = Path("analysis_workflow")

    # Create a Trj object
    universe = MDAnalysis.Universe(
        files_path / "oxygens.gro", files_path / "oxygens.xtc"
    )

    water_trj = Trj(universe)
    water_trj.dump_trj(files_path / "water_trj")

    # We want, for instance, compute LENS on this trajectory
    # From here, we work with Insight objects that contain data from a
    # Trj object
    water_lens = water_trj.get_lens(r_cut=7.5)

    # We can do spatial average on the computed LENS
    water_smooth = water_lens.spatial_average(water_trj, r_cut=7.5)

    # Or we can perform onion-clustering
    water_onion = water_smooth.get_onion(delta_t=10)

    water_onion.plot_output(files_path / "tmp_fig1.png")
    water_onion.plot_one_trj(files_path / "tmp_fig2.png", particle_id=1234)

    # Save the Insight with all the results
    water_onion.dump_insight(files_path / "water_lens")


if __name__ == "__main__":
    main()
