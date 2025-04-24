"""Creating an example for the new trajectory module."""

import MDAnalysis

from dynsight.trajectory import Trj


def main() -> None:
    """Creating an example for the new trajectory module."""
    # Create a Trj object
    universe = MDAnalysis.Universe(
        "analysis_workflow/oxygens.gro", "analysis_workflow/oxygens.xtc"
    )

    water_trj = Trj(universe)
    water_trj.save("analysis_workflow/water_trj")

    # We want for instance compute LENS on this trajectory
    # This is an Insight object
    water_lens = water_trj.get_lens(r_cut=7.5)

    # We can do spatial average on the computed LENS
    water_smooth = water_lens.spatial_average(water_trj, r_cut=7.5)

    # Or we can perform onion-clustering
    water_onion = water_smooth.get_onion(delta_t=10)
    water_onion.plot_output("analysis_workflow/tmp_fig1.png")
    water_onion.plot_one_trj(
        "analysis_workflow/tmp_fig2.png", particle_id=1234
    )

    # Save the Insight with all the results
    water_onion.save("analysis_workflow/water_lens")


if __name__ == "__main__":
    main()
