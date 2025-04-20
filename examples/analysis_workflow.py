"""Creating an example for the new trajectory module."""

import MDAnalysis

from dynsight.trajectory import Trj


def main() -> None:
    """Creating an example for the new trajectory module."""
    # Create a Trj object
    universe = MDAnalysis.Universe("trajectory.xtc", "trajectory.gro")
    water_trj = Trj(universe)
    water_trj.save()

    # We want for instance compute LENS on this trajectory
    water_lens = water_trj.get_lens(r_cut=10.0)  # This is an Insight object

    # We can do spatial average on the computed LENS
    water_lens.spatial_average(water_trj)

    # And we can perform onion-clustering. Here there should be some way to
    # do all the calculations in parallel using 'yield' in the method...
    water_lens.get_onion(delta_t=10)

    # Save the Insight with all the results
    water_lens.save()


if __name__ == "__main__":
    main()
