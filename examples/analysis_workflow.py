"""Creating an example for the new trajectory module."""

from pathlib import Path

import MDAnalysis

from dynsight.trajectory import Insight, OnionInsight, Trj


def main() -> None:
    """Creating an example for the new trajectory module."""
    files_path = Path("analysis_workflow")

    # Create a Trj object
    universe = MDAnalysis.Universe(
        files_path / "oxygens.gro", files_path / "oxygens.xtc"
    )
    trj = Trj(universe)

    # We want, for instance, compute LENS on this trajectory
    # From here, we work with an Insight, containing data computed from a Trj
    lens_file = files_path / "lens.json"
    if lens_file.exists():
        lens = Insight.load_from_json(files_path / "lens.json")
    else:
        lens = trj.get_lens(r_cut=7.5)
        lens.dump_to_json(lens_file)

    # We can do spatial average on the computed LENS
    trj_lens = trj.with_slice(slice(1, None, 1))
    lens_smooth = lens.spatial_average(trj_lens, r_cut=7.5)

    # And we can perform onion-clustering
    lens_onion = lens_smooth.get_onion_smooth(delta_t=10)

    lens_onion.plot_output(files_path / "tmp_fig1.png", lens_smooth)
    lens_onion.plot_one_trj(
        files_path / "tmp_fig2.png",
        lens_smooth,
        particle_id=1234,
    )
    lens_onion.dump_colored_trj(trj_lens, files_path / "colored_trj.xyz")

    # Save/load the Insight with all the results
    lens_onion.dump_to_json(files_path / "onion.json")
    _ = OnionInsight.load_from_json(files_path / "onion.json")


if __name__ == "__main__":
    main()
