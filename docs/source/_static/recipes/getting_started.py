"""Code from the Getting Started tutorial."""

from pathlib import Path
from dynsight.trajectory import Trj

def main() -> None:
    # Loading an example trajectory
    files_path = Path("Path/to/the/folder/where/files/are/stored")
    trj = Trj.init_from_xtc(
        traj_file=files_path / "oxygens.xtc",
        topo_file=files_path / "oxygens.gro",
    )
    # Computing a descriptor
    lens = trj.get_lens(r_cut=7.5)

    # Performing Onion Clustering on the descriptor computed
    lens_onion = lens.get_onion_smooth(delta_t=10)

    # Plotting the results
    lens_onion.plot_output(
        file_path=files_path / "output_plot.png",
        data_insight=lens,
    )
    lens_onion.plot_one_trj(
        file_path=files_path / "single_trj.png",
        data_insight=lens,
        particle_id=1234,
    )
    
    # Exporting a colored trajectory based on the clustering results
    trajslice = slice(0, -1, 1)
    sliced_trj = trj.with_slice(trajslice=trajslice)
    
    lens_onion.dump_colored_trj(
        trj=sliced_trj,
        file_path=files_path / "colored_trj.xyz",
    )

if __name__ == "__main__":
    main()