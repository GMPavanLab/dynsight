"""Code from the Cleaning Cluster Population tutorial."""

from pathlib import Path

import numpy as np

import dynsight
from dynsight.data_processing import cleaning_cluster_population
from dynsight.trajectory import Trj


def main() -> None:
    """Code from the Spatial Denoising tutorial."""
    # Loading an example trajectory
    files_path = Path.cwd()
    trj = Trj.init_from_xtc(
        traj_file=files_path / "ice_water_ox.xtc",
        topo_file=files_path / "ice_water_ox.gro",
    )

    # Computing TimeSOAP descriptor
    _, tsoap = trj.get_timesoap(
        r_cut=10,
        n_max=8,
        l_max=8,
        n_jobs=4,  # Adjust n_jobs according to your computer capabilities
    )

    # Applying Spatial Denoising
    sliced_trj = trj.with_slice(slice(0, -1, 1))
    sp_denoised_tsoap = tsoap.spatial_average(
        trj=sliced_trj,
        r_cut=10,
        n_jobs=4,  # Adjust n_jobs according to your computer capabilities
    )

    # Performing Onion Clustering on the descriptor computed
    delta_t_list, n_clust, unclass_frac, labels = (
        sp_denoised_tsoap.get_onion_analysis(
            delta_t_min=2,
            delta_t_num=20,
            fig1_path=files_path / "denoised_onion_analysis.png",
            fig2_path=files_path / "cluster_population.png",
        )
    )

    # Saving Onion output in an array
    onion_output = np.array([delta_t_list, n_clust, unclass_frac]).T

    # Assigning clusters with population <5% to the unclassified environment
    # (label=-1)
    cleaned_labels = cleaning_cluster_population(
        labels,
        threshold=0.05,
        assigned_env=-1,
    )

    # Updating the data and plotting the cleaned number of clusters and
    # unclassified fraction.
    # Since unchanged, windows can be copied from above.
    delta_t_list = onion_output[:, 0]

    n_clust = np.zeros(delta_t_list.shape[0], dtype=np.int64)
    unclass_frac = np.zeros(delta_t_list.shape[0])
    for i in range(delta_t_list.shape[0]):
        n_clust[i] = np.unique(cleaned_labels[:, :, i]).size - 1
        unclass_frac[i] = np.sum(cleaned_labels[:, :, i] == -1) / np.size(
            cleaned_labels[:, :, i]
        )

    cleaned_onion_output = np.array([delta_t_list, n_clust, unclass_frac]).T

    dynsight.onion.plot_smooth.plot_time_res_analysis(
        files_path / "cleaned_onion_analysis.png", cleaned_onion_output
    )


if __name__ == "__main__":
    main()
