"""Code from the Spatial Denoising tutorial."""

from pathlib import Path

from dynsight.trajectory import Trj


def main() -> None:
    """Code from the Spatial Denoising tutorial."""
    # Loading an example trajectory
    files_path = Path("INPUT")
    trj = Trj.init_from_xtc(
        traj_file=files_path / "ice_water_ox.xtc",
        topo_file=files_path / "ice_water_ox.gro",
    )

    # Computing SOAP descriptor
    soap = trj.get_soap(
        r_cut=10,
        n_max=8,
        l_max=8,
        n_jobs=4,  # Adjust n_jobs according to your computer capabilities
    )

    # Computing TimeSOAP descriptor
    _, tsoap = trj.get_timesoap(
        soap_insight=soap,
    )
    """
    Alternatively time soap can be also calculated as:

    soap, tsoap = trj.get_timesoap(r_cut=10, n_max=8, l_max=8)

    (this command computes SOAP and then TimeSOAP)
    """
    # Performing Onion Clustering on the descriptor computed
    tsoap.get_onion_analysis(
        delta_t_min=2,
        delta_t_num=20,
        fig1_path=files_path / "onion_analysis.png",
    )

    # Applying Spatial Denoising
    sliced_trj = trj.with_slice(slice(0, -1, 1))
    sp_denoised_tsoap = tsoap.spatial_average(
        trj=sliced_trj,
        r_cut=10,
        n_jobs=4,  # Adjust n_jobs according to your computer capabilities
    )

    # Performing Onion Clustering on the descriptor computed
    sp_denoised_tsoap.get_onion_analysis(
        delta_t_min=2,
        delta_t_num=20,
        fig1_path=files_path / "denoised_onion_analysis.png",
    )

    single_point_onion = tsoap.get_onion_smooth(
        delta_t=37,
    )
    single_point_onion.dump_colored_trj(
        sliced_trj, files_path / "onion_trj.xyz"
    )

    denoised_single_point_onion = sp_denoised_tsoap.get_onion_smooth(
        delta_t=37,
    )
    denoised_single_point_onion.dump_colored_trj(
        sliced_trj, files_path / "onion_trj_denoised.xyz"
    )


if __name__ == "__main__":
    main()
