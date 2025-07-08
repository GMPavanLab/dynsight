Information gain analysis
=========================

For the theoretical aspects of this work, see INSERT REF.

This recipe explains how to compute the information gain through clustering 
analysis. The LENS descriptor is computed on a many-body trajectory, and then
onion clustering is run on a broad range of time resolutions ∆t. The
information gain and the Shannon entropy of the environments is computed for
each value of ∆t. The analysis is implemented using both onion 1.0.13 and
2.0.0 ("onion smooth").

Let's start by creating a :class:`.trajectory.Trj` and computing LENS:

.. testcode:: recipe3-test

    from pathlib import Path
    from dynsight.trajectory import Trj

    # Loading an example trajectory
    files_path = Path("../tests/systems/")
    trj = Trj.init_from_xtc(
        traj_file=files_path / "balls_7_nvt.xtc",
        topo_file=files_path / "balls_7_nvt.gro",
    )

    _, lens = trj.get_lens(r_cut=10.0)


The following functions take as input the LENS dataset, and a list of values
of time resolutions ∆t, and for each of these perform Onion clustering, and
compute the information gain achieved through clustering with that ∆t. 

DESCRIBE THE OUTPUT

In this first function, the clustering is performed with the "old" onion
algorithm (time-series segmentation in consecutive windows):

.. testcode:: recipe3-test

    delta_t_list = np.unique(np.geomspace(1, 1000, 20, dtype=int))

    def info_gain_with_onion(delta_t_list: np.ndarray | list[int]):
        n_clusters = np.zeros(delta_t_list.size)
        clusters_frac = []
        info_gain = np.zeros(delta_t_list.size)
        clusters_entr = []

        for i, delta_t in enumerate(delta_t_list):
            reshaped_data = dynsight.onion.helpers.reshape_from_nt(
                lens.dataset, delta_t
            )
            state_list, labels = dynsight.onion.onion_uni(reshaped_data)

            n_clusters[i] = len(state_list)
            tmp_frac = [0.0]
            for state in state_list:
                tmp_frac.append(state.perc)
            tmp_frac[0] = 1.0 - np.sum(tmp_frac)
            clusters_frac.append(tmp_frac)

            # and compute the information gain:
            info_gain_y[j], *_ = dynsight.analysis.compute_entropy_gain(
                reshaped_data, labels, n_bins=40
            )
        results.append(info_gain_y)

        # Or we can do clustering using both (x, y) variables:
        info_gain_xy = np.zeros(delta_t_list.size)
        tmp1_dataset = np.transpose(dataset, (2, 0, 1))
        for j, delta_t in enumerate(delta_t_list):
            reshaped_data = dynsight.onion.helpers.reshape_from_dnt(
                tmp1_dataset, delta_t
            )
            state_list, labels = dynsight.onion.onion_multi(reshaped_data)

            if j == example_delta_t:
                dynsight.onion.plot.plot_output_multi(
                    f"Example_{i}_2D.png",
                    tmp1_dataset,
                    state_list,
                    labels,
                    delta_t,
                )

            # and compute the information gain:
            # We need an array (n_samples, n_dims), and labels (n_samples,)
            n_sequences = int(n_frames / delta_t)
            long_labels = np.repeat(labels, delta_t)
            tmp = dataset[:, : n_sequences * delta_t, :]
            ds_reshaped = tmp.reshape((-1, n_dims))

            info_gain_xy[j], *_ = dynsight.analysis.compute_entropy_gain_multi(
                ds_reshaped, long_labels, n_bins=[40, 40]
            )
        # Need to multiply by two because it's 2 dimensional, and the output
        # of the info_gain functions is normalized by the log volume of the
        # phase space, which is 2D is doubled
        info_gain_xy *= 2
        results.append(info_gain_xy)
