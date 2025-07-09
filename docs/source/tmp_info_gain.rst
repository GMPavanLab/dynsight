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

Notice that, for now, this only works with univariate datasets.

DESCRIBE THE OUTPUT

.. testcode:: recipe3-test

    import numpy as np
    from dynsight.trajectory import Insight

    n_atoms, n_frames = lens.dataset.shape
    delta_t_list = np.unique(np.geomspace(1, n_frames, 10, dtype=int))

    def info_gain_with_onion(
        delta_t_list: np.ndarray | list[int],
        data: Insight,
        n_bins: int = 40,
    ):
        n_clusters = np.zeros(delta_t_list.size)
        clusters_frac = []
        info_gain = np.zeros(delta_t_list.size)
        clusters_entr = []

        for i, delta_t in enumerate(delta_t_list):
            state_list, labels = dynsight.onion.onion_smooth_uni(
                data.dataset,
                delta_t=delta_t,
            )

            n_clusters[i] = len(state_list)
            tmp_frac = [0.0]
            for state in state_list:
                tmp_frac.append(state.perc)
            tmp_frac[0] = 1.0 - np.sum(tmp_frac)
            clusters_frac.append(tmp_frac)

            flat_data = data.dataset.flatten()
            flat_labels = labels.flatten()
            info_gain[i], *_ = dynsight.analysis.compute_entropy_gain(
                flat_data, flat_labels, n_bins=n_bins,
            )


