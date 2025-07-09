Information gain analysis
=========================

For the theoretical aspects of this work, see INSERT REF.

This recipe explains how to compute the information gain through clustering 
analysis. We use a syntetic dataset containing a signal that oscillates
between 0 and 1, with Gaussian noise. Onion clustering is run on a broad
range of time resolutions ∆t. The information gain and the Shannon entropy of
the environments is computed for each value of ∆t. The analysis is implemented 
using onion 2.0.0 ("onion smooth").

Let's start by creating a the synthetic dataset:

.. testcode:: recipe3-test

    import numpy as np

    # Parameters
    n_atoms = 10
    num_blocks = 10
    block_size = 100
    sigma = 0.1

    # Generate the array
    tmp_data = []
    for _ in range(n_atoms):
        tmp_data.append(np.concatenate([
            np.random.normal(loc=(i % 2), scale=sigma, size=block_size)
            for i in range(num_blocks)
        ]))
    data = np.array(tmp_data)


The following functions take as input the dataset, and a list of values
of time resolutions ∆t, and for each of these perform Onion clustering, and
compute the information gain achieved through clustering with that ∆t. 

Notice that, for now, this only works with univariate datasets.

The function's output is a tuple of np.ndarray, which for each value of ∆t
contain:

* the number of identified clusters - shape (delta_t_list.size,);
* the population fraction of each cluster - shape (delta_t_list.size, n_clust);
* the information gain - shape (delta_t_list.size,);
* the Shannon entropy of each cluster - shape (delta_t_list.size, n_clust)

Additionally, the (float) dataset Shannon entropy h_0 is returned.

.. testcode:: recipe3-test

    import numpy as np
    import dynsight
    from dynsight.trajectory import Insight

    def info_gain_with_onion(
        delta_t_list: np.ndarray | list[int],
        data: np.array,
        n_bins: int = 40,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
        """Performs full information gain analysis with Onion clustering."""
        data_range = (np.min(data), np.max(data))

        n_clusters = np.zeros(delta_t_list.size)
        clusters_frac = []
        info_gain = np.zeros(delta_t_list.size)
        clusters_entr = []

        for i, delta_t in enumerate(delta_t_list):
            state_list, labels = dynsight.onion.onion_uni_smooth(
                data,
                delta_t=delta_t,
            )

            n_clusters[i] = len(state_list)
            tmp_frac = [0.0]
            for state in state_list:
                tmp_frac.append(state.perc)
            tmp_frac[0] = 1.0 - np.sum(tmp_frac)
            clusters_frac.append(tmp_frac)

            flat_data = data.flatten()
            flat_labels = labels.flatten()
            info_gain[i], _, h_0, _ = dynsight.analysis.compute_entropy_gain(
                flat_data, flat_labels, n_bins=n_bins,
            )

            tmp_entr = []
            label_list = np.unique(labels)
            if label_list[0] != -1:
                tmp_entr.append(-1.0)

            for _, lab in enumerate(label_list):
                mask = labels == lab
                selected_points = data[mask]
                tmp_entr.append(
                    dynsight.analysis.compute_shannon(
                        selected_points,
                        data_range,
                        n_bins=n_bins,
                    )
                )
            clusters_entr.append(tmp_entr)

        max_n_envs = np.max([len(elem) for elem in clusters_entr])
        for i, elem in enumerate(clusters_entr):
            while len(elem) < max_n_envs:
                elem.append(-1.0)
                clusters_frac[i].append(0.0)

        cl_frac = np.array(clusters_frac)
        cl_entr = np.array(clusters_entr)

        return n_clusters, cl_frac, info_gain, cl_entr, h_0

    # Example usage
    _, n_frames = data.shape
    delta_t_list = np.unique(np.geomspace(2, n_frames, 10, dtype=int))

    n_cl, cl_frac, info_gain, cl_entr, h_0 = info_gain_with_onion(
        delta_t_list,
        data,
    )


A default visualization of the results of this analysis can be obtained with
the following function. Be aware that this could require some tweaking to ensure
that clusters identified at different ∆t are matched in the way the user want
them to.

DESCRIBE THE FIGURE

.. testcode:: recipe3-test

    from pathlib import Path
    import matplotlib.pyplot as plt

    def plot_info_results(
        delta_t_list: np.ndarray | list[int],
        cl_frac: np.ndarray,
        cl_entr: np.ndarray,
        h_0: float,
        file_path: Path,
    ) -> None:
        frac = cl_frac.T
        entr = cl_entr.T
        s_list = []
        for i, st_fr in enumerate(frac):
            s_list.append(st_fr * entr[i])
        s_cumul = [s_list[0]]
        for i, tmp_s in enumerate(s_list[1:]):
            s_cumul.append(s_cumul[-1] + tmp_s)

        fig, ax = plt.subplots()

        i_0 = (1 - h_0) * np.ones(len(delta_t_list))
        ax.plot(delta_t_list, i_0, ls="--", c="black", marker="")  # I_0
        ax.fill_between(
            delta_t_list,
            1,
            1 - s_cumul[0],
            alpha=0.5,
        )
        for i, tmp_s in enumerate(s_cumul[1:]):
            ax.fill_between(
                delta_t_list,
                1 - s_cumul[i],
                1 - tmp_s,
                alpha=0.5,
            )
        ax.fill_between(
            delta_t_list, 1 - s_cumul[-1], 1 - h_0, color="gainsboro",
        )
        ax.plot(
            delta_t_list, 1 - s_cumul[-1], c="black", marker="",
        )  # I_clust

        ax.set_ylim(0.0, 1.0)
        ax.set_xlabel(r"Time resolution $\Delta t$")
        ax.set_ylabel(r"Information $I$")
        ax.set_xscale("log")

        fig.savefig(file_path, dpi=600)
        plt.close()

    # Example usage
    plot_info_results(
        delta_t_list,
        cl_frac,
        cl_entr,
        h_0,
        Path("./source/_static/info_plot.png"),
    )


.. testcode:: recipe3-test
    :hide:

    assert np.isclose(info_gain[0], 0.1899727144974609)
