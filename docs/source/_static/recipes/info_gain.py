"""Copiable code from Recipe #3."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from numpy.typing import NDArray

import matplotlib.pyplot as plt
import numpy as np

import dynsight


def info_gain_with_onion(
    delta_t_list: NDArray[np.int64] | list[int],
    data: NDArray[np.float64],
    n_bins: int = 40,
) -> tuple[
    NDArray[np.int64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    float,
]:
    """Performs full information gain analysis with Onion clustering."""
    data_range = (np.min(data), np.max(data))

    n_clusters = np.zeros(len(delta_t_list), dtype=int)
    clusters_frac = []
    info_gain = np.zeros(len(delta_t_list))
    clusters_entr = []

    for i, delta_t in enumerate(delta_t_list):
        state_list, labels = dynsight.onion.onion_uni_smooth(
            data,
            delta_t=delta_t,
        )

        n_clusters[i] = len(state_list)
        tmp_frac = [0.0] + [state.perc for state in state_list]
        tmp_frac[0] = 1.0 - np.sum(tmp_frac)
        clusters_frac.append(tmp_frac)

        flat_data = data.flatten()
        flat_labels = labels.flatten()
        info_gain[i], _, h_0, _ = dynsight.analysis.compute_entropy_gain(
            flat_data,
            flat_labels,
            n_bins=n_bins,
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

    cl_frac = np.array(clusters_frac, dtype=float)
    cl_entr = np.array(clusters_entr)

    return n_clusters, cl_frac, info_gain, cl_entr, h_0


def plot_info_results(
    delta_t_list: NDArray[np.int64] | list[int],
    cl_frac: NDArray[np.float64],
    cl_entr: NDArray[np.float64],
    h_0: float,
    file_path: Path,
) -> None:
    """Plot information gain as a function of ∆t."""
    frac = cl_frac.T
    entr = cl_entr.T
    s_list = []
    for i, st_fr in enumerate(frac):
        s_list.append(st_fr * entr[i])
    s_cumul = [s_list[0]]
    for _, tmp_s in enumerate(s_list[1:]):
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
        delta_t_list,
        1 - s_cumul[-1],
        1 - h_0,
        color="gainsboro",
    )
    ax.plot(
        delta_t_list,
        1 - s_cumul[-1],
        c="black",
        marker="",
    )  # I_clust

    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel(r"Time resolution $\Delta t$")
    ax.set_ylabel(r"Information $I$")
    ax.set_xscale("log")

    fig.savefig(file_path, dpi=600)
    plt.close()


def main() -> None:
    """Copiable code from Recipe #3."""
    rng = np.random.default_rng(1234)

    n_atoms = 10
    num_blocks = 10
    block_size = 100
    sigma = 0.1

    # Generate the array
    tmp_data = [
        np.concatenate(
            [
                rng.normal(loc=(i % 2), scale=sigma, size=block_size)
                for i in range(num_blocks)
            ]
        )
        for _ in range(n_atoms)
    ]
    data = np.array(tmp_data)

    _, n_frames = data.shape
    delta_t_list = np.unique(np.geomspace(2, n_frames, 10, dtype=int))

    n_cl, cl_frac, info_gain, cl_entr, h_0 = info_gain_with_onion(
        delta_t_list,
        data,
    )

    plot_info_results(
        delta_t_list,
        cl_frac,
        cl_entr,
        h_0,
        Path("./source/_static/info_plot.png"),
    )


if __name__ == "__main__":
    main()
