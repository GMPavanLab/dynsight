"""How to use the code for the sample entropy computation."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from numpy.typing import NDArray

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import dynsight


def extract_sequences_for_label(
    data: NDArray[np.float64],
    reshaped_data: NDArray[np.float64],
    labels: NDArray[np.int64],
    delta_t: int,
    target_label: int,
) -> list[NDArray[np.float64]]:
    """Creates the list of sequences clustered in a specific cluster.

    Extracts sequences corresponding to a specific label from the original
    data, merging only consecutive windows with the same label.

    Parameters:
        data (np.ndarray):
            Original data of shape (n_atoms, n_frames).

        reshaped_data (np.ndarray):
            Windowed data of shape (n_atoms * n_windows, delta_t).

        labels (np.ndarray):
            Cluster labels of shape (n_atoms * n_windows,).

        delta_t (int):
            The length of each time window.

        target_label (int):
            The label for which sequences should be extracted.

    Returns:
        list: A list of sequences (np.ndarray) where each entry is a
        concatenated segment from `data` corresponding to consecutive
        occurrences of `target_label`.
    """
    n_atoms, n_frames = data.shape
    sequences = []  # List to store sequences for the target label

    # Reshape labels to match (n_atoms, n_windows)
    n_windows = n_frames // delta_t
    labels = labels.reshape((n_atoms, n_windows))

    for atom_idx in range(n_atoms):  # Iterate over each particle
        current_sequence: list[NDArray[np.float64]] = []

        for window_idx in range(n_windows):
            if labels[atom_idx, window_idx] == target_label:
                # Add the current window to the sequence
                current_sequence.extend(
                    reshaped_data[atom_idx * n_windows + window_idx, :]
                )
            # If we had an active sequence, store it and reset
            elif current_sequence:
                sequences.append(np.array(current_sequence))
                current_sequence = []

        # Append the last sequence if it was still active
        if current_sequence:
            sequences.append(np.array(current_sequence))

    return sequences


def combined_sample_entropies(
    data: list[NDArray[np.float64]] | NDArray[np.float64],
    r_factor: np.float64 | float,
    m_par: int = 2,
) -> float:
    """Compute the average sample entropy of a time-series dataset.

    The average is computed ignoring possible nan values.

    Parameters:
        data : np.ndarray of shape (n_particles, n_frames)

        r_factor : float
            The similarity threshold between signal windows. A common choice
            is 0.2 * the standard deviation of the dataset.

        m_par : int (default 2)
            The m parameter (length of the considered overlapping windows).

    Returns:
        float
            The sample entropy of the dataset (average over all the particles).
    """
    if isinstance(data, np.ndarray) and data.ndim == 1:
        data = [data]

    sampen = []
    for particle in data:
        try:
            tmp = dynsight.analysis.sample_entropy(particle, r_factor, m_par)
            sampen.append(tmp)
        except RuntimeError:  # noqa: PERF203
            continue

    return np.mean(np.array(sampen))


def main() -> None:
    """How to use the code for the sample entropy computation."""
    cwd = Path.cwd()
    folder_name = "samp_en"
    folder_path = cwd / folder_name
    if not folder_path.exists():
        folder_path.mkdir()

    delta_t_list = np.unique(np.geomspace(6, 499, 20, dtype=int))
    t_samp = 0.1

    # Let's import the LENS signals for water/ice coexistence
    data_directory = "onion_example_files/data/univariate_time-series.npy"
    data = np.load(data_directory)[::10, 1:]  # First frame of LENS is zero

    # We need to set a unique closeness threshold, to use for each cluster
    r_fact = 0.2  # This is the default value
    r_fact *= np.std(data)

    # We start computing the average sample entropy of the entire dataset
    aver_samp_en = combined_sample_entropies(data, r_factor=r_fact)

    # Then we can perform Onion Clustering at different ∆t and compute
    # the sample entropy of the different clusters
    samp_en_list = []
    fractions = []
    for _, delta_t in enumerate(delta_t_list):
        reshaped_data = dynsight.onion.helpers.reshape_from_nt(data, delta_t)
        state_list, labels = dynsight.onion.onion_uni(reshaped_data)

        tmp_list = []
        tmp_frac = []
        for label in np.unique(labels):
            # This function is necessary to extract and concatenate all the
            # sequences clustered in the cluster under analysis
            selected_data = extract_sequences_for_label(
                data,
                reshaped_data,
                labels,
                delta_t,
                label,
            )

            tmp_sampen = combined_sample_entropies(
                selected_data, r_factor=r_fact
            )
            tmp_list.append(tmp_sampen)
            fraction = np.sum(labels == label) / labels.size
            tmp_frac.append(fraction)

        samp_en_list.append(tmp_list)
        fractions.append(tmp_frac)

    max_n_states = np.max([len(tmp) for tmp in samp_en_list])
    for i, tmp in enumerate(samp_en_list):
        while len(tmp) < max_n_states:
            tmp.append(0.0)
            fractions[i].append(0.0)

    samp_en_array = np.array(samp_en_list).T
    frac_array = np.array(fractions).T

    labels = ["Unclassified", "Ice", "Interface", "Liquid", "Total SampEn"]

    fig, ax = plt.subplots()
    for i, state in enumerate(samp_en_array):
        mask = state != 0.0
        ax.plot(
            delta_t_list * t_samp,
            frac_array[i],
            label=labels[i],
            marker="o",
        )
    ax.set_xlabel(r"Time resolution $\Delta t$ [ns]")
    ax.set_ylabel("State population fraction")
    ax.set_xscale("log")
    ax.legend()
    fig.savefig(folder_path / "Fig1.png", dpi=600)

    fig, ax = plt.subplots()
    for i, state in enumerate(samp_en_array):
        mask = state != 0.0
        ax.plot(
            delta_t_list[mask] * t_samp,
            state[mask],
            label=labels[i],
            marker="o",
        )
    ax.plot(
        delta_t_list * t_samp,
        aver_samp_en * np.ones(len(delta_t_list)),
        label=labels[-1],
        ls="--",
        c="k",
    )
    ax.set_xlabel(r"Time resolution $\Delta t$ [ns]")
    ax.set_ylabel("Sample Entropy")
    ax.set_xscale("log")
    ax.set_ylim(bottom=0.0)
    ax.legend()
    fig.savefig(folder_path / "Fig2.png", dpi=600)

    fig, ax = plt.subplots()
    y_val = np.zeros(samp_en_array.shape[1])
    for i, state in enumerate(samp_en_array):
        ax.fill_between(
            delta_t_list * 0.1,
            y_val,
            y_val + state * frac_array[i],
            label=labels[i],
            alpha=0.8,
        )
        y_val += state * frac_array[i]
    ax.plot(
        delta_t_list * 0.1,
        aver_samp_en * np.ones(len(delta_t_list)),
        label=labels[-1],
        ls="--",
        c="k",
    )
    ax.set_xlabel(r"Time resolution $\Delta t$ [ns]")
    ax.set_ylabel("Weighted Sample Entropy")
    ax.set_xscale("log")
    ax.legend()
    fig.savefig(folder_path / "Fig3.png", dpi=600)

    plt.show()


if __name__ == "__main__":
    main()
