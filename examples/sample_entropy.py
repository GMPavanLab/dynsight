"""How to use the code for the sample entropy computation."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from numpy.typing import NDArray

import matplotlib.pyplot as plt
import numpy as np

import dynsight

COMPUTE = True


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


def main() -> None:
    """How to use the code for the sample entropy computation."""
    # Let's import the LENS signals for water/ice coexistence
    delta_t_list = np.unique(np.geomspace(6, 499, 20, dtype=int))
    t_samp = 0.1

    data_directory = "onion_example_files/data/univariate_time-series.npy"
    data = np.load(data_directory)[::10, 1:]  # First frame of LENS is zero

    # We need to set a unique closeness threshold, to use for each cluster
    r_fact = 0.2  # This is the default value
    r_fact *= np.std(data)

    # We start computing the average sample entropy of the entire dataset
    if COMPUTE:
        aver_samp_en = dynsight.analysis.compute_sample_entropy(
            data, r_factor=r_fact
        )
    else:
        aver_samp_en = 0.1873359455516944

    if COMPUTE:
        # Then we can perform Onion Clustering at different ∆t and compute
        # the sample entropy of the different clusters
        samp_en_list = []
        fractions = []
        for _, delta_t in enumerate(delta_t_list):
            reshaped_data = dynsight.onion.helpers.reshape_from_nt(
                data, delta_t
            )
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

                tmp_sampen = dynsight.analysis.compute_sample_entropy(
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
        np.savetxt("samp_en/tmp_sampen_array.txt", samp_en_array)
        np.savetxt("samp_en/tmp_frac_array.txt", frac_array)

    samp_en_array = np.loadtxt("samp_en/tmp_sampen_array.txt")
    frac_array = np.loadtxt("samp_en/tmp_frac_array.txt")

    labels = ["Unclassified", "Ice", "Interface", "Liquid"]

    fig, ax = plt.subplots(2, 2, figsize=(9, 8))

    for i, state in enumerate(samp_en_array):
        mask = state != 0.0
        ax[0][0].plot(
            delta_t_list[mask] * t_samp,
            state[mask],
            label=labels[i],
            marker="o",
        )
        ax[0][1].plot(
            delta_t_list * t_samp,
            frac_array[i],
            label=labels[i],
            marker="o",
        )
    ax[0][0].plot(
        delta_t_list * t_samp,
        aver_samp_en * np.ones(len(delta_t_list)),
        label="Total SampEn",
        ls="--",
        c="k",
    )
    ax[0][0].set_xlabel(r"Time resolution $\Delta t$ [ns]")
    ax[0][0].set_ylabel("Sample Entropy")
    ax[0][0].set_xscale("log")
    ax[0][0].set_ylim(bottom=0.0)
    ax[0][0].legend()

    ax[0][1].set_xlabel(r"Time resolution $\Delta t$ [ns]")
    ax[0][1].set_ylabel("State population fraction")
    ax[0][1].set_xscale("log")

    y_val = np.zeros(samp_en_array.shape[1])
    for i, state in enumerate(samp_en_array):
        ax[1][0].fill_between(
            delta_t_list * 0.1,
            y_val,
            y_val + state * frac_array[i],
            label=labels[i],
            alpha=0.8,
        )
        y_val += state * frac_array[i]
    ax[1][0].plot(
        delta_t_list * 0.1,
        aver_samp_en * np.ones(len(delta_t_list)),
        ls="--",
        c="k",
    )
    ax[1][0].set_xlabel(r"Time resolution $\Delta t$ [ns]")
    ax[1][0].set_ylabel("Weighted Sample Entropy")
    ax[1][0].set_xscale("log")

    ax[1][1].set_axis_off()

    fig.savefig("samp_en/tmp_SampleEntropy.png", dpi=600)
    plt.show()


if __name__ == "__main__":
    main()
