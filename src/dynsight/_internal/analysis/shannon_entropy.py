from __future__ import annotations

import numpy as np
import numpy.typing as npt


def compute_data_entropy(
    data: npt.NDArray[np.float64],
    data_range: tuple[float, float],
    n_bins: int,
) -> float:
    """Compute the entropy of a data distribution.

    It is normalized so that a uniform distribution has unitary entropy.

    * Original author: Matteo Becchi

    Parameters:
        data:
            the dataset of which the entropy has to be computed.

        data_range:
            a tuple (min, max) specifying the range over which the data
            histogram must be computed.

        n_bins:
            the number of bins with which the data histogram must be computed.

    Returns:
        entropy:
            the value of the normalized Shannon entropy of the dataset.
    """
    counts, _ = np.histogram(
        data,
        bins=n_bins,
        range=data_range,
    )
    probs = counts / np.sum(counts)  # Data probabilities are needed
    entropy = -np.sum([p * np.log2(p) for p in probs if p > 0.0])
    entropy /= np.log2(n_bins)
    return entropy


def compute_entropy_gain(
    data: npt.NDArray[np.float64],
    labels: npt.NDArray[np.int64],
    n_bins: int = 20,
) -> float | None:
    """Compute the relative information gained by the clustering.

    * Original author: Matteo Becchi

    Parameters:
        data:
            the dataset over which the clustering is performed.

        labels:
            the clustering labels. Has the same shape as "data".

        n_bins (default = 20):
            the number of bins with which the data histogram must be computed.

    Returns:
        a float which is the difference between the entropy of the raw and
        clustered data, relative to the entropy of the raw data.
    """
    if data.shape != labels.shape:
        msg = (
            f"data ({data.shape}) and labels ({labels.shape}) "
            "must have same shape"
        )
        raise RuntimeError(msg)

    data_range = (float(np.min(data)), float(np.max(data)))

    # Compute the entropy of the raw data
    total_entropy = compute_data_entropy(
        data,
        data_range,
        n_bins,
    )

    # Compute the fraction and the entropy of the single clusters
    n_clusters = np.unique(labels).size
    frac, entr = np.zeros(n_clusters), np.zeros(n_clusters)
    for i, label in enumerate(np.unique(labels)):
        mask = labels == label
        frac[i] = np.sum(mask) / labels.size
        entr[i] = compute_data_entropy(
            data[mask],
            data_range,
            n_bins,
        )

    # Compute the entropy of the clustered data
    clustered_entropy = np.dot(frac, entr)

    return (total_entropy - clustered_entropy) / total_entropy
