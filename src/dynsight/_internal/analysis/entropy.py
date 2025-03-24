from __future__ import annotations

import numpy as np
import numpy.typing as npt
from scipy.spatial.distance import pdist


def compute_data_entropy(
    data: npt.NDArray[np.float64],
    data_range: tuple[float, float],
    n_bins: int,
) -> float:
    """Compute the entropy of a data distribution.

    It is normalized so that a uniform distribution has unitary entropy.

    Parameters:
        data:
            The dataset for which the entropy is to be computed.

        data_range:
            A tuple (min, max) specifying the range over which the data
            histogram must be computed.

        n_bins:
            The number of bins with which the data histogram must be computed.

    Returns:
        float:
            entropy:
                The value of the normalized Shannon entropy of the dataset.

    Example:

        .. testcode:: shannon1-test

            import numpy as np
            from dynsight.analysis import compute_data_entropy

            np.random.seed(1234)
            data = np.random.rand(100, 100)
            data_range = (float(np.min(data)), float(np.max(data)))

            data_entropy = compute_data_entropy(
                data,
                data_range,
                n_bins=40,
            )

        .. testcode:: shannon1-test
            :hide:

            assert np.isclose(data_entropy, 0.9993954419344714)
    """
    if data.size == 0:
        msg = "data is empty"
        raise ValueError(msg)
    counts, _ = np.histogram(
        data,
        bins=n_bins,
        range=data_range,
    )
    probs = counts / np.sum(counts)  # Data probabilities are needed
    entropy = -np.sum([p * np.log2(p) for p in probs if p > 0.0])
    entropy /= np.log2(n_bins)
    return entropy


def compute_multivariate_entropy(
    data: npt.NDArray[np.float64],
    data_ranges: list[tuple[float, float]],
    n_bins: list[int],
) -> float:
    """Compute the entropy of a multivariate data distribution.

    It is normalized so that a uniform distribution has unitary entropy.

    Parameters:
        data:
            shape (n_samples, n_dimensions)
            The dataset for which the entropy is to be computed.

        data_ranges:
            A list of tuples [(min1, max1), (min2, max2), ...] specifying
            the range over which the histogram must be computed for each
            dimension.

        n_bins:
            A list of integers specifying the number of bins for each
            dimension.

    Returns:
        float:
            entropy:
                The value of the normalized Shannon entropy of the dataset.

    Example:

        .. testcode:: shannon-multi-test

            import numpy as np
            from dynsight.analysis import compute_multivariate_entropy

            np.random.seed(1234)
            data = np.random.rand(1000, 2)  # 2D dataset
            data_ranges = [(0.0, 1.0), (0.0, 1.0)]
            n_bins = [40, 40]

            data_entropy = compute_multivariate_entropy(
                data,
                data_ranges,
                n_bins,
            )

        .. testcode:: shannon-multi-test
            :hide:

            assert np.isclose(data_entropy, 0.8837924363474094)
    """
    n_points, n_dims = data.shape
    if data.size == 0:
        msg = "data is empty"
        raise ValueError(msg)
    if n_dims != len(data_ranges) or n_dims != len(n_bins):
        msg = "Mismatch between data dimensions, data_ranges, and n_bins"
        raise ValueError(msg)

    counts, _ = np.histogramdd(data, bins=n_bins, range=data_ranges)
    probs = counts / np.sum(counts)  # Probability distribution
    entropy = -np.sum(probs[probs > 0] * np.log2(probs[probs > 0]))
    entropy /= np.log2(np.prod(n_bins))  # Normalization

    return entropy


def compute_entropy_gain(
    data: npt.NDArray[np.float64],
    labels: npt.NDArray[np.int64],
    n_bins: int = 20,
) -> tuple[float, float, float, float]:
    """Compute the relative information gained by the clustering.

    Parameters:
        data:
            The dataset over which the clustering is performed.

        labels:
            The clustering labels. Has the same shape as "data".

        n_bins:
            The number of bins with which the data histogram must be computed.
            Default is 20.

    Returns:
        tuple[float, float, float, float]
            * The absolute information gain :math:`H_0 - H_{clust}`
            * The relative information gain :math:`(H_0 - H_{clust}) / H_0`
            * The Shannon entropy of the initial data :math:`H_0`
            * The shannon entropy of the clustered data :math:`H_{clust}`

    Example:

        .. testcode:: shannon2-test

            import numpy as np
            from dynsight.analysis import compute_entropy_gain

            np.random.seed(1234)
            data = np.random.rand(100, 100)
            labels = np.random.randint(-1, 2, size=(100, 100))

            _, entropy_gain, *_ = compute_entropy_gain(
                data,
                labels,
                n_bins=40,
            )

        .. testcode:: shannon2-test
            :hide:

            assert np.isclose(entropy_gain, 0.0010065005804883983)
    """
    if data.shape[0] != labels.shape[0]:
        msg = (
            f"data ({data.shape}) and labels ({labels.shape}) "
            "must have same shape[0]"
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
    info_gain = total_entropy - clustered_entropy

    return (
        info_gain,
        info_gain / total_entropy,
        total_entropy,
        clustered_entropy,
    )


def compute_multivariate_gain(
    data: npt.NDArray[np.float64],
    labels: npt.NDArray[np.int64],
    n_bins: list[int],
) -> tuple[float, float, float, float]:
    """Compute the relative information gained by the clustering.

    Parameters:
        data:
            shape (n_samples, n_dimensions)
            The dataset over which the clustering is performed.

        labels:
            shape (n_samples,)
            The clustering labels.

        n_bins:
            The number of bins with which the data histogram must be computed,
            one for each dimension.

    Returns:
        tuple[float, float, float, float]
            * The absolute information gain :math:`H_0 - H_{clust}`
            * The relative information gain :math:`(H_0 - H_{clust}) / H_0`
            * The Shannon entropy of the initial data :math:`H_0`
            * The shannon entropy of the clustered data :math:`H_{clust}`

    Example:

        .. testcode:: shannon2-multi-test

            import numpy as np
            from dynsight.analysis import compute_multivariate_gain

            np.random.seed(1234)
            data = np.random.rand(1000, 2)  # 2D dataset
            n_bins = [40, 40]
            labels = np.random.randint(-1, 2, size=1000)

            _, entropy_gain, *_ = compute_multivariate_gain(
                data,
                labels,
                n_bins=n_bins,
            )

        .. testcode:: shannon2-multi-test
            :hide:

            assert np.isclose(entropy_gain, 0.13171418273750357)
    """
    if data.shape[0] != labels.shape[0]:
        msg = (
            f"data ({data.shape}) and labels ({labels.shape}) "
            "must have same shape[0]"
        )
        raise RuntimeError(msg)

    data_range = [(float(np.min(tmp)), float(np.max(tmp))) for tmp in data.T]

    # Compute the entropy of the raw data
    total_entropy = compute_multivariate_entropy(
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
        entr[i] = compute_multivariate_entropy(
            data[mask],
            data_range,
            n_bins,
        )

    # Compute the entropy of the clustered data
    clustered_entropy = np.dot(frac, entr)
    info_gain = total_entropy - clustered_entropy

    return (
        info_gain,
        info_gain / total_entropy,
        total_entropy,
        clustered_entropy,
    )


def sample_entropy(
    particle: npt.NDArray[np.float64],
    m_par: int = 2,
    r_factor: float = 0.2,
) -> float:
    """Compute the sample entropy of a single time-series.

    Parameters:
        particle : np.ndarray of shape (n_frames,)
            The time-series data for a single particle.

        m_par : int (default 2)
            The m parameter (length of the considered overlapping windows).

        r_factor : float (default 0.2)
            The similarity threshold between signal windows.

    Returns:
        float
            The sample entropy of the time-seris.

    Example:

        .. testcode:: sampen1-test

            import numpy as np
            from dynsight.analysis import sample_entropy

            np.random.seed(1234)
            data = np.random.rand(1000)

            samp_en = sample_entropy(
                data,
                m_par=2,
                r_factor=0.2,
            )

        .. testcode:: sampen1-test
            :hide:

            assert np.isclose(samp_en, 2.2351853395754424)
    """
    n_frames = len(particle)
    if n_frames < m_par + 1:
        err_msg = "Time-series too short"
        raise ValueError(err_msg)
    r_th = r_factor * np.std(particle)

    # To store counts of similar pairs for m and m+1
    number_of_pairs = [0, 0]

    for i, m in enumerate([m_par + 1, m_par]):
        # Create overlapping windows of length m
        window_list = np.array(
            [particle[j : j + m] for j in range(n_frames - m + 1)]
        )

        # Compute pairwise distances (Chebyshev is typical for SampEn)
        distances = pdist(window_list, metric="chebyshev")

        # Count pairs within the threshold r_th
        number_of_pairs[i] = np.sum(distances < r_th)

    if number_of_pairs[1] == 0.0 and number_of_pairs[0] == 0.0:
        err_msg = "Distance threshold too strict"
        raise ValueError(err_msg)

    return -np.log(number_of_pairs[0] / number_of_pairs[1])


def compute_sample_entropy(
    data: npt.NDArray[np.float64],
    m_par: int = 2,
    r_factor: float = 0.2,
) -> float:
    """Compute the average sample entropy of a time-series dataset.

    Parameters:
        data : np.ndarray of shape (n_particles, n_frames)

        m_par : int (default 2)
            The m parameter (length of the considered overlapping windows).

        r_factor : float (default 0.2)
            The similarity threshold between signal windows.

    Returns:
        float
            The sample entropy of the dataset (average over all the particles).

    Example:

        .. testcode:: sampen2-test

            import numpy as np
            from dynsight.analysis import compute_sample_entropy

            np.random.seed(1234)
            data = np.random.rand(100, 100)

            aver_samp_en = compute_sample_entropy(
                data,
                m_par=2,
                r_factor=0.2,
            )

        .. testcode:: sampen2-test
            :hide:

            assert np.isclose(aver_samp_en, 2.210674176898837)
    """
    sampen = [sample_entropy(particle, m_par, r_factor) for particle in data]

    return float(np.nanmean(sampen))
