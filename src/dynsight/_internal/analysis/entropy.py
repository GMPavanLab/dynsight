from __future__ import annotations

import numpy as np
import numpy.typing as npt
from scipy.spatial.distance import cdist


def compute_shannon(
    data: npt.NDArray[np.float64],
    data_range: tuple[float, float],
    n_bins: int,
) -> float:
    """Compute the Shannon entropy of a univariate data distribution.

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
            from dynsight.analysis import compute_shannon

            np.random.seed(1234)
            data = np.random.rand(100, 100)
            data_range = (float(np.min(data)), float(np.max(data)))

            data_entropy = compute_shannon(
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


def compute_shannon_multi(
    data: npt.NDArray[np.float64],
    data_ranges: list[tuple[float, float]],
    n_bins: list[int],
) -> float:
    """Compute the Shannon entropy of a multivariate data distribution.

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
            from dynsight.analysis import compute_shannon_multi

            np.random.seed(1234)
            data = np.random.rand(1000, 2)  # 2D dataset
            data_ranges = [(0.0, 1.0), (0.0, 1.0)]
            n_bins = [40, 40]

            data_entropy = compute_shannon_multi(
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
    total_entropy = compute_shannon(
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
        entr[i] = compute_shannon(
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


def compute_entropy_gain_multi(
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
            from dynsight.analysis import compute_entropy_gain_multi

            np.random.seed(1234)
            data = np.random.rand(1000, 2)  # 2D dataset
            n_bins = [40, 40]
            labels = np.random.randint(-1, 2, size=1000)

            _, entropy_gain, *_ = compute_entropy_gain_multi(
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
    total_entropy = compute_shannon_multi(
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
        entr[i] = compute_shannon_multi(
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


def pairwise_probabilities(
    particle: npt.NDArray[np.float64],
    r_factor: np.float64 | float,
    m_par: int = 2,
) -> float | None:
    """Finds the sequence matchings for computing sample entropy.

    Counts the number of data sequences of length m_par and m_par + 1 which
    have Chebyshev distance smaller than r_factor.

    .. warning::
        This function is Work In Progress. Do not trust its output.

    Parameters:
        particle : np.ndarray of shape (n_frames,)
            The time-series data for a single particle.

        r_factor : float
            The similarity threshold between signal windows. A common choice
            is 0.2 * the standard deviation of the time-series.

        m_par : int (default 2)
            The m parameter (length of the considered overlapping windows).

    Returns:
        float | None
            Sample entropy of the input time-series. If the result is not a
            float, None is returned.

    Example:

        .. testcode:: sampen1-test

            import numpy as np
            from dynsight.analysis import pairwise_probabilities

            np.random.seed(1234)
            data = np.random.rand(100)
            r_factor = 0.5 * np.std(data)

            sampen = pairwise_probabilities(
                data,
                r_factor=r_factor,
                m_par=2,
            )

        .. testcode:: sampen1-test
            :hide:

            assert np.isclose(sampen, 1.6094379124341003)
    """
    n_sum = len(particle) - m_par

    if n_sum < 1:
        msg = "Time-series too short"
        raise ValueError(msg)

    m1_seq = np.array([particle[i : i + m_par + 1] for i in range(n_sum)])
    m2_seq = np.array([particle[i : i + m_par + 2] for i in range(n_sum - 1)])

    dist_matrix_1 = cdist(m1_seq, m1_seq, metric="chebyshev")
    dist_matrix_2 = cdist(m2_seq, m2_seq, metric="chebyshev")

    np.fill_diagonal(dist_matrix_1, 2 * r_factor)
    np.fill_diagonal(dist_matrix_2, 2 * r_factor)

    pos = np.sum(dist_matrix_1 < r_factor)
    mat = np.sum(dist_matrix_2 < r_factor)

    if pos == 0 or mat == 0:
        return None

    return -np.log(mat / pos)


def compute_sample_entropy(
    data: list[npt.NDArray[np.float64]] | npt.NDArray[np.float64],
    r_factor: np.float64 | float,
    m_par: int = 2,
) -> float:
    """Compute the average sample entropy of a time-series dataset.

    .. warning::
        This function is Work In Progress. Do not trust its output.

    The average is computed ignoring the eventual nan values.

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

    Example:

        .. testcode:: sampen2-test

            import numpy as np
            from dynsight.analysis import compute_sample_entropy

            np.random.seed(1234)
            data = np.random.rand(10, 100)
            r_factor = 0.5 * np.std(data)

            aver_samp_en = compute_sample_entropy(
                data,
                m_par=2,
                r_factor=r_factor,
            )

        .. testcode:: sampen2-test
            :hide:

            assert np.isclose(aver_samp_en, 1.4060395369518306)
    """
    if isinstance(data, np.ndarray) and data.ndim == 1:
        data = [data]

    sampen = []
    for particle in data:
        tmp = pairwise_probabilities(particle, r_factor, m_par)
        if tmp is not None:
            sampen.append(tmp)

    return float(np.mean(sampen))
