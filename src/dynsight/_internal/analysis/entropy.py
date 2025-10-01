from __future__ import annotations

from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from numpy.typing import NDArray

import numpy as np
import numpy.typing as npt
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist
from scipy.special import digamma, gamma


def compute_shannon(
    data: NDArray[np.float64],
    data_range: tuple[float, float],
    n_bins: int,
    units: Literal["bit", "nat", "frac"] = "frac",
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

        units:
            The units of measure of the output entropy. If "frac", entropy is
            normalized between 0 and 1 by dividing by log(n_bins).

    Returns:
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
    if units not in ["bit", "nat", "frac"]:
        msg = "units must be bit, nat or frac."
        raise ValueError(msg)
    counts, _ = np.histogram(
        data,
        bins=n_bins,
        range=data_range,
    )
    probs = counts / np.sum(counts)  # Data probabilities are needed
    entropy = -np.sum([p * np.log2(p) for p in probs if p > 0.0])

    if units == "bit":
        return entropy
    if units == "nat":
        return entropy * np.log(2)
    return entropy / np.log2(n_bins)


def compute_kl_entropy(
    data: NDArray[np.float64],
    n_neigh: int = 1,
    units: Literal["bit", "nat"] = "bit",
) -> float:
    """Estimate Shannon differential entropy using Kozachenko-Leonenko.

    The Kozachenko-Leonenko k-nearest neighbors method approximates
    differential entropy based on distances to nearest neighbors
    in the sample space. It's main advantage is being parameter-free.

    Parameters:
        data:
            The dataset for which the entropy is to be computed.
            Shape (n_data,)

        n_neigh:
            The number of neighbors considered in the KL estimator.

        units:
            The units of measure of the output entropy.

    Returns:
        The Shannon differential entropy of the dataset, in bits.

    Example:

        .. testcode:: kl-entropy-test

            import numpy as np
            from dynsight.analysis import compute_kl_entropy

            np.random.seed(1234)
            data = np.random.rand(10000)

            data_entropy = compute_kl_entropy(data)

        .. testcode:: kl-entropy-test
            :hide:

            assert np.isclose(data_entropy, -3.650626496174274)

    """
    if units not in ["bit", "nat"]:
        msg = "units must be bit or nat."
        raise ValueError(msg)
    data = np.sort(data.flatten())
    n_data = len(data)
    eps = data[n_neigh:] - data[:-n_neigh]  # n_neigh-th neighbor distances
    eps = np.clip(eps, 1e-10, None)  # avoid log(0)
    const = digamma(n_data) - digamma(n_neigh) + np.log(2)  # 1D volume
    h_bits = const + np.mean(np.log2(eps))
    if units == "bit":
        return h_bits
    # nat
    return h_bits * np.log(2)


def compute_negentropy(
    data: NDArray[np.float64],
    units: Literal["bit", "nat"] = "bit",
) -> float:
    """Estimate negentropy of a dataset.

    Negentropy is a measure of non-Gaussianity representing the distance
    from a Gaussian distribution; it's used to quantify the amount of
    information in a signal, the Gaussian being the less informative
    distribution for a given variance.

    .. math::

        Neg(X) = H(X_{Gauss}) - H(X)

    Parameters:
        data:
            The dataset for which the entropy is to be computed.

        units:
            The units of measure of the output negentropy.

    Returns:
        The negentropy of the dataset.


    Example:

        .. testcode:: negentropy-test

            import numpy as np
            from dynsight.analysis import compute_negentropy

            np.random.seed(1234)
            data = np.random.rand(10000)

            negentropy = compute_negentropy(data)

        .. testcode:: negentropy-test
            :hide:

            assert np.isclose(negentropy, 0.2609932580146541)

    """
    if units not in ["bit", "nat"]:
        msg = "units must be bit or nat."
        raise ValueError(msg)
    data = data.flatten()
    rng = np.random.default_rng(seed=1234)
    data_norm = (data - np.mean(data)) / np.std(data, ddof=1)
    sigma = np.std(data_norm, ddof=1)
    data_gauss = rng.normal(loc=0.0, scale=sigma, size=data.size)
    h_gauss = compute_kl_entropy(data_gauss, units=units)
    h_data = compute_kl_entropy(data_norm, units=units)
    return h_gauss - h_data


def compute_shannon_multi(
    data: NDArray[np.float64],
    data_ranges: list[tuple[float, float]],
    n_bins: list[int],
    units: Literal["bit", "nat", "frac"] = "frac",
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

        units:
            The units of measure of the output entropy. If "frac", entropy is
            normalized between 0 and 1 by dividing by log(n_bins).

    Returns:
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
    if data.size == 0:
        msg = "data is empty"
        raise ValueError(msg)
    _, n_dims = data.shape
    if n_dims != len(data_ranges) or n_dims != len(n_bins):
        msg = "Mismatch between data dimensions, data_ranges, and n_bins"
        raise ValueError(msg)
    if units not in ["bit", "nat", "frac"]:
        msg = "units must be bit, nat or frac."
        raise ValueError(msg)

    counts, _ = np.histogramdd(data, bins=n_bins, range=data_ranges)
    probs = counts / np.sum(counts)  # Probability distribution
    entropy = -np.sum(probs[probs > 0] * np.log2(probs[probs > 0]))

    if units == "bit":
        return entropy
    if units == "nat":
        return entropy * np.log(2)
    return entropy / np.log2(np.prod(n_bins))  # Normalization


def compute_kl_entropy_multi(
    data: NDArray[np.float64],
    n_neigh: int = 1,
    units: Literal["bit", "nat"] = "bit",
) -> float:
    """Estimate Shannon differential entropy using Kozachenko-Leonenko.

    This function works for multivariate distribution.
    The Kozachenko-Leonenko k-nearest neighbors method approximates
    differential entropy based on distances to nearest neighbors
    in the sample space. It's main advantage is being parameter-free.

    Parameters:
        data:
            The dataset for which the entropy is to be computed.
            Shape (n_data, n_dims)

        n_neigh:
            The number of neighbors considered in the KL estimator.

        units:
            The units of measure of the output entropy.

    Returns:
        The Shannon differential entropy of the dataset, in bits.

    Example:

        .. testcode:: klm-entropy-test

            import numpy as np
            from dynsight.analysis import compute_kl_entropy_multi

            np.random.seed(1234)
            data = np.random.rand(10000, 2)

            data_entropy = compute_kl_entropy_multi(data)

        .. testcode:: klm-entropy-test
            :hide:

            assert np.isclose(data_entropy, -4.319358938644518)

    """
    if units not in ["bit", "nat"]:
        msg = "units must be bit or nat."
        raise ValueError(msg)
    n_samples, dim = data.shape
    tree = cKDTree(data)
    eps, _ = tree.query(data, k=n_neigh + 1, p=2)
    eps = eps[:, -1]  # distance to the n_neigh-th neighbor
    eps = np.clip(eps, 1e-10, None)  # avoid log(0)
    unit_ball_volume = (np.pi ** (dim / 2)) / gamma(dim / 2 + 1)
    entropy = (
        digamma(n_samples)
        - digamma(n_neigh)
        + np.log2(unit_ball_volume)
        + (dim / n_samples) * np.sum(np.log2(eps))
    )

    if units == "bit":
        return entropy
    return entropy * np.log(2)


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


def sample_entropy(
    time_series: NDArray[np.float64],
    r_factor: np.float64 | float,
    m_par: int = 2,
) -> float:
    """Computes the Sample Entropy of a single time-series.

    The Chebyshev distance is used. SampEn takes values between 0 and +inf.
    If the time-series is too short for the chosen m_par, raises ValueError.
    If no matching sequences can be found, raises RuntimeError.

    Parameters:
        time_series : np.ndarray of shape (n_frames,)
            The time-series data.

        r_factor : float
            The similarity threshold between signal windows. A common choice
            is 0.2 * the standard deviation of the time-series.

        m_par : int (default 2)
            The m parameter (length of the considered overlapping windows).

    Returns:
        Sample entropy of the input time-series.

    Example:

        .. testcode:: sampen1-test

            import numpy as np
            from dynsight.analysis import sample_entropy

            np.random.seed(1234)
            data = np.random.rand(100)
            r_factor = 0.5 * np.std(data)

            sampen = sample_entropy(
                data,
                r_factor=r_factor,
                m_par=2,
            )

        .. testcode:: sampen1-test
            :hide:

            assert np.isclose(sampen, 1.6094379124341003)

    """
    n_sum = len(time_series) - m_par

    if n_sum < 1:
        msg = "Time-series too short"
        raise ValueError(msg)

    m1_seq = np.array([time_series[i : i + m_par + 1] for i in range(n_sum)])
    m2_seq = np.array(
        [time_series[i : i + m_par + 2] for i in range(n_sum - 1)]
    )

    dist_matrix_1 = cdist(m1_seq, m1_seq, metric="chebyshev")
    dist_matrix_2 = cdist(m2_seq, m2_seq, metric="chebyshev")

    np.fill_diagonal(dist_matrix_1, 2 * r_factor)
    np.fill_diagonal(dist_matrix_2, 2 * r_factor)

    pos = np.sum(dist_matrix_1 < r_factor)
    mat = np.sum(dist_matrix_2 < r_factor)

    if pos == 0 or mat == 0:
        msg = "No matching sequences found."
        raise RuntimeError(msg)

    return -np.log(mat / pos)
