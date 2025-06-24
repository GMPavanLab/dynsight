"""Functions for computing correlation functions between particles."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from numpy.typing import NDArray

import numpy as np


def self_time_correlation(
    data: NDArray[np.float64],
    max_delay: int | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Computes the mean self time correlation function for time-series.

    Takes as input an array of shape (n_particles, n_frames), where the
    element [i][t] is the value of some quantity for particle i at time t.
    For each particle, computes the self time correlation function (self-TCF).
    Returns the self-TCF averaged over all the particles, normalized so that
    the value at t = 0 is equal to 1.

    Parameters:
        data:
            Array of shape (n_particles, n_frames).

        max_delay:
            The TCF will be computed for all the delay values from zero to
            max_delay - 1 frames. If None, il will be set to the maximum
            possible delay. Default is None.

    Returns:
        tuple[np.ndarray, np.ndarray]:
            * The values of the TCF for all delays from zero to max_delay - 1.

            * The stndard error on the TCF for all delays from zero to
                max_delay - 1.

    Example:

        .. testcode:: tcf-test

            import numpy as np
            from dynsight.analysis import self_time_correlation

            # Create random input
            np.random.seed(1234)
            n_particles = 100
            n_frames = 100
            data = np.random.rand(n_particles, n_frames)

            time_corr, _ = self_time_correlation(data)

        .. testcode:: tcf-test
            :hide:

            assert np.isclose(time_corr[1], 0.005519088806189553)
    """
    n_part, n_frames = data.shape
    data2 = data.copy()
    # Subtract the mean for each particle
    data2 = data2 - np.mean(data2, axis=1, keepdims=True)

    if max_delay is None:
        max_delay = n_frames

    correlation = np.zeros(max_delay)
    correlation_error = np.zeros(max_delay)

    for t_prime in range(max_delay):
        # Compute correlation for time lag t_prime
        valid_t = n_frames - t_prime
        corr_sum = [
            np.dot(
                tmp[:valid_t],
                tmp[t_prime : valid_t + t_prime],
            )
            for tmp in data2
        ]
        correlation[t_prime] = np.mean(corr_sum) / valid_t
        correlation_error[t_prime] = np.std(corr_sum) / (
            valid_t * np.sqrt(n_part)
        )

    # Normalize the correlation function
    norm_fact = correlation[0]
    correlation /= norm_fact
    correlation_error /= norm_fact

    return correlation, correlation_error


def cross_time_correlation(
    data: NDArray[np.float64],
    max_delay: int | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Computes the mean cross time correlation function for time-series.

    Takes as input an array of shape (n_particles, n_frames), where the
    element [i][t] is the value of some quantity for particle i at time t.
    For each particles pair, computes the cross time correlation function
    (cross-TCF). Returns the cross-TCF averaged over all the particles pairs.

    Parameters:
        data:
            Array of shape (n_particles, n_frames).

        max_delay:
            The TCF will be computed for all the delay values from zero to
            max_delay - 1 frames. If None, il will be set to the maximum
            possible delay. Default is None.

    Returns:
        tuple[np.ndarray, np.ndarray]:
            * The values of the TCF for all delays from zero to max_delay - 1.

            * The stndard error on the TCF for all delays from zero to
                max_delay - 1.

    Example:

        .. testcode:: tcf-test

            import numpy as np
            from dynsight.analysis import cross_time_correlation

            # Create random input
            np.random.seed(1234)
            n_particles = 100
            n_frames = 100
            data = np.random.rand(n_particles, n_frames)

            time_corr, _ = cross_time_correlation(data)

        .. testcode:: tcf-test
            :hide:

            assert np.isclose(time_corr[0], 0.0002474572311281272)
    """
    n_part, n_frames = data.shape

    # Subtract the mean for each particle
    data = data - np.mean(data, axis=1, keepdims=True)

    if max_delay is None:
        max_delay = n_frames

    # Initialize the cross-correlation array
    cross_correlation = np.zeros(max_delay)
    correlation_error = np.zeros(max_delay)

    # Loop over all time lags t'
    for t_prime in range(max_delay):
        valid_t = n_frames - t_prime
        corr_sum: list[float] = []

        # Compute cross-correlation for each pair of particles (i, j)
        for i in range(n_part):
            corr_sum.extend(
                np.dot(
                    data[i, :valid_t],
                    data[j, t_prime : valid_t + t_prime],
                )
                for j in range(n_part)
                if i != j  # Skip self-correlation
            )

        # Average over particle pairs (i, j)
        cross_correlation[t_prime] = np.mean(corr_sum) / valid_t
        correlation_error[t_prime] = np.std(corr_sum) / (
            valid_t * np.sqrt(n_part * (n_part - 1))
        )

    return cross_correlation, correlation_error
