from __future__ import annotations

from typing import Any

import numpy as np


def self_time_correlation(
    data: np.ndarray[float, Any],
    max_delay: int | None = None,
) -> np.ndarray[float, Any]:
    """Computes the mean self time correlation function for time-series.

    Takes as input an array of shape (n_particles, n_frames), where the
    element [i][t] is the value of some quantity for particle i at time t.
    For each particle, computes the self time correlation function (self-TCF).
    Returns the self-TCF averaged over all the particles, normalized so that
    the value at t = 0 is equal to 1.

    * Author: Matteo Becchi

    Parameters:
        data:
            Array of shape (n_particles, n_frames).

        max_delay:
            The TCF will be computed for all the delay values from zero to
            max_delay - 1. If None, il will be set to the maximum possible
            delay. Default is None.

    Returns:
        np.ndarray of shape (max_delay,):
            The values of the TCF for all delays from zero to max_delay - 1.

    Example:

        .. testcode:: tcf-test

            import numpy as np
            from dynsight.analysis import self_time_correlation

            # Create random input
            np.random.seed(1234)
            n_particles = 100
            n_frames = 100
            data = np.random.rand(n_particles, n_frames)

            time_corr = self_time_correlation(data)

        .. testcode:: tcf-test
            :hide:

            assert time_corr[0] == 1.0

    """
    n_part, n_frames = data.shape
    # Subtract the mean for each particle
    data = data - np.mean(data, axis=1, keepdims=True)

    if max_delay is None:
        max_delay = n_frames

    correlation = np.zeros(max_delay)
    for t_prime in range(max_delay):
        # Compute correlation for time lag t_prime
        valid_t = n_frames - t_prime
        corr_sum = 0
        for n in range(n_part):
            corr_sum += np.dot(
                data[n, :valid_t],
                data[n, t_prime : valid_t + t_prime],
            )
        correlation[t_prime] = corr_sum / (n_part * valid_t)

    # Normalize the correlation function
    correlation /= correlation[0]
    return correlation


def cross_time_correlation(
    data: np.ndarray[float, Any],
    max_delay: int | None = None,
) -> np.ndarray[float, Any]:
    """Computes the mean cross time correlation function for time-series.

    Takes as input an array of shape (n_particles, n_frames), where the
    element [i][t] is the value of some quantity for particle i at time t.
    For each particles pair, computes the cross time correlation function
    (cross-TCF). Returns the cross-TCF averaged over all the particles pairs.

    * Author: Matteo Becchi

    Parameters:
        data:
            Array of shape (n_particles, n_frames).

        max_delay:
            The TCF will be computed for all the delay values from zero to
            max_delay - 1. If None, il will be set to the maximum possible
            delay. Default is None.

    Returns:
        np.ndarray of shape (max_delay,):
            The values of the TCF for all delays from zero to max_delay - 1.

    Example:

        .. testcode:: tcf-test

            import numpy as np
            from dynsight.analysis import cross_time_correlation

            # Create random input
            np.random.seed(1234)
            n_particles = 100
            n_frames = 100
            data = np.random.rand(n_particles, n_frames)

            time_corr = cross_time_correlation(data)

        .. testcode:: tcf-test
            :hide:

            assert time_corr[0] > 0.0

    """
    n_part, n_frames = data.shape

    # Subtract the mean for each particle
    data = data - np.mean(data, axis=1, keepdims=True)

    if max_delay is None:
        max_delay = n_frames

    # Initialize the cross-correlation array
    cross_correlation = np.zeros(max_delay)

    # Loop over all time lags t'
    for t_prime in range(max_delay):
        valid_t = n_frames - t_prime
        corr_sum = 0

        # Compute cross-correlation for each pair of particles (i, j)
        for i in range(n_part):
            for j in range(n_part):
                if i != j:  # Skip self-correlation
                    corr_sum += np.dot(
                        data[i, :valid_t],
                        data[j, t_prime : valid_t + t_prime],
                    )

        # Average over particle pairs (i, j)
        cross_correlation[t_prime] = corr_sum / (
            n_part * (n_part - 1) * valid_t
        )

    return cross_correlation