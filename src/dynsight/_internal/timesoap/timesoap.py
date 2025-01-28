"""Compute timeSOAP."""

# Author: Matteo Becchi <bechmath@gmail.com>
# Date: January 28, 2025

from __future__ import annotations

from typing import Any

import numpy as np


def normalize_soap(
    soaptrajectory: np.ndarray[float, Any],
) -> np.ndarray[float, Any]:
    """Returns the SOAP spectra normalized to unitary length.

    Parameters
    ----------
    soaptrajectory : np.ndarray of shape (n_particles, n_frames, n_components)
        The SOAP spctra for the trajectory.

    Returns:
    -------
    np.ndarray of shape (n_particles, n_frames, n_components)
        The normalized SOAP spectra.
    """
    norms = np.linalg.norm(soaptrajectory, axis=-1)
    mask = norms > 0.0
    norm_soap = np.zeros(soaptrajectory.shape)
    norm_soap[mask] = soaptrajectory[mask] / norms[..., np.newaxis][mask]
    return norm_soap


def soap_distance(
    v_1: np.ndarray[float, Any],
    v_2: np.ndarray[float, Any],
) -> np.ndarray[float, Any]:
    r"""Computes the Kernel SOAP distance between 2 SOAP spectra.

    The SOAP distance is calculated with:

    .. math::
        d(\vec{a},\vec{b}) =
        \sqrt{2-2\frac{\vec{a}\cdot\vec{b}}{||\vec{a}||\cdot||\vec{b}||}}

    This is equivalent to:

    .. math::
        d(\vec{a},\vec{b})=\sqrt{2-2\hat{a}\cdot\hat{b}} =
        \sqrt{\hat{a}\cdot\hat{a}+\hat{b}\cdot\hat{b}-2\hat{a}\cdot\hat{b}} =
        \sqrt{(\hat{a}-\hat{b})\cdot(\hat{a}-\hat{b})} =
        ||\hat{a}-\hat{b}||

    This represents the Euclidean distance between the versors.

    Parameters
    ----------
    v_1, v_2 : np.ndarray
        SOAP spectra.

    Returns:
    -------
    float : the SOAP distance between the input spectra.
    """
    norm_v_1 = normalize_soap(v_1)
    norm_v_2 = normalize_soap(v_2)

    cos_theta = np.sum(norm_v_1 * norm_v_2, axis=-1)

    # Takes care of numerical errors
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    tsoap = np.maximum(0, 2 - 2 * cos_theta)

    return np.sqrt(tsoap)


def timesoap(
    soaptrajectory: np.ndarray[float, Any],
    delay: int = 1,
) -> np.ndarray[float, Any]:
    """Performs the 'timeSOAP' analysis on the given SOAP trajectory.

    Parameters
    ----------
    soaptrajectory : np.ndarray of shape (n_particles, n_frames, n_components)
        The SOAP spctra for the trajectory.

    delay : int, default=1
        The delay between frames on which timeSOAP is computed.

    Returns:
    -------
    timesoap : np.ndarray of shape (n_particles, n_frames - 1)
        Values of timesoap of each particle at each frame.
    """
    if delay < 1 or delay >= soaptrajectory.shape[1]:
        raise ValueError

    return soap_distance(
        soaptrajectory[:, :-delay, :], soaptrajectory[:, delay:, :]
    )
