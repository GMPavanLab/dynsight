"""Compute timeSOAP."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from numpy.typing import NDArray

import numpy as np


def normalize_soap(
    soaptrajectory: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Returns the SOAP spectra normalized to unitary length.

    Parameters:
        soaptrajectory : shape (n_particles, n_frames, n_comp)
            The SOAP spctra for the trajectory.

    Returns:
        np.ndarray of shape (n_particles, n_frames, n_components)
            The normalized SOAP spectra.

    Example:

        .. testsetup:: tsoap1-test

            import pathlib

            path = pathlib.Path('source/_static/ex_test_files')

        .. testcode:: tsoap1-test

            import numpy as np
            import MDAnalysis
            from dynsight.soap import (
                saponify_trajectory,
                normalize_soap,
            )

            univ = MDAnalysis.Universe(path / "trajectory.xyz")
            cutoff = 2.0

            soap = saponify_trajectory(univ, cutoff, soap_respectpbc=False)
            unitary_soap = normalize_soap(soap)

        .. testcode:: tsoap1-test
            :hide:

            assert np.isclose(
                np.sum(unitary_soap[0]), 21.987915602216525,
                atol=1e-6, rtol=1e-3)
    """
    norms = np.linalg.norm(soaptrajectory, axis=-1)
    mask = norms > 0.0
    norm_soap = np.zeros(soaptrajectory.shape)
    norm_soap[mask] = soaptrajectory[mask] / norms[..., np.newaxis][mask]
    return norm_soap


def soap_distance(
    v_1: NDArray[np.float64],
    v_2: NDArray[np.float64],
) -> NDArray[np.float64]:
    r"""Computes the Kernel SOAP distance between two SOAP spectra.

    The SOAP distance is calculated with:

    .. math::
        d(\vec{v_1},\vec{v_2}) =
        \sqrt{2-2\frac{\vec{v_1}\cdot\vec{v_2}}{||\vec{v_1}||\cdot||\vec{v_2}||}}

    This is equivalent to:

    .. math::
        d(\vec{v_1},\vec{v_2})=\sqrt{2-2\hat{v_1}\cdot\hat{v_2}} =
        \sqrt{\hat{v_1}\cdot\hat{v_1}+\hat{v_2}\cdot\hat{v_2}-2\hat{v_1}
        \cdot\hat{v_2}} =
        \\
        \sqrt{(\hat{v_1}-\hat{v_2})\cdot(\hat{v_1}-\hat{v_2})} =
        ||\hat{v_1}-\hat{v_2}||

    This represents the Euclidean distance between the versors.

    Parameters:
        v_1, v_2 :
            SOAP spectra.

    Returns:
        NDArray[np.float64] : the SOAP distances between the input spectra.

    Example:

        .. testsetup:: tsoap2-test

            import pathlib

            path = pathlib.Path('source/_static/ex_test_files')

        .. testcode:: tsoap2-test

            import numpy as np
            import MDAnalysis
            from dynsight.soap import saponify_trajectory, soap_distance

            univ = MDAnalysis.Universe(path / "trajectory.xyz")
            cutoff = 2.0

            soap = saponify_trajectory(univ, cutoff, soap_respectpbc=False)
            soap_dist = soap_distance(soap[0][0], soap[0][1])

        .. testcode:: tsoap2-test
            :hide:

            assert np.isclose(soap_dist, 0.10292206044570047,
                atol=1e-6, rtol=1e-3)
    """
    norm_v_1 = normalize_soap(v_1)
    norm_v_2 = normalize_soap(v_2)

    cos_theta = np.sum(norm_v_1 * norm_v_2, axis=-1)

    # Takes care of numerical errors
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    tsoap = np.maximum(0, 2 - 2 * cos_theta)

    return np.sqrt(tsoap)


def timesoap(
    soaptrajectory: NDArray[np.float64],
    delay: int = 1,
) -> NDArray[np.float64]:
    """Performs the 'timeSOAP' analysis on the given SOAP trajectory.

    timeSOAP was developed by Cristina Caurso. See for reference the paper
    https://doi.org/10.1063/5.0147025.

    Parameters:
        soaptrajectory: shape (n_particles, n_frames, n_comp)
            The SOAP spctra for the trajectory.

        delay:
            The delay between frames on which timeSOAP is computed.
            Default is 1.

    Returns:
        NDArray[np.float64]
            Values of timesoap of each particle at each frame.
            Has shape (n_particles, n_frames - 1).

    Example:

        .. testsetup:: tsoap3-test

            import pathlib

            path = pathlib.Path('source/_static/ex_test_files')

        .. testcode:: tsoap3-test

            import numpy as np
            import MDAnalysis
            from dynsight.soap import saponify_trajectory, timesoap

            univ = MDAnalysis.Universe(path / "trajectory.xyz")
            cutoff = 2.0

            soap = saponify_trajectory(univ, cutoff, soap_respectpbc=False)
            tsoap = timesoap(soap)

        .. testcode:: tsoap3-test
            :hide:

            assert np.isclose(np.sum(tsoap), 191.3446863900592,
                atol=1e-6, rtol=1e-3)
    """
    value_error = "delay value outside bounds"
    if delay < 1 or delay >= soaptrajectory.shape[1]:
        raise ValueError(value_error)

    return soap_distance(
        soaptrajectory[:, :-delay, :], soaptrajectory[:, delay:, :]
    )
