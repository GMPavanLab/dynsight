from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    import h5py
    import numpy as np
import SOAPify


def timesoap(
    soaptrajectory: np.ndarray[float, Any],
    window: int = 1,
    stride: int | None = None,
    backward: bool = False,
    returndiff: bool = True,
    distancefunction: Callable[
        [np.ndarray[float, Any], np.ndarray[float, Any], int], float
    ] = SOAPify.simpleSOAPdistance,
) -> tuple[np.ndarray[float, Any], np.ndarray[float, Any]]:
    """Performs the 'timeSOAP' analysis on the given SOAP trajectory.

    * Original author: Cristina Caruso
    * Maintainer: Matteo Becchi

    Parameters:
        soaptrajectory:
            A trajectory of SOAP fingerprints. Should have shape
            (nFrames, nAtoms, SOAPlength).
        window:
            The dimension of the windows between each state confrontation.
            Defaults to 1.
        stride:
            The stride in frames between each state confrontation.
            **NOT IN USE**. Defaults to None.
        backward:
            If True, the SOAP distance is referred to the previous frame.
            **NOT IN USE**. Defaults to True.
        returndiff:
            If True, returns the first derivative of timeSOAP.
            Defaults to True.
        distancefunction:
            The function that defines the distance. Defaults to
            :func:`SOAPify.distances.simpleSOAPdistance`.

    Returns:
        tuple[np.ndarray[float, Any], np.ndarray[float, Any]]:
            A tuple of elements:
                - **timedSOAP**: The timeSOAP values, shape (frames-1, nAtoms).
                - **deltaTimedSOAP**: The derivatives of timeSOAP,
                  shape (nAtoms, frames-2).
    """
    return SOAPify.timeSOAP(
        SOAPTrajectory=soaptrajectory,
        window=window,
        stride=stride,
        backward=backward,
        returnDiff=returndiff,
        distanceFunction=distancefunction,
    )


def timesoapsimple(
    soaptrajectory: np.ndarray[float, Any],
    window: int = 1,
    stride: int | None = None,
    backward: bool = False,
    returndiff: bool = True,
) -> tuple[np.ndarray[float, Any], np.ndarray[float, Any]]:
    r"""Performs 'timeSOAP' analysis on **normalized** SOAP trajectory.

    * Original author: Cristina Caruso
    * Maintainer: Matteo Becchi

    This function is optimized to use
    :func:`SOAPify.distances.simpleSOAPdistance`,
    without directly calling it.

    .. warning:: This function works **only** with normalized numpy.float64
        SOAP vectors!

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

    Parameters:
        soaptrajectory:
            A **normalized** trajectory of SOAP fingerprints. Should have shape
            (nFrames, nAtoms, SOAPlength).
        window:
            The dimension of the window between each state confrontation.
            Defaults to 1.
        stride:
            The stride in frames between each state confrontation.
            **NOT IN USE**. Defaults to None.
        backward:
            If True, the SOAP distance is referred to the previous frame.
            **NOT IN USE**. Defaults to True.
        returndiff:
            If True, returns the first derivative of timeSOAP.
            Defaults to True.

    Returns:
        tuple[np.ndarray[float, Any], np.ndarray[float, Any]]:
            - **timedSOAP**: The timeSOAP values, shape (frames-1, nAtoms).
            - **deltaTimedSOAP**: The derivatives of timeSOAP,
                shape (nAtoms, frames-2).
    """
    return SOAPify.timeSOAPsimple(
        SOAPTrajectory=soaptrajectory,
        window=window,
        stride=stride,
        backward=backward,
        returnDiff=returndiff,
    )


def gettimesoapsimple(
    soapdataset: h5py.Dataset,
    window: int = 1,
    stride: int | None = None,
    backward: bool = False,
) -> tuple[np.ndarray[float, Any], np.ndarray[float, Any]]:
    """Shortcut to extract the timeSOAP from large datasets.

    * Original author: Cristina Caruso
    * Maintainer: Cristina Caruso

    This function is equivalent to the following (old cpctools version):

    - Loading a chunk of the trajectory from an h5py.Dataset containing SOAP
      fingerprints.
    - Filling the vector with :func:`SOAPify.utils.fillSOAPVectorFromdscribe`.
    - Normalizing it with :func:`SOAPify.utils.normalizeArray`.
    - Calculating the timeSOAP with :func:`timeSOAPsimple`, and then returning
      timeSOAP and its derivative.

    Parameters:
        soapdataset:
            The dataset containing the SOAP fingerprints.
        window:
            The dimension of the window between each state confrontation.
            See :func:`timeSOAPsimple`.
            Defaults to 1.
        stride:
            The stride in frames between each state confrontation.
            See :func:`timeSOAPsimple`.
            Defaults to None.
        backward:
            If True, the SOAP distance is referred to the previous frame.
            See :func:`timeSOAPsimple`. Defaults to True.

    Returns:
        tuple[np.ndarray[float, Any], np.ndarray[float, Any]]:
            - **timedSOAP**: The timeSOAP values, shape (frames-1, nAtoms).
            - **deltaTimedSOAP**: The derivatives of timeSOAP, shape
                (nAtoms, frames-2).
    """
    return SOAPify.getTimeSOAPSimple(
        soapDataset=soapdataset,
        window=window,
        stride=stride,
        backward=backward,
    )
