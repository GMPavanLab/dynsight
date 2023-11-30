from __future__ import annotations

from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    import h5py
    import numpy as np
import SOAPify


def timesoap(
    soaptrajectory: np.ndarray,  # type: ignore[type-arg]
    window: int = 1,
    stride: int | None = None,
    backward: bool = False,
    returndiff: bool = True,
    distancefunction: Callable = SOAPify.simpleSOAPdistance,  # type: ignore[type-arg]
) -> tuple[np.ndarray, np.ndarray]:  # type: ignore[type-arg]
    """Performs the 'timeSOAP' analysis on the given SOAP trajectory.

    * Original author: Cristina Caruso
    * Mantainer: Daniele Rapetti

    Parameters:
        soaptrajectory:
            a trajectory of SOAP fingerprints, should have shape
            (nFrames,nAtoms,SOAPlenght)
        window:
            the dimension of the windows between each state confrontations.
            Defaults to 1.
        stride:
            the stride in frames between each state confrontation.
            **NOT IN USE**.
            Defaults to None.
        backward:
            If true the soap distance is referred to the previous frame.
             **NOT IN USE**. Defaulst to True.
        returndiff:
            If true returns also the first derivative of timeSOAP.
            Defaults to True.
        distancefunction:
            the function that define the distance. Defaults to
            :func:`SOAPify.distances.simpleSOAPdistance`.

    Returns:
        - **timedSOAP** the timeSOAP values, shape(frames-1,natoms)
        - **deltaTimedSOAP** the derivatives of timeSOAP,
             shape(natoms, frames-2)
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
    soaptrajectory: np.ndarray,  # type: ignore[type-arg]
    window: int = 1,
    stride: int | None = None,
    backward: bool = False,
    returndiff: bool = True,
) -> tuple[np.ndarray, np.ndarray]:  # type: ignore[type-arg]
    r"""Performs 'timeSOAP' analysis on **normalized** SOAP trajectory.

    this is optimized to use :func:`SOAPify.distances.simpleSOAPdistance`,
    without calling it.

    .. warning:: this function works **only** with normalized numpy.float64
        soap vectors!

        The SOAP distance is calculated with

        .. math::
            d(\vec{a},\vec{b})=\\sqrt{2-2\frac{\vec{a}\\cdot\vec{b}}{\\left\\|\vec{a}\right\\|\\left\\|\vec{b}\right\\|}}

        That is equivalent to

        .. math::
            d(\vec{a},\vec{b})=\\sqrt{2-2\\hat{a}\\cdot\\hat{b}} =
            \\sqrt{\\hat{a}\\cdot\\hat{a}+\\hat{b}\\cdot
            \\hat{b}-2\\hat{a}\\cdot\\hat{b}} =

            \\sqrt{(\\hat{a}-\\hat{b})\\cdot(\\hat{a}-\\hat{b})}

        That is the euclidean distance between the versors

    * Original author: Cristina Caruso
    * Mantainer: Daniele Rapetti

    Parameters:
        soaptrajectory:
            a **normalized** trajectory of SOAP fingerprints, should have shape
            (nFrames,nAtoms,SOAPlenght)
        window:
            the dimension of the windows between each state confrontations.
            Defaults to 1.
        stride:
            the stride in frames between each state confrontation.
            **NOT IN USE**.
            Defaults to None.
        backward:
            If true the soap distance is referred to the previous frame.
             **NOT IN USE**. Defaulst to True.
        returndiff:
            If true returns also the first derivative of timeSOAP.
            Defaults to True.

    Returns:
        - **timedSOAP** the timeSOAP values, shape(frames-1,natoms)
        - **deltaTimedSOAP** the derivatives of timeSOAP, shape(natoms,
            frames-2)
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
) -> tuple[np.ndarray, np.ndarray]:  # type: ignore[type-arg]
    """Shortcut to extract the timeSOAP from large datasets.

    This function is the equivalent to (old cpctools version below):

    - loading a chunk of the trajectory from a h5py.Dataset with a SOAP
      fingerprints trajectory
    - filling the vector with :func:`SOAPify.utils.fillSOAPVectorFromdscribe`
    - normalizing it with :func:`SOAPify.utils.normalizeArray`
    - calculating the timeSOAP with  :func:`timeSOAPsimple`
      and then returning timeSOAP and the derivative

    Parameters:
        soapdataset:
            the dataset with the SOAP fingerprints
        window:
            the dimension of the windows between each state confrontations.
            See :func:`timeSOAPsimple`
            Defaults to 1.
        stride:
            the stride in frames between each state confrontation.
            See :func:`timeSOAPsimple`
            Defaults to None.
        backward:
            If true the soap distance is referred to the previous frame.
            See :func:`timeSOAPsimple` . Defaulst to True.

    Returns:
        - **timedSOAP** the timeSOAP values, shape(frames-1,natoms)
        - **deltaTimedSOAP** the derivatives of timeSOAP,
              shape(natoms, frames-2)
    """
    return SOAPify.getTimeSOAPSimple(
        soapDataset=soapdataset,
        window=window,
        stride=stride,
        backward=backward,
    )
