from __future__ import annotations

from typing import TYPE_CHECKING

try:
    import SOAPify
except ImportError:
    SOAPify = None

if TYPE_CHECKING:
    import numpy as np


def simplekernelsoap(x: np.ndarray, y: np.ndarray) -> float:  # type: ignore[type-arg]
    """A simpler SOAP Kernel than :func:`KernelSoap`, power is always 1.

    * Original author: Daniele Rapetti

    Parameters:
        x:
            a SOAP fingerprint.
        y:
            a SOAP fingerprint.

    Returns:
        kernel value
    """
    if SOAPify is None:
        msg = "Please install SOAPify|h5py with cpctools."
        raise ModuleNotFoundError(msg)

    return SOAPify.simpleKernelSoap(x, y)


def simplesoapdistance(x: np.ndarray, y: np.ndarray) -> float:  # type: ignore[type-arg]
    """A simpler SOAP distance than :func:`SOAPdistance`, power is always 1.

    * Original author: Daniele Rapetti

    Parameters:
        x:
            a SOAP fingerprint.
        y:
            a SOAP fingerprint.

    Returns:
        float: the distance between the two fingerprints, between
        :math:`0` and :math:`2`.
    """
    if SOAPify is None:
        msg = "Please install SOAPify|h5py with cpctools."
        raise ModuleNotFoundError(msg)

    return SOAPify.simpleSOAPdistance(x, y)


def kernelsoap(x: np.ndarray, y: np.ndarray, n: int) -> float:  # type: ignore[type-arg]
    """The SOAP Kernel with a variable power.

    * Original author: Daniele Rapetti

    Parameters:
        x:
            a SOAP fingerprint.
        y:
            a SOAP fingerprint.
        n:
            the power to elevate the result of the kernel

    Returns:
        kernel value
    """
    if SOAPify is None:
        msg = "Please install SOAPify|h5py with cpctools."
        raise ModuleNotFoundError(msg)

    return SOAPify.kernelSoap(x, y, n)


def soapdistance(x: np.ndarray, y: np.ndarray, n: int = 1) -> float:  # type: ignore[type-arg]
    """The SOAP distance between two SOAP fingerprints.

    * Original author: Daniele Rapetti

    Parameters:
        x:
            a SOAP fingerprint.
        y:
            a SOAP fingerprint.
        n:
            the power to elevate the result of the kernel

    Returns:
        float: the distance between the two fingerprints, between
        :math:`0` and :math:`2`
    """
    if SOAPify is None:
        msg = "Please install SOAPify|h5py with cpctools."
        raise ModuleNotFoundError(msg)

    return SOAPify.SOAPdistance(x, y, n)


def soapdistancenormalized(x: np.ndarray, y: np.ndarray) -> float:  # type: ignore[type-arg]
    """The SOAP distance between two normalized SOAP fingerprints.

    * Original author: Daniele Rapetti

    The pre-normalized vectors should net some performace over the classic
    kernel.

    Parameters:
        x:
            a normalized SOAP fingerprint.
        y:
            a normalized SOAP fingerprint.

    Returns:
        float: the distance between the two fingerprints, between
        :math:`0` and :math:`2`
    """
    if SOAPify is None:
        msg = "Please install SOAPify|h5py with cpctools."
        raise ModuleNotFoundError(msg)

    return SOAPify.SOAPdistanceNormalized(x, y)
