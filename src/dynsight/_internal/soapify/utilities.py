from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import h5py
    import numpy as np
import SOAPify


def fill_soap_vector_from_dscribe(
    soapfromdscribe: np.ndarray,  # type: ignore[type-arg]
    lmax: int,
    nmax: int,
    atomtypes: list | None = None,  # type: ignore[type-arg]
    atomicslices: dict | None = None,  # type: ignore[type-arg]
) -> np.ndarray:  # type: ignore[type-arg]
    """Returns the SOAP power spectrum from dsribe results.

    With also the symmetric part explicitly stored, see the note in
    https://singroup.github.io/dscribe/1.2.x/tutorials/descriptors/soap.html

    No controls are implemented on the shape of the soapFromdscribe vector.

    Parameters:
        soapfromdscribe:
            the result of the SOAP calculation from the dscribe utility
        lmax:
            the l_max specified in the calculation.
        nmax:
            the n_max specified in the calculation.
        atomtypes:
            Needs docs.
        atomicslices:
            Needs docs.

    Returns:
        numpy.ndarray:
            The full soap spectrum, with the symmetric part sored explicitly
    """
    return SOAPify.fillSOAPVectorFromdscribe(
        soapFromdscribe=soapfromdscribe,
        lMax=lmax,
        nMax=nmax,
        atomTypes=atomtypes,
        atomicSlices=atomicslices,
    )


def get_soap_settings(fitsetdata: h5py.Dataset) -> dict:  # type: ignore[type-arg]
    """Gets the settings of the SOAP calculation.

    you can feed directly this output to :func:`fillSOAPVectorFromdscribe`

    Parameters:
        fitsetdata:
            A soap dataset with attributes.

    Returns:
        dict: a dictionary with the following components:
            - **nMax**
            - **lMax**
            - **atomTypes**
            - **atomicSlices**

    """
    return SOAPify.getSOAPSettings(fitsetdata)
