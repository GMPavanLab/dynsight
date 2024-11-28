from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import h5py
    import numpy as np
import SOAPify


def fill_soap_vector_from_dscribe(
    soapfromdscribe: np.ndarray[float, Any],
    lmax: int,
    nmax: int,
    atomtypes: list[str] | None = None,
    atomicslices: dict[str, Any] | None = None,
) -> np.ndarray[float, Any]:
    """Returns the SOAP power spectrum from dscribe results.

    * Original author: Daniele Rapetti
    * Mantainer: Matteo Becchi

    With also the symmetric part explicitly stored, see the note in
    https://singroup.github.io/dscribe/1.2.x/tutorials/descriptors/soap.html

    No controls are implemented on the shape of the soapfromdscribe vector.

    Parameters:
        soapfromdscribe:
            the result of the SOAP calculation from the dscribe utility
        lmax:
            the l_max specified in the calculation.
        nmax:
            the n_max specified in the calculation.
        atomtypes:
            the list of atomic species. Defaults to None.
        atomicslices:
            the slices of the SOAP vector relative to the atomic species
            combinations. Defaults to None.

    Returns:
        numpy.ndarray:
            The full soap spectrum, with the symmetric part sorted explicitly.
    """
    return SOAPify.fillSOAPVectorFromdscribe(
        soapFromdscribe=soapfromdscribe,
        lMax=lmax,
        nMax=nmax,
        atomTypes=atomtypes,
        atomicSlices=atomicslices,
    )


def get_soap_settings(fitsetdata: h5py.Dataset) -> dict[str, Any]:
    """Gets the settings of the SOAP calculation.

    * Original author: Daniele Rapetti
    * Mantainer: Matteo Becchi

    You can feed directly this output to :func:`fillSOAPVectorFromdscribe`

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
