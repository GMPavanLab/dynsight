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
    """Return the SOAP power spectrum from dscribe results.

    * Original author: Daniele Rapetti
    * Maintainer: Matteo Becchi

    Includes the symmetric part explicitly stored. See the note in
    https://singroup.github.io/dscribe/1.2.x/tutorials/descriptors/soap.html.

    No controls are implemented on the shape of the `soapfromdscribe` vector.

    Parameters:
        soapfromdscribe:
            The result of the SOAP calculation from the dscribe utility.
        lmax:
            The l_max specified in the calculation.
        nmax:
            The n_max specified in the calculation.
        atomtypes:
            The list of atomic species. Defaults to None.
        atomicslices:
            The slices of the SOAP vector relative to the atomic species
            combinations. Defaults to None.

    Returns:
        numpy.ndarray:
            The full SOAP spectrum, with the symmetric part explicitly stored.
    """
    return SOAPify.fillSOAPVectorFromdscribe(
        soapFromdscribe=soapfromdscribe,
        lMax=lmax,
        nMax=nmax,
        atomTypes=atomtypes,
        atomicSlices=atomicslices,
    )


def get_soap_settings(fitsetdata: h5py.Dataset) -> dict[str, Any]:
    """Get the settings of the SOAP calculation.

    * Original author: Daniele Rapetti
    * Maintainer: Matteo Becchi

    You can feed this output directly to :func:`fillSOAPVectorFromdscribe`.

    Parameters:
        fitsetdata:
            A SOAP dataset with attributes.

    Returns:
        dict:
            A dictionary with the following components:
                - **nMax**
                - **lMax**
                - **atomTypes**
                - **atomicSlices**
    """
    return SOAPify.getSOAPSettings(fitsetdata)
