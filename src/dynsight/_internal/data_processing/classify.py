from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import h5py
    import numpy as np
from typing import Callable

import SOAPify


def createreferencesfromtrajectory(
    h5soapdataset: h5py.Dataset,
    addresses: dict,  # type: ignore[type-arg]
    lmax: int,
    nmax: int,
    donormalize: bool = True,
) -> SOAPify.SOAPReferences:
    """Generate a SOAPReferences object.

    by storing the data found from h5SOAPDataSet.
    The atoms are selected trough the addresses dictionary.

    Parameters:
        h5soapdataset:
            the dataset with the SOAP fingerprints
        addresses:
            the dictionary with the names and addresses of the fingerprints.
            The keys will be used as the names of the references and the values
            assigned to the keys must be tuples or similar with the number of
            the chosen frame and the atom number (for example
            ``dict(exaple=(framenum, atomID))``)
        lmax:
            To be done.
        nmax:
            To be done.
        donormalize:
            If True normalizes the SOAP vector before storing them.
            Defaults to True.
        settingsUsedInDscribe:
            If none the SOAP vector are not preprocessed, if not none the SOAP
            vectors are decompressed, as dscribe omits the symmetric part of
            the spectra. Defaults to None.

    Returns:
        SOAPReferences:
            the container with the selected references
    """
    return SOAPify.createReferencesFromTrajectory(
        h5SOAPDataSet=h5soapdataset,
        addresses=addresses,
        lmax=lmax,
        nmax=nmax,
        doNormalize=donormalize,
    )


def getdistancebetween(
    data: np.ndarray,  # type: ignore[type-arg]
    spectra: np.ndarray,  # type: ignore[type-arg]
    distancecalculator: Callable,  # type: ignore[type-arg]
) -> np.ndarray:  # type: ignore[type-arg]
    """Generate an array with the distances between the the data and `spectra`.

    Parameters:
        data:
            the array of the data
        spectra:
            the references
        distancecalculator:
            the function to calculate the distances

    Returns:
        np.ndarray:
            the array of the distances (the shape is
            `(data.shape[0], spectra.shape[0])`)
    """
    return SOAPify.getDistanceBetween(
        data=data,
        spectra=spectra,
        distanceCalculator=distancecalculator,
    )


def getdistancesfromref(
    soaptrajdata: h5py.Dataset,
    references: SOAPify.SOAPReferences,
    distancecalculator: Callable,  # type: ignore[type-arg]
    donormalize: bool = False,
) -> np.ndarray:  # type: ignore[type-arg]
    """Generates distances between a SOAP-hdf5 trajectory and references.

    Parameters:
        soaptrajdata:
            the dataset containing the SOAP trajectory
        references:
            the contatiner of the references
        distancecalculator:
            the function to calculate the distances
        donormalize:
            informs the function if the given data needs to be normalized
            before calculating the distance. Defaults to False.

    Returns:
        np.ndarray: the "trajectory" of distance from the given references
    """
    return SOAPify.getDistancesFromRef(
        SOAPTrajData=soaptrajdata,
        references=references,
        distanceCalculator=distancecalculator,
        doNormalize=donormalize,
    )


def getdistancesfromrefnormalized(
    soaptrajdata: h5py.Dataset,
    references: SOAPify.SOAPReferences,
) -> np.ndarray:  # type: ignore[type-arg]
    """Shortcut for :func:`.getDistancesFromRef` with normalization.

    see :func:`SOAPify.SOAPClassify.getDistancesFromRef`,
    the distance calculator is :func:`SOAPdistanceNormalized` and
    doNormalize is set to True

    Parameters:
        soaptrajdata:
            the dataset containing the SOAP trajectory
        references:
            the contatiner of the references
    Returns:
        the trajectory of distance from the given references
    """
    return SOAPify.getDistancesFromRefNormalized(
        SOAPTrajData=soaptrajdata,
        references=references,
    )


def mergereferences(*x: SOAPify.SOAPReferences) -> SOAPify.SOAPReferences:
    """Merges a list of :class:`SOAPReferences` into a single object.

    Raises:
        ValueError:
            if the lmax and the nmax of the references are not the same

    Returns:
        SOAPReferences:
            a new `SOAPReferences` that contains the concatenated list of
            references
    """
    return SOAPify.mergeReferences(x)


def savereferences(
    h5position: h5py.Group | h5py.File,
    targetdatasetname: str,
    refs: SOAPify.SOAPReferences,
) -> None:
    """Export the given references in the indicated group/hdf5 file.

    Parameters:
        h5position:
            The file object of the group where to save the references
        targetdatasetname:
            the name to give to the list of references
        refs:
            the `SOAPReferences` object to be exported
    """
    return SOAPify.saveReferences(
        h5position=h5position,
        targetDatasetName=targetdatasetname,
        refs=refs,
    )


def getreferencesfromdataset(dataset: h5py.Dataset) -> SOAPify.SOAPReferences:
    """Returns a :class:`SOAPReferences` with the initializated data.

    TODO: check if the dataset contains the needed references

    Parameters:
        dataset:
            the dataset with the references

    Returns:
        SOAPReferences: the prepared references container
    """
    return SOAPify.getReferencesFromDataset(dataset)


def applyclassification(
    soaptrajdata: h5py.Dataset,
    references: SOAPify.SOAPReferences,
    distancecalculator: Callable,  # type: ignore[type-arg]
    donormalize: bool = False,
) -> SOAPify.SOAPclassification:
    """Applies the references to a dataset.

    generates the distances from the given references and then classify all of
    the atoms by the closest element in the dictionary

    Parameters:
        soaptrajdata:
            the dataset containing the SOAP trajectory
        references:
            the contatiner of the references
        distancecalculator:
            the function to calculate the distances
        donormalize:
            informs the function if the given data needs to be normalized
            before caclulating the distance. Defaults to False.

    Returns:
        SOAPclassification: The result of the classification
    """
    return SOAPify.applyClassification(
        SOAPTrajData=soaptrajdata,
        references=references,
        distanceCalculator=distancecalculator,
        doNormalize=donormalize,
    )
