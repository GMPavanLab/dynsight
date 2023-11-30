from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections import abc

    import h5py
import SOAPify


def saponify_trajectory(
    trajcontainer: h5py.Group | h5py.File,
    soapoutcontainer: h5py.Group | h5py.File,
    soaprcut: float,
    soapnmax: int,
    soaplmax: int,
    soapoutputchunkdim: int = 100,
    soapnjobs: int = 1,
    soapatommask: str | None = None,
    centersmask: abc.Iterable | None = None,  # type: ignore[type-arg]
    soap_respectpbc: bool = True,
    soapkwargs: dict | None = None,  # type: ignore[type-arg]
    usesoapfrom: SOAPify.engine.KNOWNSOAPENGINES = "dscribe",
    dooverride: bool = False,
    verbose: bool = True,
    usetype: str = "float64",
) -> None:
    """Calculate the SOAP fingerprints for each atom in a hdf5 trajectory.

    Works exaclty as :func:`saponifyMultipleTrajectories` except for that it
    calculates the fingerprints only for the passed trajectory group
    (see :func:`SOAPify.HDF5er.HDF5erUtils.isTrajectoryGroup`).

    `SOAPatomMask` and `centersMask` are mutually exclusive (see
    :func:`SOAPify.engine.getSoapEngine`)

    Parameters:
        trajcontainer:
            Container of trajectory.
        trajFname (str):
            The name of the hdf5 file in wich the trajectory is stored
        trajectoryGroupPath (str):
            the path of the group that contains the trajectory in trajFname
        soapoutcontainer (str):
            the name of the hdf5 file that will contain the ouput or the SOAP
            analysis.
        exportDatasetName (str):
            the name of the dataset that will contain the SOAP results,
            it will be saved in the group called "SOAP"
        soapoutputchunkdim (int, optional):
            The dimension of the chunck of data in the SOAP results dataset.
            Defaults to 100.
        soapnjobs (int, optional):
            the number of concurrent SOAP calculations (option passed to the
            desired SOAP engine). Defaults to 1.
        soapatommask (str, optional):
            the symbols of the atoms whose SOAP fingerprint will be calculated
            (option passed to the desired SOAP engine). Defaults to None.
        centersmask:
            Mask.
        soaprcut (float, optional):
            The cutoff for local region in angstroms. Should be bigger than 1
            angstrom (option passed to the desired SOAP engine).
            Defaults to 8.0.
        soapnmax (int, optional):
            The number of radial basis functions (option passed to the desired
            SOAP engine). Defaults to 8.
        soaplmax (int, optional):
            The maximum degree of spherical harmonics (option passed to the
            desired SOAP engine). Defaults to 8.
        soap_respectpbc (bool, optional):
            Determines whether the system is considered to be periodic
            (option passed to the desired SOAP engine). Defaults to True.
        soapkwargs (dict, optional):
            additional keyword arguments to be passed to the SOAP engine.
            Defaults to {}.
        usesoapfrom (KNOWNSOAPENGINES, optional):
            This string determines the selected SOAP engine for the
            calculations.
            Defaults to "dscribe".
        dooverride (bool, optional):
            if False will raise and exception if the user ask to override an
            already existing DataSet. Defaults to False.
        verbose (bool, optional):
            regulates the verbosity of the step by step operations.
            Defaults to True.
        usetype (str,optional):
            The precision used to store the data. Defaults to "float64".
    """
    SOAPify.saponifyTrajectory(
        trajContainer=trajcontainer,
        SOAPoutContainer=soapoutcontainer,
        SOAPrcut=soaprcut,
        SOAPnmax=soapnmax,
        SOAPlmax=soaplmax,
        SOAPOutputChunkDim=soapoutputchunkdim,
        SOAPnJobs=soapnjobs,
        SOAPatomMask=soapatommask,
        centersMask=centersmask,
        SOAP_respectPBC=soap_respectpbc,
        SOAPkwargs=soapkwargs,
        useSoapFrom=usesoapfrom,
        doOverride=dooverride,
        verbose=verbose,
        useType=usetype,
    )


def saponify_multiple_trajectories(
    trajcontainers: h5py.Group | h5py.File,
    soapoutcontainers: h5py.Group | h5py.File,
    soaprcut: float,
    soapnmax: int,
    soaplmax: int,
    soapoutputchunkdim: int = 100,
    soapnjobs: int = 1,
    soapatommask: list[str] | None = None,
    centersmask: abc.Iterable | None = None,  # type: ignore[type-arg]
    soap_respectpbc: bool = True,
    soapkwargs: dict | None = None,  # type: ignore[type-arg]
    usesoapfrom: SOAPify.engine.KNOWNSOAPENGINES = "dscribe",
    dooverride: bool = False,
    verbose: bool = True,
    usetype: str = "float64",
) -> None:
    """Calculate and store SOAP descriptor for all trajectories in group/file.

    `saponifyMultipleTrajectories` checks if any of the group contained in
    `trajContainers` is a "trajectory group"
    (see :func:`SOAPify.HDF5er.HDF5erUtils.isTrajectoryGroup`) and then
    calculates the soap fingerprints for that trajectory and saves the result
    in a dataset within the `SOAPoutContainers` group or file

    `SOAPatomMask` and `centersMask` are mutually exclusive (see
    :func:`SOAPify.engine.getSoapEngine`)

    Parameters:
        trajcontainers (h5py.Group|h5py.File):
            The file/group that contains the trajectories
        soapoutcontainers (h5py.Group|h5py.File):
            The file/group that will store the SOAP results
        soaprcut (float):
            The cutoff for local region in angstroms. Should be bigger than 1
            angstrom (option passed to the desired SOAP engine).
            Defaults to 8.0.
        soapnmax (int):
            The number of radial basis functions (option passed to the desired
            SOAP engine). Defaults to 8.
        soaplmax (int):
            The maximum degree of spherical harmonics (option passed to the
            desired SOAP engine). Defaults to 8.
        soapoutputchunkdim (int, optional):
            The dimension of the chunck of data in the SOAP results dataset.
            Defaults to 100.
        soapnjobs (int, optional):
            the number of concurrent SOAP calculations (option passed to the
            desired SOAP engine). Defaults to 1.
        soapatommask (list[str], optional):
            the symbols of the atoms whose SOAP fingerprint will be calculated
            (option passed to getSoapEngine). Defaults to None.
        centersmask (Iterable, optional):
            the indexes of the atoms whose SOAP fingerprint will be calculated
            (option passed getSoapEngine). Defaults to None.
        soap_respectpbc (bool, optional):
            Determines whether the system is considered to be periodic (option
            passed to the desired SOAP engine). Defaults to True.
        soapkwargs (dict, optional):
            additional keyword arguments to be passed to the selected SOAP
            engine.
            Defaults to {}.
        usesoapfrom (KNOWNSOAPENGINES, optional):
            This string determines the selected SOAP engine for the
            calculations.
            Defaults to "dscribe".
        dooverride (bool, optional):
            if False will raise and exception if the user ask to override an
            already existing DataSet. Defaults to False.
        verbose (bool, optional):
            regulates the verbosity of the step by step operations.
            Defaults to True.
        usetype (str,optional):
            The precision used to store the data. Defaults to "float64".

    """
    SOAPify.saponifyMultipleTrajectories(
        trajContainers=trajcontainers,
        SOAPoutContainers=soapoutcontainers,
        SOAPrcut=soaprcut,
        SOAPnmax=soapnmax,
        SOAPlmax=soaplmax,
        SOAPOutputChunkDim=soapoutputchunkdim,
        SOAPnJobs=soapnjobs,
        SOAPatomMask=soapatommask,
        centersMask=centersmask,
        SOAP_respectPBC=soap_respectpbc,
        SOAPkwargs=soapkwargs,
        useSoapFrom=usesoapfrom,
        doOverride=dooverride,
        verbose=verbose,
        useType=usetype,
    )
