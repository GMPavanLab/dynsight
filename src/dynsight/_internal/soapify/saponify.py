from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
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
    centersmask: list[int] | None = None,
    soap_respectpbc: bool = True,
    soapkwargs: dict[Any, Any] | None = None,
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
            The file/group that contains the trajectory.
        soapoutcontainer:
            The file/group that will store the SOAP results.
        soaprcut:
            The cutoff for local region in angstroms. Should be bigger than 1
            angstrom (option passed to the desired SOAP engine).
            Defaults to 8.0.
        soapnmax:
            The number of radial basis functions (option passed to the desired
            SOAP engine). Defaults to 8.
        soaplmax:
            The maximum degree of spherical harmonics (option passed to the
            desired SOAP engine). Defaults to 8.
        soapoutputchunkdim:
            The dimension of the chunck of data in the SOAP results dataset.
            Defaults to 100.
        soapnjobs:
            the number of concurrent SOAP calculations (option passed to the
            desired SOAP engine). Defaults to 1.
        soapatommask:
            the symbols of the atoms whose SOAP fingerprint will be calculated
            (option passed to the desired SOAP engine). Defaults to None.
        centersmask:
            the indexes of the atoms whose SOAP fingerprint will be calculated
            (option passed to the desired SOAP engine). Defaults to None.
        soap_respectpbc:
            Determines whether the system is considered to be periodic
            (option passed to the desired SOAP engine). Defaults to True.
        soapkwargs (dict, optional):
            additional keyword arguments to be passed to the SOAP engine.
            Defaults to {}.
        usesoapfrom:
            This string determines the selected SOAP engine for the
            calculations.
            Defaults to "dscribe".
        dooverride:
            if False will raise and exception if the user ask to override an
            already existing DataSet. Defaults to False.
        verbose:
            regulates the verbosity of the step by step operations.
            Defaults to True.
        usetype:
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
    centersmask: list[int] | None = None,
    soap_respectpbc: bool = True,
    soapkwargs: dict[Any, Any] | None = None,
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
        trajcontainers:
            The file/group that contains the trajectories.
        soapoutcontainers:
            The file/group that will store the SOAP results.
        soaprcut:
            The cutoff for local region in angstroms. Should be bigger than 1
            angstrom (option passed to the desired SOAP engine).
            Defaults to 8.0.
        soapnmax:
            The number of radial basis functions (option passed to the desired
            SOAP engine). Defaults to 8.
        soaplmax:
            The maximum degree of spherical harmonics (option passed to the
            desired SOAP engine). Defaults to 8.
        soapoutputchunkdim:
            The dimension of the chunck of data in the SOAP results dataset.
            Defaults to 100.
        soapnjobs:
            the number of concurrent SOAP calculations (option passed to the
            desired SOAP engine). Defaults to 1.
        soapatommask:
            the symbols of the atoms whose SOAP fingerprint will be calculated
            (option passed to getSoapEngine). Defaults to None.
        centersmask:
            the indexes of the atoms whose SOAP fingerprint will be calculated
            (option passed getSoapEngine). Defaults to None.
        soap_respectpbc:
            Determines whether the system is considered to be periodic (option
            passed to the desired SOAP engine). Defaults to True.
        soapkwargs:
            additional keyword arguments to be passed to the selected SOAP
            engine.
            Defaults to {}.
        usesoapfrom:
            This string determines the selected SOAP engine for the
            calculations.
            Defaults to "dscribe".
        dooverride:
            if False will raise and exception if the user ask to override an
            already existing DataSet. Defaults to False.
        verbose:
            regulates the verbosity of the step by step operations.
            Defaults to True.
        usetype:
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
