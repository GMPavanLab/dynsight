from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pathlib

    import h5py
    from MDAnalysis import AtomGroup, Universe

import SOAPify


def mda_to_hdf5(
    mdatrajectory: Universe | AtomGroup,
    targethdf5file: str | pathlib.Path,
    groupname: str,
    trajchunksize: int = 100,
    override: bool = False,
    attrs: dict | None = None,  # type: ignore[type-arg]
    trajslice: slice | None = None,
    usetype: str = "float64",
) -> None:
    """Creates an HDF5 trajectory groupfrom an mda trajectory.

    Opens or creates the given HDF5 file, request the user's chosen group,
    then uploads an mda.Universe or an mda.AtomGroup to a h5py.Group in an
    hdf5 file

    **WARNING**: in the HDF5 file if the chosen group is already present it
    will be overwritten by the new data

    Parameters:
        mdatrajectory:
            the container with the trajectory data
        targethdf5file:
            the name of HDF5 file
        groupname:
            the name of the group in wich save the trajectory data within the
            `targetHDF5File`
        trajchunksize:
            The desired dimension of the chunks of data that are stored in the
            hdf5 file. Defaults to 100.
        override:
            If true the hdf5 file will be completely overwritten.
            Defaults to False.
        attrs:
            Additional attributes.
        trajslice:
            Slice of trajectory.
        usetype:
            The precision used to store the data. Defaults to "float64".
    """
    if trajslice is None:
        trajslice = slice(None)
    SOAPify.HDF5er.MDA2HDF5(
        mdaTrajectory=mdatrajectory,
        targetHDF5File=targethdf5file,
        groupName=groupname,
        trajChunkSize=trajchunksize,
        override=override,
        attrs=attrs,
        trajslice=trajslice,
        useType=usetype,
    )


def universe_to_hdf5(
    mdatrajectory: Universe | AtomGroup,
    trajfolder: h5py.Group,
    trajchunksize: int = 100,
    trajslice: slice | None = None,
    usetype: str = "float64",
) -> None:
    """Uploads an mda.Universe or mda.AtomGroup to a h5py.Group in hdf5 file.

    Parameters:
        mdatrajectory:
            the container with the trajectory data
        trajfolder:
            the group in which store the trajectory in the hdf5 file
        trajchunksize:
            The desired dimension of the chunks of data that are stored in the
            hdf5 file. Defaults to 100.
        trajslice:
            Slice of trajectory.
        usetype:
            The precision used to store the data. Defaults to "float64".
    """
    if trajslice is None:
        trajslice = slice(None)
    SOAPify.HDF5er.universe2HDF5(
        mdaTrajectory=mdatrajectory,
        trajFolder=trajfolder,
        trajChunkSize=trajchunksize,
        trajslice=trajslice,
        useType=usetype,
    )
