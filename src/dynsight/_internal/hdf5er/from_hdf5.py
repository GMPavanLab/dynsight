from __future__ import annotations

from typing import TYPE_CHECKING

try:
    import h5py
    import SOAPify
except ImportError:
    h5py = None
    SOAPify = None


if TYPE_CHECKING:
    import MDAnalysis


def create_universe_from_slice(
    trajectorygroup: h5py.Group,
    useslice: slice | None = None,
) -> MDAnalysis.Universe:
    """Creates a MDanalysis.Universe from a trajectory group.

    * Original author: Daniele Rapetti

    Parameters:
        trajectorygroup:
            the given trajectory group
        useslice:
            the asked slice from wich create an universe.
            Defaults to slice(None).

    Returns:
        an universe containing the wnated part of the trajectory
    """
    if SOAPify is None or h5py is None:
        msg = "Please install SOAPify|h5py with cpctools."
        raise ModuleNotFoundError(msg)

    if useslice is None:
        useslice = slice(None)
    return SOAPify.HDF5er.createUniverseFromSlice(
        trajectoryGroup=trajectorygroup,
        useSlice=useslice,
    )
