from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import h5py
    import MDAnalysis

import SOAPify


def create_universe_from_slice(
    trajectorygroup: h5py.Group,
    useslice: slice | None = None,
) -> MDAnalysis.Universe:
    """Creates a MDanalysis.Universe from a trajectory group.

    Parameters:
        trajectorygroup:
            the given trajectory group
        useslice:
            the asked slice from wich create an universe.
            Defaults to slice(None).

    Returns:
        an universe containing the wnated part of the trajectory
    """
    if useslice is None:
        useslice = slice(None)
    return SOAPify.HDF5er.createUniverseFromSlice(
        trajectoryGroup=trajectorygroup,
        useSlice=useslice,
    )
