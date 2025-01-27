"""Compute SOAP spectra for each atom in a trajectory."""

# Author: Matteo Becchi <bechmath@gmail.com>
# Date January 27, 2025

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import MDAnalysis
    import numpy as np

from ase import Atoms
from dscribe.descriptors import SOAP


def saponify_trajectory(
    universe: MDAnalysis.Universe,
    soaprcut: float,
    soapnmax: int = 8,
    soaplmax: int = 8,
    selection: str = "all",
    soap_respectpbc: bool = True,
    n_core: int = 1,
    **soapkwargs: Any,
) -> np.ndarray[float, Any]:
    """Calculate the SOAP fingerprints for each atom in a MDA universe.

    * Author: Simone Martino

    Parameters:
        universe : MDAnalysis.Universe
            Contains the trajectory.
        soaprcut:
            The cutoff for the local region in angstroms. Should be greater
            than 1 angstrom (option passed to the desired SOAP engine).
            Defaults to 8.0.
        soapnmax:
            The number of radial basis functions (option passed to the desired
            SOAP engine). Defaults to 8.
        soaplmax:
            The maximum degree of spherical harmonics (option passed to the
            desired SOAP engine). Defaults to 8.
        selection : str = "all"
            Selection of atoms in the Universe of which SOAP will be computed.
        soap_respectpbc:
            Determines whether the system is considered to be periodic
            (option passed to the desired SOAP engine). Defaults to True.
        soapkwargs (dict, optional):
            Additional keyword arguments to be passed to the SOAP engine.
            Defaults to {}.
        n_core : int = 1
            Number of core used for parallel processing.
    """
    sel = universe.select_atoms(selection)
    species = list(set(sel.atoms.types))

    soap = SOAP(
        species=species,
        r_cut=soaprcut,
        n_max=soapnmax,
        l_max=soaplmax,
        **soapkwargs,
    )
    traj = []
    for t_s in universe.trajectory:
        pos = sel.atoms.positions
        symbols = sel.atoms.types
        box = t_s.dimensions

        frame = Atoms(
            positions=pos,
            symbols=symbols,
            cell=box,
            pbc=soap_respectpbc,
        )
        traj.append(frame)

    return soap.create(system=traj, n_jobs=n_core)
