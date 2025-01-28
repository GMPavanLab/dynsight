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

    Returns:
        np.ndarray of shape (n_atoms, n_frames, n_components)
        The SOAP spectra for all the particles and frames.
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

    tmp_soap = soap.create(system=traj, n_jobs=n_core)

    return np.transpose(tmp_soap, (1, 0, 2))


def fill_soap_vector_from_dscribe(
    soapfromdscribe: np.ndarray[float, Any],
    lmax: int = 8,
    nmax: int = 8,
) -> np.ndarray[float, Any]:
    """Return the SOAP power spectrum from dscribe results.

    * Author: Matteo Becchi

    Parameters:
        soapfromdscribe:
            The result of the SOAP calculation from the dscribe utility.
        lmax:
            The l_max specified in the calculation.
        nmax:
            The n_max specified in the calculation.

    Returns:
        numpy.ndarray:
            The full SOAP spectrum, with the symmetric part explicitly stored.
    """
    n_particles, n_frames, _ = soapfromdscribe.shape
    indices = np.triu_indices(nmax)
    matrix_shape = (lmax + 1, nmax, nmax)
    reshaped_soap = soapfromdscribe.reshape(
        n_particles, n_frames, lmax + 1, -1
    )

    matrix = np.zeros((n_particles, n_frames, *matrix_shape))
    matrix[:, :, :, indices[0], indices[1]] = reshaped_soap
    matrix[:, :, :, indices[1], indices[0]] = reshaped_soap

    return matrix.reshape(n_particles, n_frames, -1)
