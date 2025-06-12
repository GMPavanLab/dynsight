"""Compute SOAP spectra for each atom in a trajectory."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import MDAnalysis
    from numpy.typing import NDArray

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
    centers: str = "all",
    trajslice: slice | None = None,
) -> NDArray[np.float64]:
    """Calculate the SOAP fingerprints for each atom in a MDA universe.

    * Author: Simone Martino

    Parameters:
        universe:
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
        selection:
            Selection of atoms taken from the Universe for the computation.
            More information concerning the selection language can be found
            `here <https://userguide.mdanalysis.org/stable/selections.html>`_
        centers:
            Selection of atoms used as centers for the SOAP calculation. If not
            specified all the atoms present in the selection will be used
            as centers. More information concerning the selection language can
            be found `here <https://userguide.mdanalysis.org/stable/selections.html>`_
        soap_respectpbc:
            Determines whether the system is considered to be periodic
            (option passed to the desired SOAP engine). Defaults to True.
        n_core:
            Number of core used for parallel processing. Default to 1.
        trajslice:
            The slice of the trajectory to consider. Defaults to slice(None).

    Returns:
        The SOAP spectra for all the particles and frames. np.ndarray of shape
        (n_atoms, n_frames, n_components)

    Example:

        .. testsetup:: soap1-test

            import pathlib

            path = pathlib.Path('source/_static/ex_test_files')

        .. testcode:: soap1-test

            import numpy as np
            import MDAnalysis
            from dynsight.soap import saponify_trajectory

            univ = MDAnalysis.Universe(path / "trajectory.xyz")
            cutoff = 2.0

            soap = saponify_trajectory(univ, cutoff, soap_respectpbc=False)

        .. testcode:: soap1-test
            :hide:

            assert np.isclose(
                np.sum(soap[0]), 8627.847941030795, atol=1e-6, rtol=1e-3)
    """
    if trajslice is None:
        trajslice = slice(None)
    sel = universe.select_atoms(selection)
    centers_list_id = sel.select_atoms(centers).indices.tolist()
    centers_list = [
        i for i, idx in enumerate(sel.indices) if idx in centers_list_id
    ]
    species = list(set(sel.atoms.types))

    soap = SOAP(
        species=species,
        r_cut=soaprcut,
        n_max=soapnmax,
        l_max=soaplmax,
        periodic=soap_respectpbc,
    )
    traj = []
    for t_s in universe.trajectory[trajslice]:
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
    tmp_soap = soap.create(
        system=traj,
        n_jobs=n_core,
        centers=[centers_list] * len(traj),
    )

    return np.transpose(tmp_soap, (1, 0, 2))


def fill_soap_vector_from_dscribe(
    soapfromdscribe: NDArray[np.float64],
    lmax: int = 8,
    nmax: int = 8,
) -> NDArray[np.float64]:
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
        The full SOAP spectrum, with the symmetric part explicitly stored.

    Example:

        .. testsetup:: soap2-test

            import pathlib

            path = pathlib.Path('source/_static/ex_test_files')

        .. testcode:: soap2-test

            import numpy as np
            import MDAnalysis
            from dynsight.soap import (
                saponify_trajectory,
                fill_soap_vector_from_dscribe,
            )

            univ = MDAnalysis.Universe(path / "trajectory.xyz")
            cutoff = 2.0

            soap = saponify_trajectory(univ, cutoff, soap_respectpbc=False)

            full_soap = fill_soap_vector_from_dscribe(soap)

        .. testcode:: soap2-test
            :hide:

            assert full_soap.shape[2] == 576
    """
    input_shape = soapfromdscribe.shape

    # Ensure the input is 3D for consistent reshaping
    if soapfromdscribe.ndim == 1:
        soapfromdscribe = soapfromdscribe[np.newaxis, np.newaxis, :]
    elif soapfromdscribe.ndim == 2:  # noqa: PLR2004
        soapfromdscribe = soapfromdscribe[:, np.newaxis, :]

    n_particles, n_frames, _ = soapfromdscribe.shape
    indices = np.triu_indices(nmax)
    matrix_shape = (lmax + 1, nmax, nmax)
    reshaped_soap = soapfromdscribe.reshape(
        n_particles, n_frames, lmax + 1, -1
    )

    matrix = np.zeros((n_particles, n_frames, *matrix_shape))
    matrix[:, :, :, indices[0], indices[1]] = reshaped_soap
    matrix[:, :, :, indices[1], indices[0]] = reshaped_soap

    output = matrix.reshape(n_particles, n_frames, -1)

    # Restore the original shape
    if len(input_shape) == 1:
        return output.squeeze()
    if len(input_shape) == 2:  # noqa: PLR2004
        return output.squeeze(1)
    return output
