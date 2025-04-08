"""tICA package."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from numpy.typing import NDArray

import numpy as np
from deeptime.decomposition import TICA


def many_body_tica(
    data: NDArray[np.float64],
    lag_time: int,
    tica_dim: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Perform tICA on trajectories from a many-body system.

    The tICA model is fitted on the entire dataset, concatenating different
    atoms one after the other. Then, with this model, the trajectories of the
    individual atoms are trnsformed individually.

    This function uses the TICA module from the deeptime package; refer to
    Moritz Hoffmann et al 2022 Mach. Learn.: Sci. Technol. 3 015009.

    Parameters:
        data: shape (n_atoms, n_frames, n_dims)
            The multivariated data for tICA dimensionality reduction.

        lag_time
            The time at which time correlations are maximized.

        tica_dim
            The number of tIC to compute.

    Returns:
        * The typical relaxation time-scale of each tIC
        * The coefficient matrix for the tICA projection;
            shape (tica_dim, n_dims)
        * The original dataset projected onto tICs;
            shape (n_atoms, n_frames, tica_dim)

    Example:

        .. testcode:: tica1-test

            import numpy as np
            import dynsight

            np.random.seed(42)
            random_array = np.random.rand(100, 100, 10)

            relax_times, *_ = dynsight.tica.many_body_tica(
                random_array, lag_time=10, tica_dim=3)

        .. testcode:: tica1-test
            :hide:

            assert np.isclose(relax_times[0], 3.104290041516281)

    """
    *_, n_dim = data.shape

    tica = TICA(lagtime=lag_time, dim=tica_dim)
    full_dataset = data.reshape(-1, n_dim)
    tica.fit(full_dataset, scaling="kinetic_map")

    koopman_model = tica.fetch_model()
    relax_times = koopman_model.timescales()

    eigenvectors = koopman_model.instantaneous_coefficients

    tica_soap = np.array([tica.transform(atom) for atom in data])

    return relax_times, eigenvectors[:tica_dim], tica_soap
