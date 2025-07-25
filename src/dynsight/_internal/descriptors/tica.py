"""tICA functions."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from numpy.typing import NDArray

import numpy as np

try:
    from deeptime.decomposition import TICA
except ImportError:
    TICA = None


def many_body_tica(
    data: NDArray[np.float64],
    lag_time: int,
    tica_dim: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Perform tICA on trajectories from a many-body system.

    The tICA model is fitted on the entire dataset, concatenating different
    atoms one after the other. Then, with this model, the trajectories of the
    individual atoms are trnsformed individually.
    The model is fitted using the 'kinetic_map' scaling, which is the
    suggested one if data are subsequently clustered.

    This function uses the TICA module from the deeptime package; refer to
    `Moritz Hoffmann et al 2022 Mach. Learn.: Sci. Technol. 3 015009
    <https://iopscience.iop.org/article/10.1088/2632-2153/ac3de0/meta>`_.

    Parameters:
        data: shape (n_atoms, n_frames, n_dims)
            The multivariated data for tICA dimensionality reduction.

        lag_time
            The lagtime under which time correlations are maximized.

        tica_dim
            The number of dimensions to keep.

    Returns:
        * The typical relaxation time-scale of each tIC
        * The coefficient matrix for the tICA projection;
            shape (tica_dim, n_dims)
        * The original dataset projected onto tICs;
            shape (n_atoms, n_frames, tica_dim)

    Example:

        .. code-block:: python

            import numpy as np
            from dynsight.descriptors import many_body_tica

            np.random.seed(42)
            random_array = np.random.rand(100, 100, 10)

            relax_times, coeffs, tica_data = many_body_tica(
                random_array, lag_time=10, tica_dim=3)
    """
    *_, n_dim = data.shape

    if TICA is None:
        msg = "Please install deeptime using pip or conda."
        raise ModuleNotFoundError(msg)

    tica = TICA(lagtime=lag_time, dim=tica_dim)
    full_dataset = data.reshape(-1, n_dim)
    tica.fit(full_dataset, scaling="kinetic_map")

    koopman_model = tica.fetch_model()
    relax_times = koopman_model.timescales()

    eigenvectors = koopman_model.instantaneous_coefficients

    tica_data = np.array([tica.transform(atom) for atom in data])

    return relax_times, eigenvectors[:tica_dim], tica_data
