from __future__ import annotations

from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from numpy.typing import NDArray

import numpy as np
from scipy.spatial.distance import cdist


def two_nn_estimator(
    data: NDArray[np.float64],
    metric: Callable[[NDArray[np.float64], NDArray[np.float64]], float],
) -> float:
    """Computes the intrinsic dimension with Two-NN estimator.

    See for reference: Facco, E., d'Errico, M., Rodriguez, A. et al.,
    Sci Rep 7, 12140 (2017). https://doi.org/10.1038/s41598-017-11873-y

    Parameters:
        data:
            Has shape (n_atoms, n_frames, n_dims)

        metric:
            Distance metric for scipy.spatial.distance.cdist.

    Example:

        .. testcode:: 2nn-test

            from sklearn.datasets import make_swiss_roll
            import numpy as np
            from dynsight.analysis import two_nn_estimator
            from scipy.spatial.distance import euclidean

            # Swiss roll is intrinsically 2-dimensional
            frame, _ = make_swiss_roll(
                n_samples=500,
                noise=0.05,
                random_state=42,
            )

            # Create artificial trajectory with 3 identical frames
            data = np.stack([frame, frame, frame], axis=1)

            intr_dim = two_nn_estimator(data, euclidean)

        .. testcode:: 2nn-test
            :hide:

            assert np.isclose(intr_dim, 2.0, rtol=0.02)
    """
    list_of_ratios = []
    for time in range(data.shape[1]):
        frame = data[:, time, :]
        distances = cdist(frame, frame, metric=metric)
        sorted_dist = np.sort(distances, axis=1)
        d1 = sorted_dist[:, 1]
        d2 = sorted_dist[:, 2]
        mask = d1 > 0
        mu = d2[mask] / d1[mask]
        list_of_ratios.append(mu)

    ratios = np.concatenate(list_of_ratios)
    return ratios.size / np.sum(np.log(ratios))
