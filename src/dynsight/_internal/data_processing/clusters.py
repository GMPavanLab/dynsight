from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray


def cleaning_cluster_population(
    labels: NDArray[np.int64],
    threshold: float,
    assigned_env: int = 1,
) -> NDArray[np.int64]:
    dimension = 2
    if labels.ndim != dimension:
        msg = "Labels must be 2D [part,frames]"
        raise ValueError(msg)

    flat = labels.ravel()
    unique, counts = np.unique(flat, return_counts=True)

    populations = counts / flat.size
    small_clusters = unique[populations <= threshold]

    new_labels = labels.copy()
    if small_clusters.size > 0:
        new_labels[np.isin(labels, small_clusters)] = assigned_env

    return new_labels
