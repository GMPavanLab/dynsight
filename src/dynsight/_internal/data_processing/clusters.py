from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

from dynsight.logs import logger


def cleaning_cluster_population(
    labels: NDArray[np.int64],
    threshold: float,
    assigned_env: int,
    excluded_env: int | list[int] | None = None,
) -> NDArray[np.int64]:
    """Replace labels of low-population clusters with a reference label.

    This function identifies clusters whose relative population is below a
    given threshold and reassigns their labels to a specified environment.
    The population of each cluster is computed as the fraction of elements
    belonging to that label, either for 2D inputs (`(n_atoms, n_frames)`)
    or for 3D inputs (`(n_atoms, n_frames, n_dims)`, where n_dims can
    correspond to the different ∆t from Onion clustering).
    Clusters with a population smaller than or equal to the `threshold` are
    considered negligible and are replaced by the `assigned_env` label,
    while all other labels are preserved.
    `excluded_env` give the possibility to exclude some clusters from
    the re-labeling.

    Parameters:
        labels:
            NumPy array containing the label values.
            The array should have dimensions corresponding
            to either (n_atoms, n_frames) for 2D inputs,
            or (n_atoms, n_frames, n_dims) for 3D inputs.
        threshold:
            A float value from 0 to 1 that defines the threshold at which
            small clusters are neglected.
        assigned_env:
            The label at which smaller clusters are assigned to, if the label
            already exists the population extracted will be merged to the
            existing one.
        excluded_env:
            Clusters that need to be preserved even if their population is
            under the threshold.

    Returns:
        A NumPy array of the same shape as the input descriptor array,
        containing the updated labels. If the input
        array is 2D (n_atoms, n_frames), the output will be a 2D array of
        the same shape. Otherwise, if the input is 3D
        (n_atoms, n_frames, n_dims), the output will also be a 3D array
        of the same shape.
        The labels of bigger clusters are uneffected by the re-labeling.

    Raises:
        ValueError:
            If the input descriptor array does not have 2 or 3 dimensions,
            an error is raised.

    Example:

        .. code-block:: python

            from dynsight.data_processing import cleaning_cluster_population
            import numpy as np

            original_labels = np.load('labels_array.npy')

            cleaned_labels = cleaning_cluster_population(
                labels=original_labels,
                threshold=0.1,
                assigned_env=99,
            )

        In this example, the labels of the smaller clusters (lower than 10%)
        from `original_labels` are replaced with label 99. The result is
        stored in `cleaned_labels`, a NumPy array.
    """
    dimension = 2
    if labels.ndim < dimension or labels.ndim > dimension + 1:
        msg = "descriptor_array must be 2D or 3D."
        raise ValueError(msg)

    if excluded_env is None:
        excluded_arr: NDArray[np.int64] = np.array([], dtype=np.int64)
    elif isinstance(excluded_env, int):
        excluded_arr = np.array([excluded_env], dtype=np.int64)
    else:
        excluded_arr = np.array(excluded_env, dtype=np.int64)

    missing = np.setdiff1d(excluded_arr, np.unique(labels))

    if missing.size > 0:
        logger.log(f"Excluded value(s) not found in labels: {missing}")

    if labels.ndim == dimension:
        flat = labels.ravel()
        unique, counts = np.unique(flat, return_counts=True)

        populations = counts / flat.size
        small_clusters = unique[populations <= threshold]

        small_clusters = small_clusters[~np.isin(small_clusters, excluded_arr)]

        new_labels = labels.copy()
        if small_clusters.size > 0:
            new_labels[np.isin(labels, small_clusters)] = assigned_env

    if labels.ndim == dimension + 1:
        new_labels = labels.copy()
        for i in range(labels.shape[2]):
            lab = labels[:, :, i]
            flat = lab.ravel()
            unique, counts = np.unique(flat, return_counts=True)

            populations = counts / flat.size
            small_clusters = unique[populations <= threshold]

            small_clusters = small_clusters[
                ~np.isin(small_clusters, excluded_arr)
            ]

            if small_clusters.size > 0:
                mask = np.isin(lab, small_clusters)
                new_labels[:, :, i][mask] = assigned_env

    return new_labels
