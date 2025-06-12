from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from MDAnalysis import Universe
    from MDAnalysis.core.groups import AtomGroup
    from numpy.typing import NDArray

import ctypes
from collections import defaultdict
from multiprocessing import Array, Pool

import numpy as np
from MDAnalysis.lib.nsgrid import FastNS

array: NDArray[np.float64]


def initworker(
    shared_arr: NDArray[np.float64],
    shape: int,
    dtype: tuple[float, Any],
) -> None:
    # The use of global statement is necessary for the correct functioning
    # of the code. Ruff error PLW0603 is therefore ignored.
    global array  # noqa: PLW0603
    array = np.frombuffer(shared_arr, dtype=dtype).reshape(shape)


def processframe(
    args: tuple[Universe, AtomGroup, float, int, int, bool],
) -> tuple[int, NDArray[np.float64]]:
    universe, selection, cutoff, traj_frame, result_index, is_vector = args

    universe.trajectory[traj_frame]
    positions = selection.positions
    box = universe.trajectory.ts.dimensions

    ns = FastNS(cutoff, positions, box=box)
    pairs = ns.self_search().get_pairs()

    neighbor_dict = defaultdict(list)
    for i, j in pairs:
        neighbor_dict[i].append(j)
        neighbor_dict[j].append(i)

    n_atoms = len(selection)
    if is_vector:
        n_features = array.shape[2]
        sp_array_frame1 = np.zeros((n_atoms, n_features), dtype=array.dtype)
        for key in range(n_atoms):
            neighbors = neighbor_dict[key] + [key]
            sp_array_frame1[key] = np.mean(
                array[neighbors, result_index, :], axis=0
            )
        return result_index, sp_array_frame1
    sp_array_frame2 = np.zeros(n_atoms, dtype=array.dtype)
    for key in range(n_atoms):
        neighbors = neighbor_dict[key] + [key]
        sp_array_frame2[key] = np.mean(array[neighbors, result_index])

    return result_index, sp_array_frame2


def spatialaverage(
    universe: Universe,
    descriptor_array: NDArray[np.float64],
    selection: str,
    cutoff: float,
    trajslice: slice | None = None,
    num_processes: int = 1,
) -> NDArray[np.float64]:
    """Compute spatially averaged descriptor values over neighboring particles.

    This function takes a molecular dynamics (MD) simulation stored in an
    `MDAnalysis.Universe` object and a NumPy array of descriptor values (such
    as a physical property for each particle in each frame of the simulation).
    For each particle in the system, the function calculates the average of
    its descriptor values with the descriptor values of its neighboring
    particles within a specified cutoff radius. The calculation is parallelized
    across multiple processes for efficiency.

    .. caution::
        This function utilizes multiprocessing and **must** be called from
        within a `main()` function. To avoid runtime errors, ensure that the
        script includes the following guard:

        .. code-block:: python

            if __name__ == "__main__":
                main()

        Failure to follow this structure may result in unexpected behavior
        or crashes, especially on Windows and MacOS.

    .. important::
        - Supports both scalar descriptors (2D) and vector descriptors (3D).
        - Utilizes multiprocessing to speed up the computation.

    Parameters:
        universe:
            The MDAnalysis `Universe` object containing the molecular dynamics
            simulation data, including atom positions and trajectory.
        descriptor_array:
            NumPy array containing the descriptor values.
            The array should have dimensions corresponding
            to either (n_atoms, n_frames) for scalar descriptors,
            or (n_atoms, n_frames, n_features) for vector descriptors.
        selection:
            An atom selection string compatible with MDAnalysis. This defines
            the subset of atoms for which the spatial averaging
            will be computed.
        cutoff:
            The distance cutoff (in the same units as the trajectory) that
            defines the neighborhood radius within which particles are
            considered as neighbors.
        traj_cut:
            The number of frames to exclude from the end of the trajectory.
        num_processes:
            The number of processes to use for parallel computation.
            **Warning:** Adjust this based on the available cores.

    Returns:
        A NumPy array of the same shape as the input descriptor array,
        containing the spatially averaged descriptor values. If the input
        array is 2D (n_atoms, n_frames), the output will be a 2D array of
        the same shape with spatially averaged values.
        Otherwise, if the input is 3D (n_atoms, n_frames, n_features),
        the output will also be a 3D array of the same shape with averaged
        vector values.

    Raises:
        ValueError:
            If the input descriptor array does not have 2 or 3 dimensions,
            an error is raised.

    Example:

        .. code-block:: python

            from dynsight.analysis import spatialaverage
            import numpy as np

            u = MDAnalysis.Universe('topology.gro', 'trajectory.xtc')
            descriptor = np.load('descriptor_array.npy')

            averaged_values = spatialaverage(
                universe=u,
                descriptor_array=descriptor,
                selection='name CA',
                cutoff=5.0,
                num_processes=8)

        This example computes the spatial averages of the descriptor values
        for atoms selected as `CA` atoms, within a cutoff of 5.0 units, using 8
        processes in parallel. The result is stored in `averaged_values`, a
        NumPy array. All supported input file formats by MDAnalysis can be used
        to set up the Universe.
    """
    selection = universe.select_atoms(selection)
    if trajslice is None:
        trajslice = slice(None)

    shape = descriptor_array.shape
    dtype = descriptor_array.dtype
    shared_array = Array(ctypes.c_double, descriptor_array.size, lock=False)
    # np.frombuffer type is 'buffer-like', which contains 'Array'.
    # The mypy error [call-overload] is ignored as it is considered
    # not significant.
    shared_array_np = np.frombuffer(shared_array, dtype=dtype).reshape(shape)  # type: ignore[call-overload]

    np.copyto(shared_array_np, descriptor_array)

    two_dim = 2
    three_dim = 3
    sp_array = np.zeros(descriptor_array.shape)
    if descriptor_array.ndim == two_dim:
        is_vector = False
    elif descriptor_array.ndim == three_dim:
        is_vector = True
    else:
        msg = "descriptor_array must have ndim == 2 or ndim == 3."
        raise ValueError(msg)

    pool = Pool(
        processes=num_processes,
        initializer=initworker,
        initargs=(shared_array, shape, dtype),
    )

    frame_indices = list(
        range(*trajslice.indices(universe.trajectory.n_frames))
    )
    args = [
        (universe, selection, cutoff, traj_frame, i, is_vector)
        for i, traj_frame in enumerate(frame_indices)
    ]
    results = pool.map(processframe, args)
    pool.close()
    pool.join()

    if is_vector:
        for frame, sp_array_frame in results:
            sp_array[:, frame, :] = sp_array_frame
    else:
        for frame, sp_array_frame in results:
            sp_array[:, frame] = sp_array_frame

    return sp_array
