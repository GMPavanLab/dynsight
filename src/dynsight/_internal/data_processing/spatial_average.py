from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import MDAnalysis

import ctypes
from multiprocessing import Array, Pool
from pathlib import Path

import numpy as np
from MDAnalysis.analysis.distances import distance_array

array: np.ndarray[float, Any]


def initworker(
    shared_array: np.ndarray[float, Any], shape: int, dtype: tuple[float, Any]
) -> None:
    # The use of global statement is necessary for the correct functioning
    # of the code. Ruff error PLW0603 is therefore ignored.
    global array  # noqa: PLW0603
    array = np.frombuffer(shared_array, dtype=dtype).reshape(shape)


def processframe(args: Any) -> tuple[int, np.ndarray[float, Any]]:
    universe, selection, cutoff, frame, vector = args
    universe.trajectory[frame]
    distances = distance_array(
        selection.positions, selection.positions, box=universe.dimensions
    )
    atom_id = np.argsort(distances, axis=1)
    nn = np.sum(distances < cutoff, axis=1)

    rows = np.arange(distances.shape[0])
    sp_dict = {row: atom_id[row, : nn[row]] for row in rows}

    if vector:
        sp_array_frame = np.zeros((array.shape[0], array.shape[2]))
        for key, value in sp_dict.items():
            if len(value) == 0:
                continue
            sp_array_frame[key, :] = np.mean(array[value, frame, :], axis=0)
    else:
        sp_array_frame = np.zeros(array.shape[0])
        for key, value in sp_dict.items():
            if len(value) == 0:
                continue
            sp_array_frame[key] = np.mean(array[value, frame])

    return frame, sp_array_frame


def spatialaverage(
    universe: MDAnalysis.Universe,
    array_path: Path,
    selection: str,
    cutoff: float,
    traj_cut: int = 0,
    num_processes: int = 4,
) -> np.ndarray[float, Any]:
    """Perform bla bla bla.

    Parameters:
        universe (MDAnalysis.Universe):
            ciao come stai?
    """
    selection = universe.select_atoms(selection)
    array = np.load(Path(array_path))

    shape = array.shape
    dtype = array.dtype
    shared_array = Array(ctypes.c_double, array.size, lock=False)
    # np.frombuffer type is 'buffer-like', which contains 'Array'.
    # The mypy error [call-overload] is ignored as it is considered
    # not significant.
    shared_array_np = np.frombuffer(shared_array, dtype=dtype).reshape(shape)  # type: ignore[call-overload]

    np.copyto(shared_array_np, array)
    two_dim = 2
    three_dim = 3
    if array.ndim == two_dim:
        sp_array = np.zeros((array.shape[0], array.shape[1]))
        vector = False
    elif array.ndim == three_dim:
        sp_array = np.zeros((array.shape[0], array.shape[1], array.shape[2]))
        vector = True
    else:
        error_string = "INVALID ARRAY SHAPE"
        raise ValueError(error_string)

    num_frames = len(universe.trajectory) - traj_cut
    pool = Pool(
        processes=num_processes,
        initializer=initworker,
        initargs=(shared_array, shape, dtype),
    )
    args = [
        (universe, selection, cutoff, frame, vector)
        for frame in range(num_frames)
    ]
    results = pool.map(processframe, args)
    pool.close()
    pool.join()
    for frame, sp_array_frame in results:
        if vector:
            sp_array[:, frame, :] = sp_array_frame
        else:
            sp_array[:, frame] = sp_array_frame
    return sp_array
