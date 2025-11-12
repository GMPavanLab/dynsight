"""Compute LENS for each atom in a trajectory."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from MDAnalysis import AtomGroup, Universe
    from numpy.typing import NDArray

from multiprocessing import Pool

import numba
import numpy as np
from numba import njit, prange


@njit(cache=True, fastmath=True)  # type: ignore[misc]
def _pbc_diff(
    dx: NDArray[np.float64],
    box: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Find distances in PBC box."""
    for k in range(3):
        if box[k] > 0.0:
            dx[k] -= np.rint(dx[k] / box[k]) * box[k]
    return dx


@njit(cache=True, fastmath=True)  # type: ignore[misc]
def build_cell_list(
    positions: NDArray[np.float64],
    box: NDArray[np.float64],
    cell_size: float,
) -> tuple[
    NDArray[np.int32],
    NDArray[np.int32],
    NDArray[np.int32],
    NDArray[np.int32],
]:
    """Build a 3D periodic cell list."""
    n_atoms = positions.shape[0]
    min_cells = 3
    ncellx = max(min_cells, int(box[0] // cell_size))
    ncelly = max(min_cells, int(box[1] // cell_size))
    ncellz = max(min_cells, int(box[2] // cell_size))
    n_cells = ncellx * ncelly * ncellz

    head = np.full(n_cells, -1, dtype=np.int32)
    next_ = np.full(n_atoms, -1, dtype=np.int32)
    cell_ids = np.empty((n_atoms, 3), dtype=np.int32)

    for i in range(n_atoms):
        cx = int(positions[i, 0] / box[0] * ncellx) % ncellx
        cy = int(positions[i, 1] / box[1] * ncelly) % ncelly
        cz = int(positions[i, 2] / box[2] * ncellz) % ncellz
        cell_ids[i, 0] = cx
        cell_ids[i, 1] = cy
        cell_ids[i, 2] = cz
        cindex = cx * ncelly * ncellz + cy * ncellz + cz
        next_[i] = head[cindex]
        head[cindex] = i
    n_cell = np.array([ncellx, ncelly, ncellz])

    return cell_ids, head, next_, n_cell


# We need a function this complex and deep for numba to work
# This is why we are ignoring ruff complaints C901, PLR0912
@njit(cache=True, fastmath=True, parallel=True)  # type: ignore[misc]
def neighbor_list_celllist_centers(  # noqa: C901, PLR0912
    positions_env: NDArray[np.float64],
    positions_cent: NDArray[np.float64],
    r_cut: float,
    box: NDArray[np.float64],
    respect_pbc: bool,
) -> tuple[NDArray[np.int32], NDArray[np.int32]]:
    """Build a CSR neighbor list *only for the centers*."""
    n_cent = positions_cent.shape[0]
    r_cut2 = (r_cut - 1e-6) ** 2

    cell_ids, head, next_, n_cell = build_cell_list(positions_env, box, r_cut)
    nx, ny, nz = n_cell

    n_neigh = np.zeros(n_cent, dtype=np.int32)
    # ---- count the neighbors for each center ----
    for i in prange(n_cent):
        cx = cell_ids[i, 0]  # int(positions_cent[i, 0] / box[0] * nx) % nx
        cy = cell_ids[i, 1]  # int(positions_cent[i, 1] / box[1] * ny) % ny
        cz = cell_ids[i, 2]  # int(positions_cent[i, 2] / box[2] * nz) % nz

        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    nx_ = (cx + dx) % nx
                    ny_ = (cy + dy) % ny
                    nz_ = (cz + dz) % nz
                    cidx = nx_ * ny * nz + ny_ * nz + nz_
                    j = head[cidx]
                    while j != -1:
                        dr = positions_env[j] - positions_cent[i]
                        if respect_pbc:
                            dr = _pbc_diff(dr, box)
                        dr2 = dr[0] ** 2 + dr[1] ** 2 + dr[2] ** 2
                        if j != i and dr2 < r_cut2:
                            n_neigh[i] += 1
                        j = next_[j]

    indptr = np.empty(n_cent + 1, dtype=np.int32)
    indptr[0] = 0
    for i in range(n_cent):
        indptr[i + 1] = indptr[i] + n_neigh[i]

    indices = np.empty(indptr[-1], dtype=np.int32)
    cursor = np.zeros(n_cent, dtype=np.int32)

    # ---- fill up neighbors' lists ----
    for i in prange(n_cent):
        cx = int(positions_cent[i, 0] / box[0] * nx) % nx
        cy = int(positions_cent[i, 1] / box[1] * ny) % ny
        cz = int(positions_cent[i, 2] / box[2] * nz) % nz

        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    nx_ = (cx + dx) % nx
                    ny_ = (cy + dy) % ny
                    nz_ = (cz + dz) % nz
                    cidx = nx_ * ny * nz + ny_ * nz + nz_
                    j = head[cidx]
                    while j != -1:
                        dr = positions_env[j] - positions_cent[i]
                        dr = _pbc_diff(dr, box)
                        if (dr[0] ** 2 + dr[1] ** 2 + dr[2] ** 2) < r_cut2:
                            pos_i = indptr[i] + cursor[i]
                            indices[pos_i] = j
                            cursor[i] += 1
                        j = next_[j]

    # sort each list (needed for computing intersections)
    for i in range(n_cent):
        s, e = indptr[i], indptr[i + 1]
        if e > s + 1:
            indices[s:e].sort()

    return indptr, indices


@njit(cache=True, fastmath=True)  # type: ignore[misc]
def lens_from_two_csr(
    indptr1: NDArray[np.int32],
    indices1: NDArray[np.int32],
    indptr2: NDArray[np.int32],
    indices2: NDArray[np.int32],
) -> NDArray[np.float64]:
    """Return LENS distance between two neighbor lists.

    Note: CSR lists do NOT include the particle istelf.
    """
    n_centers = len(indptr1) - 1
    out = np.zeros(n_centers, dtype=np.float64)
    for u in range(n_centers):
        s1, e1 = indptr1[u], indptr1[u + 1]
        s2, e2 = indptr2[u], indptr2[u + 1]
        a = e1 - s1
        b = e2 - s2
        denom = a + b  # -2
        if denom <= 0:
            out[u] = 0.0
            continue

        i, j = s1, s2
        inter = 0
        while i < e1 and j < e2:
            vi = indices1[i]
            vj = indices2[j]
            if vi == vj:
                inter += 1
                i += 1
                j += 1
            elif vi < vj:
                i += 1
            else:
                j += 1

        numer = (a + b) - 2 * inter
        out[u] = numer / denom
    return out


def compute_lens_over_trj(
    universe: Universe,
    r_cut: float,
    delay: int = 1,
    centers: str = "all",
    selection: str = "all",
    trajslice: slice | None = None,
    respect_pbc: bool = True,
    n_jobs: int = 1,
) -> tuple[NDArray[np.float64], list[tuple[int, int]], AtomGroup]:
    r"""Compute LENS over a trajectory.

    LENS was developed by Martina Crippa. See for reference the paper
    https://doi.org/10.1073/pnas.2300565120.
    The current implementation is mainly due to @SimoneMartino98.

    The LENS value of a particle between two frames is deined as:

    .. math::
        LENS(t, t + \\delta t) =
        \frac{|C(t)\\cup C(t+\\delta t)| - |C(t)\\cap C(t+\\delta t)|}
        {|C(t)| + |C(t+\\delta t|}

    where C(t) and C(t+\\delta t) are the neighbors' list of the particle
    at frames t and t+\\delta t.

    Parameters:
        universe : mda.Universe
            MDAnalysis Universe containing the trajectory.
        r_cut : float
            r_cut distance (Å) for defining neighbors.
        selection : str, optional
            Atom selection string defining the environment (default "all").
        centers : str, optional
            Atom selection string for the centers where LENS is computed.
        delay : int, optional
            Number of frames separating the pairs for comparison.
        start, stop, step : int or None, optional
            Frame slicing parameters for trajectory iteration.
        use_pbc : bool, optional
            Whether to apply periodic boundary conditions.

    Returns:
        tuple
            lens_array : (n_centers, n_pairs) NDArray[np.float64]
                LENS values for each pair of frames and each center.
            pairs : list[tuple[int, int]]
                Frame index pairs used for comparison.
            ag_cent : mda.AtomGroup
                The AtomGroup corresponding to the centers selection.
    """
    numba.set_num_threads(n_jobs)  # Not sure this works

    ag_env = universe.select_atoms(selection)
    ag_cent = universe.select_atoms(centers)

    if trajslice is not None:
        fr_idx = list(range(universe.trajectory.n_frames))[trajslice]
    else:
        fr_idx = list(range(universe.trajectory.n_frames))
    pairs = [
        (fr_idx[i], fr_idx[i + delay]) for i in range(len(fr_idx) - delay)
    ]
    if not pairs:
        msg = "No valid pairs found."
        raise RuntimeError(msg)

    lens_array = np.zeros((ag_cent.n_atoms, len(pairs)), dtype=np.float64)
    for k, (t1, t2) in enumerate(pairs):
        # ---- frame t1 ----
        universe.trajectory[t1]
        pos_env1 = ag_env.positions.astype(np.float64)
        pos_cent1 = ag_cent.positions.astype(np.float64)
        if respect_pbc and universe.trajectory.ts.dimensions is not None:
            box = universe.trajectory.ts.dimensions[:3]
        else:
            coords = universe.atoms.positions
            mins = coords.min(axis=0)
            maxs = coords.max(axis=0)
            box = (maxs - mins) * 1.01
        indptr_t1, indices_t1 = neighbor_list_celllist_centers(
            positions_env=pos_env1,
            positions_cent=pos_cent1,
            r_cut=r_cut,
            box=box,
            respect_pbc=respect_pbc,
        )

        # ---- frame t2 ----
        universe.trajectory[t2]
        pos_env2 = ag_env.positions.astype(np.float64)
        pos_cent2 = ag_cent.positions.astype(np.float64)
        if respect_pbc and universe.trajectory.ts.dimensions is not None:
            box = universe.trajectory.ts.dimensions[:3]
        else:
            coords = universe.atoms.positions
            mins = coords.min(axis=0)
            maxs = coords.max(axis=0)
            box = (maxs - mins) * 1.01
        indptr_t2, indices_t2 = neighbor_list_celllist_centers(
            positions_env=pos_env2,
            positions_cent=pos_cent2,
            r_cut=r_cut,
            box=box,
            respect_pbc=respect_pbc,
        )

        # ---- LENS ----
        lens_array[:, k] = lens_from_two_csr(
            indptr1=indptr_t1,
            indices1=indices_t1,
            indptr2=indptr_t2,
            indices2=indices_t2,
        )

    return lens_array, pairs, ag_cent


def list_neighbours_along_trajectory(
    universe: Universe,
    r_cut: float,
    centers: str = "all",
    selection: str = "all",
    trajslice: slice | None = None,
    respect_pbc: bool = True,
    n_jobs: int = 1,
) -> list[list[AtomGroup]]:
    """Produce a per-frame list of neighbors.

    Parameters:
        universe : mda.Universe
            The Universe containing the trajectory.
        r_cut : float
            Maximum neighbor distance.
        selection : str
            Atom selection string.
        trajslice : slice | None
            Slice of trajectory to consider. Defaults to full trajectory.
        n_jobs : int
            Number of processes for parallel computation.

    Returns:
        list[list[AtomGroup]]
            List of frames, each frame a list of AtomGroups for each atom.
    """
    if trajslice is None:
        trajslice = slice(None)
    center_atoms = universe.select_atoms(centers)
    selected_atoms = universe.select_atoms(selection)
    n_selected = selected_atoms.n_atoms

    frame_indices = list(
        range(*trajslice.indices(universe.trajectory.n_frames))
    )

    def _compute_frame_neighbors(frame_idx: int) -> list[AtomGroup]:
        universe.trajectory[frame_idx]
        env_positions = selected_atoms.positions.astype(np.float64)
        centers_positions = center_atoms.positions.astype(np.float64)
        if universe.trajectory.ts.dimensions is not None:
            box = universe.trajectory.ts.dimensions[:3]
        else:
            coords = universe.atoms.positions
            mins = coords.min(axis=0)
            maxs = coords.max(axis=0)
            box = maxs - mins

        # Compute neighbor lists using the new fast functions
        indptr, indices = neighbor_list_celllist_centers(
            positions_env=env_positions,
            positions_cent=centers_positions,
            r_cut=r_cut,
            box=box,
            respect_pbc=respect_pbc,
        )

        # Build the AtomGroups for each atom
        frame_neighbors: list[AtomGroup] = []
        for i in range(n_selected):
            start, end = indptr[i], indptr[i + 1]
            neighbor_atoms = universe.atoms[indices[start:end]]
            frame_neighbors.append(neighbor_atoms)
        return frame_neighbors

    if n_jobs == 1:
        return [_compute_frame_neighbors(frame) for frame in frame_indices]

    # Parallel computation
    args = frame_indices
    with Pool(processes=n_jobs) as pool:
        return pool.map(_compute_frame_neighbors, args)
