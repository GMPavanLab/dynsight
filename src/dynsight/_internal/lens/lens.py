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


@njit(cache=True, fastmath=True)  # type: ignore[decorator]
def _pbc_diff(
    dx: NDArray[np.float64],
    box: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Find distances in PBC box."""
    for k in range(3):
        if box[k] > 0.0:
            dx[k] -= np.rint(dx[k] / box[k]) * box[k]
    return dx


@njit(cache=True, fastmath=True)  # type: ignore[decorator]
def _build_cell_list(
    positions: NDArray[np.float64],
    box: NDArray[np.float64],
    cell_size: float,
) -> tuple[
    NDArray[np.int32],
    NDArray[np.int32],
    NDArray[np.int32],
    int,
    int,
    int,
    float,
]:
    """Build a 3D periodic cell list."""
    n_atoms = positions.shape[0]
    ncellx = max(1, int(box[0] // cell_size))
    ncelly = max(1, int(box[1] // cell_size))
    ncellz = max(1, int(box[2] // cell_size))
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

    return cell_ids, head, next_, ncellx, ncelly, ncellz, cell_size


@njit  # type: ignore[decorator]
def _neighbor_cells_loop(
    pos_env: NDArray[np.float64],
    pos_cent: NDArray[np.float64],
    head: NDArray[np.int32],
    next_: NDArray[np.int32],
    nx: int,
    ny: int,
    nz: int,
    box: NDArray[np.float64],
    cutoff2: float,
    mode: int,  # 0=count, 1=fill
    indptr: NDArray[np.int32],
    indices: NDArray[np.int32],
    cursor: NDArray[np.int32],
) -> NDArray[np.int32]:
    """Loop over neighboring cells to count or fill neighbors for center atoms.

    Parameters:
        pos_env : NDArray[np.float64]
            Positions of environment atoms.
        pos_cent : NDArray[np.float64]
            Positions of center atoms.
        head, next_ : NDArray[np.int32]
            Linked-list cell list arrays for environment atoms.
        nx, ny, nz : int
            Number of cells along each axis.
        box : NDArray[np.float64]
            Box dimensions (for periodic boundary conditions).
        cutoff2 : float
            Squared cutoff distance.
        mode : int
            0 to count neighbors, 1 to fill CSR arrays.
        indptr : NDArray[np.int32]
            CSR row pointers (required if mode=1).
        indices : NDArray[np.int32]
            CSR indices array (required if mode=1).
        cursor : NDArray[np.int32]
            CSR insertion cursor (required if mode=1).

    Returns:
        NDArray[np.int32]
            Array of neighbor counts per center (mode=0), or None (mode=1).
    """
    n_cent = pos_cent.shape[0]
    n_neigh = np.zeros(n_cent, dtype=np.int32)

    for i in prange(n_cent):
        cx = int(pos_cent[i, 0] / box[0] * nx) % nx
        cy = int(pos_cent[i, 1] / box[1] * ny) % ny
        cz = int(pos_cent[i, 2] / box[2] * nz) % nz

        for dx in range(-1, 2):
            for dy in range(-1, 2):
                for dz in range(-1, 2):
                    nx_ = np.int32((cx + dx) % nx)
                    ny_ = np.int32((cy + dy) % ny)
                    nz_ = np.int32((cz + dz) % nz)
                    cidx = nx_ * ny * nz + ny_ * nz + nz_
                    j = head[cidx]
                    while j != -1:
                        dr = pos_env[j] - pos_cent[i]
                        dr = _pbc_diff(dr, box)
                        if (dr[0] ** 2 + dr[1] ** 2 + dr[2] ** 2) < cutoff2:
                            if mode == 0:
                                n_neigh[i] += 1
                            else:
                                pos_i = indptr[i] + cursor[i]
                                indices[pos_i] = j
                                cursor[i] += 1
                        j = next_[j]
    return n_neigh


@njit(cache=True, fastmath=True, parallel=True)  # type: ignore[decorator]
def _neighbor_list_celllist_centers(
    positions_env: NDArray[np.float64],
    positions_cent: NDArray[np.float64],
    cell_size: float,
    box: NDArray[np.float64],
    batch_size: int = 10000,
) -> tuple[NDArray[np.int32], NDArray[np.int32]]:
    """Build a CSR-like neighbor list for center atoms using a cell list.

    Parameters:
        positions_env : NDArray[np.float64]
            Positions of environment atoms.
        positions_cent : NDArray[np.float64]
            Positions of center atoms.
        cell_size : float
            Nominal cell size for the cell list.
        box : NDArray[np.float64]
            Box dimensions (for periodic boundary conditions).
        batch_size : int, optional
            Number of centers to process per batch (default 10000).

    Returns:
        tuple[NDArray[np.int32], NDArray[np.int32]]
            indptr : CSR row pointer array for all centers.
            indices : Flattened neighbor indices for all centers.
    """
    # ---- build cell list for environment ----
    _, head, next_, nx, ny, nz, _ = _build_cell_list(
        positions=positions_env,
        box=box,
        cell_size=cell_size,
    )

    n_cent = positions_cent.shape[0]
    cutoff2 = (cell_size - 1e-6) ** 2

    # ---- storage for CSR ----
    all_indptr = np.empty(n_cent + 1, dtype=np.int32)
    all_indptr[0] = 0

    # Pre-allocate enough space for indices
    max_possible = n_cent * 200  # conservative estimate; resized later
    all_indices = np.empty(max_possible, dtype=np.int32)

    cursor_global = 0

    # ---- loop over center batches ----
    for batch_start in range(0, n_cent, batch_size):
        batch_end = min(batch_start + batch_size, n_cent)
        batch_cent = positions_cent[batch_start:batch_end]

        # ---- count neighbors ----
        n_neigh = _neighbor_cells_loop(
            positions_env,
            batch_cent,
            head,
            next_,
            nx,
            ny,
            nz,
            box,
            cutoff2,
            mode=0,
            indptr=np.empty(1, dtype=np.int32),
            indices=np.empty(1, dtype=np.int32),
            cursor=np.empty(1, dtype=np.int32),
        )

        # ---- build batch CSR ----
        batch_indptr = np.empty(len(n_neigh) + 1, dtype=np.int32)
        batch_indptr[0] = 0
        for i in range(len(n_neigh)):
            batch_indptr[i + 1] = batch_indptr[i] + n_neigh[i]

        batch_indices = np.empty(batch_indptr[-1], dtype=np.int32)
        batch_cursor = np.zeros(len(n_neigh), dtype=np.int32)

        # ---- fill batch indices ----
        _neighbor_cells_loop(
            positions_env,
            batch_cent,
            head,
            next_,
            nx,
            ny,
            nz,
            box,
            cutoff2,
            mode=1,
            indptr=batch_indptr,
            indices=batch_indices,
            cursor=batch_cursor,
        )

        # ---- append to global CSR ----
        for i in range(len(n_neigh)):
            all_indptr[batch_start + i + 1] = (
                all_indptr[batch_start + i] + n_neigh[i]
            )

        if cursor_global + len(batch_indices) > all_indices.size:
            new_size = int(all_indices.size * 1.5) + len(batch_indices)
            new_arr = np.empty(new_size, dtype=np.int32)
            new_arr[:cursor_global] = all_indices[:cursor_global]
            all_indices = new_arr

        all_indices[cursor_global : cursor_global + len(batch_indices)] = (
            batch_indices
        )
        cursor_global += len(batch_indices)

    # ---- trim and sort ----
    all_indices = all_indices[:cursor_global]
    for i in range(n_cent):
        s, e = all_indptr[i], all_indptr[i + 1]
        if e > s + 1:
            all_indices[s:e].sort()

    return all_indptr, all_indices


@njit(cache=True, fastmath=True)  # type: ignore[decorator]
def _lens_from_two_csr(
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
            Cutoff distance (Å) for defining neighbors.
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
            lens_mat : (n_centers, n_pairs) NDArray[np.float64]
                LENS values for each pair of frames and each center.
            pairs : list[tuple[int, int]]
                Frame index pairs used for comparison.
            ag_cent : mda.AtomGroup
                The AtomGroup corresponding to the centers selection.
    """
    numba.set_num_threads(n_jobs)
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

    lens_mat = np.zeros((len(pairs), ag_cent.n_atoms), dtype=np.float64)

    for k, (t1, t2) in enumerate(pairs):
        # ---- frame t1 ----
        universe.trajectory[t1]
        pos_env1 = ag_env.positions.astype(np.float64)
        pos_cent1 = ag_cent.positions.astype(np.float64)
        if respect_pbc and universe.trajectory.ts.dimensions is not None:
            box1 = universe.trajectory.ts.dimensions[:3]
        else:
            coords = universe.atoms.positions
            mins = coords.min(axis=0)
            maxs = coords.max(axis=0)
            box1 = maxs - mins
        csr1 = _neighbor_list_celllist_centers(
            pos_env1, pos_cent1, r_cut, box1
        )

        # ---- frame t2 ----
        universe.trajectory[t2]
        pos_env2 = ag_env.positions.astype(np.float64)
        pos_cent2 = ag_cent.positions.astype(np.float64)
        if respect_pbc and universe.trajectory.ts.dimensions is not None:
            box2 = universe.trajectory.ts.dimensions[:3]
        else:
            coords = universe.atoms.positions
            mins = coords.min(axis=0)
            maxs = coords.max(axis=0)
            box2 = maxs - mins
        csr2 = _neighbor_list_celllist_centers(
            pos_env2, pos_cent2, r_cut, box2
        )

        # ---- LENS ----
        lens_vec = _lens_from_two_csr(csr1[0], csr1[1], csr2[0], csr2[1])
        lens_mat[k, :] = lens_vec

    return lens_mat.T, pairs, ag_cent


def list_neighbours_along_trajectory(
    universe: Universe,
    r_cut: float,
    selection: str = "all",
    trajslice: slice | None = None,
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
    selected_atoms = universe.select_atoms(selection)
    n_selected = selected_atoms.n_atoms

    frame_indices = list(
        range(*trajslice.indices(universe.trajectory.n_frames))
    )

    def _compute_frame_neighbors(frame_idx: int) -> list[AtomGroup]:
        universe.trajectory[frame_idx]
        env_positions = universe.atoms.positions.astype(np.float64)
        sel_positions = selected_atoms.positions.astype(np.float64)
        if universe.trajectory.ts.dimensions is not None:
            box = universe.trajectory.ts.dimensions[:3]
        else:
            coords = universe.atoms.positions
            mins = coords.min(axis=0)
            maxs = coords.max(axis=0)
            box = maxs - mins

        # Compute neighbor lists using the new fast functions
        indptr, indices = _neighbor_list_celllist_centers(
            positions_env=env_positions,
            positions_cent=sel_positions,
            cell_size=r_cut,
            box=box,
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
