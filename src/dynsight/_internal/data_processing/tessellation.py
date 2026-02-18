from __future__ import annotations

from itertools import combinations
from typing import TYPE_CHECKING

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import Delaunay

if TYPE_CHECKING:
    from numpy.typing import NDArray


class Tessellate:
    def __init__(
        self,
        vertices_file: str,
        radii_file: str,
        r_min: float = 0.01,
        r_max: float = 10.0,
        xyzcols: tuple[int, int] = (4, 7),
        target_volume: float = 5.6,
    ) -> None:
        """Class for computing cavities between spherical molecular entities.

        This class takes the XYZ coordinates of tangent sphere centers and
        their radii, performs a Delaunay triangulation of the convex hull,
        filters tetrahedra by a maximum circumsphere radius (alpha), and
        merges connected tetrahedra into cavities.
        It computes geometric properties of the cavities, including volume,
        centroid coordinates, surface area, and effective spherical radius.

        Parameters:
            vertices_file: str
                Path to the vertices file.

            radii_file: str
                Path to the radii file.

            r_min: float
                Minimum radius to consider (default 0.01).

            r_max: float
                Maximum radius to consider (default 10.0).

            xyzcols: tuple
                Columns to use from the vertices_file (default (4,7)).

            target_volume: float
                Target volume for voids (default 5.6).
        """
        self.vertices_file = vertices_file
        self.radii_file = radii_file
        self.r_min = r_min
        self.r_max = r_max
        self.xyzcols = xyzcols
        self.target_volume = target_volume

        # Data
        self.centers: NDArray[np.float64]
        self.radii: NDArray[np.float64]

        # Delaunay-related
        self.delaunay: Delaunay
        self.simplices: NDArray[np.float64]
        self.tetra_pts: NDArray[np.float64]
        self.circ_radii: NDArray[np.float64]
        self.tetra_volumes: NDArray[np.float64]

        # Alpha shape
        self.kept_tetra: NDArray[np.float64] | None = None
        self.kept_tetra_pts: NDArray[np.float64] | None = None

        # Cavities
        self.labels = None
        self.n_cavities = None
        self.cavity_volumes: NDArray[np.float64] | None = None
        self.cavity_centroids: NDArray[np.float64] | None = None
        self.cavity_areas: NDArray[np.float64] | None = None
        self.cavity_radii: NDArray[np.float64] | None = None

        self._load_data()

    def _load_data(self) -> None:
        self.centers = np.loadtxt(
            self.vertices_file, usecols=range(self.xyzcols[0], self.xyzcols[1])
        )
        self.radii = np.loadtxt(self.radii_file)

        mask = (self.radii >= self.r_min) & (self.radii <= self.r_max)
        self.centers = self.centers[mask]
        self.radii = self.radii[mask]

    # ---------------------------
    # Geometry
    # ---------------------------

    def _tetra_circumradius(self, points: NDArray[np.float64]) -> float:

        a, b, c, d = points
        m = np.vstack([b - a, c - a, d - a]).T
        rhs = (
            np.array(
                [
                    np.dot(b, b) - np.dot(a, a),
                    np.dot(c, c) - np.dot(a, a),
                    np.dot(d, d) - np.dot(a, a),
                ]
            )
            / 2
        )
        try:
            center = np.linalg.solve(m, rhs)
            return float(np.linalg.norm(center - a))
        except np.linalg.LinAlgError:
            return np.inf

    def _tetra_volume(self, points: NDArray[np.float64]) -> float:
        a, b, c, d = points
        return float(abs(np.dot((a - d), np.cross(b - d, c - d))) / 6.0)

    def _triangle_area(self, points: NDArray[np.float64]) -> float:
        a, b, c = points
        return float(0.5 * np.linalg.norm(np.cross(b - a, c - a)))

    # ---------------------------
    # Delaunay
    # ---------------------------

    def _compute_delaunay(self) -> None:
        self.delaunay = Delaunay(self.centers)
        self.simplices = self.delaunay.simplices
        self.tetra_pts = self.centers[self.simplices]

        self.circ_radii = np.array(
            [self._tetra_circumradius(t) for t in self.tetra_pts]
        )

        self.tetra_volumes = np.array(
            [self._tetra_volume(t) for t in self.tetra_pts]
        )

    def compute_delaunay(self) -> None:
        """Public wrapper."""
        self._compute_delaunay()

    # ---------------------------
    # Alpha optimization
    # ---------------------------

    def optimize_alpha(
        self,
        alpha_min: float = 0.5,
        alpha_max: int = 30,
        threshold: float = 1e-5,
        max_iter: int = 200,
    ) -> None:

        if self.delaunay is None:
            errormsg = "Call compute_delaunay() first."
            raise RuntimeError(errormsg)

        low, high = float(alpha_min), float(alpha_max)

        alphas_tried = []
        max_volumes = []
        optimal_alpha = None
        # break loop if two consecutive alpha values differ by breakcond
        breakcond = 1e-6
        for _ in range(max_iter):
            alpha = 0.5 * (low + high)
            mask = self.circ_radii <= alpha

            max_vol = (
                float(np.max(self.tetra_volumes[mask]))
                if np.any(mask)
                else 0.0
            )

            alphas_tried.append(alpha)
            max_volumes.append(max_vol)

            if abs(max_vol - self.target_volume) <= threshold:
                optimal_alpha = alpha
                break

            if max_vol < self.target_volume:
                low = alpha
            else:
                high = alpha
            # break loop if two consecutive alpha values differ by breakcond
            if high - low < breakcond:
                break

        if optimal_alpha is None and len(alphas_tried) > 0:
            idx = int(
                np.argmin(np.abs(np.array(max_volumes) - self.target_volume))
            )
            optimal_alpha = float(alphas_tried[idx])
        if optimal_alpha is None:
            errormsg = "optimal_alpha should have been set by now"
            raise ValueError(errormsg)
        self.alpha = optimal_alpha
        self.alpha_history = alphas_tried
        self.volume_history = max_volumes

    # ---------------------------
    # Alpha filtering
    # ---------------------------

    def _filter_tetrahedra(self) -> None:
        keep_mask = self.circ_radii <= self.alpha
        self.kept_tetra = self.simplices[keep_mask]
        self.kept_tetra_pts = self.centers[self.kept_tetra]

    def build_alpha_shape(self) -> None:
        """Public alpha-shape construction."""
        if self.delaunay is None:
            errormsg = "Call compute_delaunay() first."
            raise RuntimeError(errormsg)

        self._filter_tetrahedra()

    # ---------------------------
    # Cavities
    # ---------------------------

    def compute_cavities(self) -> None:

        if self.kept_tetra is None:
            errormsg = "Call build_alpha_shape() first."
            raise RuntimeError(errormsg)

        face_dict: dict[tuple[int, int, int], list[int]] = {}

        for tidx, tet in enumerate(self.kept_tetra):
            for face in combinations(sorted(tet), 3):
                face_dict.setdefault(face, []).append(tidx)

        adjacency = np.zeros(
            (len(self.kept_tetra), len(self.kept_tetra)), dtype=bool
        )
        mintetra_faces = 2  # exactly two tetra are paired if they share faces
        for tlist in face_dict.values():
            if len(tlist) == mintetra_faces:
                i, j = tlist
                adjacency[i, j] = adjacency[j, i] = True

        graph = csr_matrix(adjacency)
        n_components, labels = connected_components(
            csgraph=graph, directed=False
        )

        self.labels = labels
        self.n_cavities = n_components

        self.cavity_volumes = np.zeros(n_components)
        self.cavity_centroids = np.zeros((n_components, 3))
        self.cavity_areas = np.zeros(n_components)
        self.cavity_radii = np.zeros(n_components)

        for c in range(n_components):
            mask = labels == c
            tets = self.kept_tetra[mask]
            t_pts = self.centers[tets]

            self.cavity_volumes[c] = np.sum(
                [self._tetra_volume(t) for t in t_pts]
            )

            face_count: dict[tuple[int, int, int], int] = {}
            for tet in tets:
                for face in combinations(sorted(tet), 3):
                    face_count[face] = face_count.get(face, 0) + 1

            boundary_faces = [
                f for f, count in face_count.items() if count == 1
            ]

            self.cavity_areas[c] = np.sum(
                [
                    self._triangle_area(self.centers[list(f)])
                    for f in boundary_faces
                ]
            )

            self.cavity_radii[c] = (
                3 * self.cavity_volumes[c] / (4 * np.pi)
            ) ** (1 / 3)

            self.cavity_centroids[c] = np.mean(t_pts.reshape(-1, 3), axis=0)
