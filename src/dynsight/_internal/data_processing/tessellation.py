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
        r_min: float,
        r_max: float,
        target_volume: float,
        xyzcols: tuple[int, int],
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
                Minimum radius of the tangent spheres to consider.

            r_max: float
                Maximum radius of the tangent spheres to consider.

            xyzcols: tuple
                Columns to use from the vertices_file (default (4,7)).

            target_volume: float
                Maximum allowed volume of the sphere circumscribing
                each tetrahedron.
                It is the largest volume of a probe particle
                that can fit in that tetrahedron's circumscribed sphere.
        """
        self.vertices_file = vertices_file
        self.radii_file = radii_file
        self.r_min: float = r_min
        self.r_max: float = r_max
        self.target_volume: float = target_volume

        self.xyzcols = xyzcols

        # Core data structures for alpha-shape cavity analysis:
        # - Raw filtered tangent sphere data: centers, radii
        # - Delaunay triangulation data: delaunay, simplices, tetra_pts
        # - Per-tetrahedron geometry: circ_radii, tetra_volumes
        # - Alpha-filtered tetrahedra: kept_tetra, kept_tetra_pts
        # - Merged cavity properties: labels, n_cavities,
        #   cavity_volumes, cavity_centroids, cavity_areas, cavity_radii

        self.centers: NDArray[np.float64]
        self.radii: NDArray[np.float64]
        self.delaunay: Delaunay
        self.simplices: NDArray[np.float64]
        self.tetra_pts: NDArray[np.float64]
        self.circ_radii: NDArray[np.float64]
        self.tetra_volumes: NDArray[np.float64]


        self.kept_tetra: NDArray[np.float64] | None = None
        self.kept_tetra_pts: NDArray[np.float64] | None = None

        self.labels = None
        self.n_cavities = None
        self.cavity_volumes: NDArray[np.float64] | None = None
        self.cavity_centroids: NDArray[np.float64] | None = None
        self.cavity_areas: NDArray[np.float64] | None = None
        self.cavity_radii: NDArray[np.float64] | None = None

        self._load_data()

    def _load_data(self) -> None:
        """Function to initialize the data and prepare the variables."""
        self.centers = np.loadtxt(
            self.vertices_file, usecols=range(self.xyzcols[0], self.xyzcols[1])
        )
        self.radii = np.loadtxt(self.radii_file)
        mask = (self.radii >= self.r_min) & (self.radii <= self.r_max)
        self.centers = self.centers[mask]
        self.radii = self.radii[mask]

    def _tetra_circumradius(self, points: NDArray[np.float64]) -> float:
        """Compute the circumradius of a tetrahedron.

        Given four vertices a,b,c,d of a tetrahedron,
        this function finds the radius of the unique circumsphere that passes
        through all four points.

        Mathematical idea:
        1. The circumsphere center r satisfies the condition that it is
        equidistant from all four vertices: |r - a|^2 = |r - b|^2 = |r - c|^2 =
        |r - d|^2 = R^2 where R is the circumradius.
        2. Subtracting the first equation from the others eliminates R^2
        and yields three linear equations in the unknown center vector r:
        (b - a) ⋅ r = 1/2 (|b|^2 - |a|^2)
        (c - a) ⋅ r = 1/2 (|c|^2 - |a|^2)
        (d - a) ⋅ r = 1/2 (|d|^2 - |a|^2)
        3. These three dot-product equations can be written compactly in matrix
        form: A r = rhs
        where A is a 3x3 matrix whose rows are the vectors (b - a)^T, (c - a)^T
        , (d - a)^T,
        and rhs is the vector of the corresponding 1/2 * (|p|^2 - |a|^2) terms.
        4. Solving this linear system gives the coordinates of the circumsphere
        center r.
        5. The circumradius is then simply the distance from r to any vertex.
        Special cases:
        - If the tetrahedron is degenerate (e.g., points are coplanar),
        the matrix A is singular, and no unique circumsphere exists.In that
        case, the function returns np.inf.

        Parameters:
            points : The 3D coordinates of the tetrahedron
            vertices: a, b, c, d.
        """
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
        """Compute the volume of a tetrahedron defined by four points in 3D.

        Mathematical idea:
        1. Let a, b, c, d be the vertices of the tetrahedron.
        2. The volume V of a tetrahedron can be computed from the
        scalar triple product: V = |(a - d) ⋅ ((b - d) x (c - d))| /6
        where: - (b - d) x (c - d) gives a vector perpendicular to
        the base triangle b-c-d - Dotting with (a - d) projects the
        height of the tetrahedron along this normal - Division by 6
        comes from the geometric formula V = (1/3)*base_area*height, simplified
        for a tetrahedron using vectors.
        3. The absolute value is used to ensure the volume is
        positive regardless of vertex ordering.

        Parameters:
            points : The 3D coordinates of the
            tetrahedron vertices: a, b, c, d.
        """
        a, b, c, d = points
        return float(abs(np.dot((a - d), np.cross(b - d, c - d))) / 6.0)

    def _triangle_area(self, points: NDArray[np.float64]) -> float:
        """Compute the area of a triangle defined by three points in 3D.

        Mathematical idea:
        1. Let a, b, c be the vertices of the triangle.
        2. The area can be computed using the cross product:
        Area = 0.5 * || (b - a) x (c - a) || where: - (b - a) x (c - a)
        produces a vector perpendicular to the triangle plane -
        The norm of this vector gives the area of the parallelogram spanned by
        (b - a) and (c - a) - Dividing by 2 gives the area of the triangle.
        3. This formula works for any orientation of the triangle in 3D space.

        Parameters:
            points : The 3D coordinates of the triangle vertices a, b, c.
        """
        a, b, c = points
        return float(0.5 * np.linalg.norm(np.cross(b - a, c - a)))

    # ---------------------------
    # Delaunay
    # ---------------------------

    def compute_delaunay(self) -> None:
        """Perform the Delaunay triangulation of the tangent spheres.

        It takes the tangent sphere centers and partitions them into
        non-overlapping tetrahedra.
        """
        # Delaunay triangulation and per-tetrahedron geometric properties:
        #Build triangulation from filtered sphere centers
        #Extract tetrahedral vertex indices and coordinates
        #Compute circumsphere radii and tetrahedron volumes
        self.delaunay = Delaunay(self.centers)
        self.simplices = self.delaunay.simplices
        self.tetra_pts = self.centers[self.simplices]

        self.circ_radii = np.array(
            [self._tetra_circumradius(t) for t in self.tetra_pts]
        )
        self.tetra_volumes = np.array(
            [self._tetra_volume(t) for t in self.tetra_pts]
        )



    def optimize_alpha(
        self,
        alpha_min: float,
        alpha_max: float,
        threshold: float,
        max_iter: int,
    ) -> None:
        """Optimize alpha to achieve a target maximum tetrahedron volume.

        This method performs search between alpha_min and alpha_max. For each
        alpha, tetrahedra are filtered by requiring their circumsphere radius
        is less than or equal to alpha.
        The maximum tetrahedron volume among the filtered tetrahedra is then
        compared to self.target_volume.

        Parameters:
            alpha_min : Minimum alpha value to try.
            alpha_max : Maximum alpha value to try.
            threshold : Allowed deviation from target volume
            max_iter : Maximum number of iterations.
        """
        if self.delaunay is None:
            msg = "Call compute_delaunay() first."
            raise RuntimeError(msg)

        low, high = float(alpha_min), float(alpha_max)

        alphas_tried = []
        max_volumes = []
        optimal_alpha = None

        breakcond = 1e-6
        # Binary search for optimal alpha:
        # - Iteratively bisect [low, high]
        # - Filter tetrahedra by circumsphere radius
        # - Track maximum retained tetra volume per alpha
        # - Stop if target volume is matched within threshold
        #   or interval width < break condition
        # - If no exact match, choose alpha minimizing |max_vol - target|
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
            if high - low < breakcond:
                break

        if optimal_alpha is None and len(alphas_tried) > 0:
            idx = int(
                np.argmin(np.abs(np.array(max_volumes) - self.target_volume))
            )
            optimal_alpha = float(alphas_tried[idx])
        if optimal_alpha is None:
            msg = "optimal_alpha should have been set by now"
            raise ValueError(msg)
        self.alpha = optimal_alpha
        self.alpha_history = alphas_tried
        self.volume_history = max_volumes


    def build_alpha_shape(self) -> None:
        """Public alpha-shape construction.

        The function must be called after compute_delaunay()
        and optimize_alpha() have been computed.
        It will filter the tetrahedra based on the optimum alpha value and
        populate kept_tetra and kept_tetra_pts
        """
        if self.delaunay is None:
            msg = "Call compute_delaunay() first."
            raise RuntimeError(msg)

        keep_mask = self.circ_radii <= self.alpha
        self.kept_tetra = self.simplices[keep_mask]
        self.kept_tetra_pts = self.centers[self.kept_tetra]

    def compute_cavities(self) -> None:
        """Identify connected tetrahedra from their alphashape.

        This method determines connected components of the tetrahedra stored in
        self.kept_tetra by detecting shared triangular faces. Two tetrahedra
        are considered adjacent if they share exactly one common face (i.e.,
        three common vertex indices). The resulting adjacency
        matrix is converted
        to a sparse graph and analyzed to label each connected set of
        tetrahedra as a cavity.
        """
        if self.kept_tetra is None:
            msg = "Call build_alpha_shape() first."
            raise RuntimeError(msg)

        face_dict: dict[tuple[int, int, int], list[int]] = {}

        for tidx, tet in enumerate(self.kept_tetra):
            for face in combinations(sorted(tet), 3):
                face_dict.setdefault(face, []).append(tidx)

        adjacency = np.zeros(
            (len(self.kept_tetra), len(self.kept_tetra)), dtype=bool
        )
        mintetra_faces = 2
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
