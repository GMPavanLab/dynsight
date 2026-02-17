import numpy as np
from itertools import combinations
from scipy.spatial import Delaunay
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components


class tessellate:
    def __init__(
        self,
        vertices_file: str,
        radii_file: str,
        r_min: float = 0.01,
        r_max: float = 10.0,
        xyzcols: tuple = (4, 7),
        target_volume: float = 5.6,
        alpha: float = None
    ):
        
        """
        Class for computing cavities between molecular entities represented as spheres.

        This class takes the XYZ coordinates of tangent sphere centers and their radii,
        performs a Delaunay triangulation of the convex hull, filters tetrahedra by a
        maximum circumsphere radius (alpha), and merges connected tetrahedra into cavities.
        It computes geometric properties of the cavities including volume, centroid,
        surface area, and effective spherical radius.

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
            alpha: float
                Alpha parameter for alpha shape (default None, auto-optimized).

        Attributes:
            centers: ndarray
                Filtered coordinates of sphere centers.
            radii: ndarray
                Filtered radii of spheres.
            delaunay: Delaunay object
                Computed Delaunay triangulation of centers.
            tetra_pts: ndarray
                Coordinates of tetrahedra vertices.
            simplices: ndarray
                Indices of points forming each tetrahedron.
            circ_radii: ndarray
                Circumsphere radius of each tetrahedron.
            tetra_volumes: ndarray
                Volume of each tetrahedron.
            kept_tetra: ndarray
                Indices of tetrahedra passing the alpha filter.
            labels: ndarray
                Cavity ID for each kept tetrahedron.
            n_cavities: int
                Total number of cavities.
            cavity_volumes: ndarray
                Volume of each cavity.
            cavity_centroids: ndarray
                Centroid of each cavity.
            cavity_areas: ndarray
                Surface area of each cavity.
            cavity_radii: ndarray
                Effective spherical radius of each cavity.
        """


        self.vertices_file = vertices_file
        self.radii_file = radii_file
        self.r_min = r_min
        self.r_max = r_max
        self.xyzcols = xyzcols
        self.target_volume = target_volume
        self.alpha = alpha

        # Data
        self.centers = None
        self.radii = None

        # Delaunay-related
        self.delaunay = None
        self.simplices = None
        self.tetra_pts = None
        self.circ_radii = None
        self.tetra_volumes = None

        # Alpha shape
        self.kept_tetra = None
        self.kept_tetra_pts = None

        # Cavities
        self.labels = None
        self.n_cavities = None
        self.cavity_volumes = None
        self.cavity_centroids = None
        self.cavity_areas = None
        self.cavity_radii = None

        self._load_data()


    def _load_data(self):
        self.centers = np.loadtxt(
            self.vertices_file,
            usecols=range(self.xyzcols[0], self.xyzcols[1])
        )
        self.radii = np.loadtxt(self.radii_file)

        mask = (self.radii >= self.r_min) & (self.radii <= self.r_max)
        self.centers = self.centers[mask]
        self.radii = self.radii[mask]

    # ---------------------------
    # Geometry
    # ---------------------------

    def _tetra_circumradius(self, points):

        a, b, c, d = points
        A = np.vstack([b - a, c - a, d - a]).T
        rhs = np.array([
            np.dot(b, b) - np.dot(a, a),
            np.dot(c, c) - np.dot(a, a),
            np.dot(d, d) - np.dot(a, a)
        ]) / 2
        try:
            center = np.linalg.solve(A, rhs)
            return np.linalg.norm(center - a)
        except np.linalg.LinAlgError:
            return np.inf

    def _tetra_volume(self, points):
        a, b, c, d = points
        return abs(np.dot((a - d), np.cross(b - d, c - d))) / 6.0

    def _triangle_area(self, points):
        a, b, c = points
        return 0.5 * np.linalg.norm(np.cross(b - a, c - a))

    # ---------------------------
    # Delaunay
    # ---------------------------

    def _compute_delaunay(self):
        self.delaunay = Delaunay(self.centers)
        self.simplices = self.delaunay.simplices
        self.tetra_pts = self.centers[self.simplices]

        self.circ_radii = np.array(
            [self._tetra_circumradius(t) for t in self.tetra_pts]
        )

        self.tetra_volumes = np.array(
            [self._tetra_volume(t) for t in self.tetra_pts]
        )

    def compute_delaunay(self):
        """Public wrapper."""
        self._compute_delaunay()

    # ---------------------------
    # Alpha optimization
    # ---------------------------

    def optimize_alpha(self, alpha_min=0.5, alpha_max=30,
                       threshold=1e-5, max_iter=200):

        if self.delaunay is None:
            raise RuntimeError("Call compute_delaunay() first.")

        low, high = float(alpha_min), float(alpha_max)

        alphas_tried = []
        max_volumes = []
        optimal_alpha = None

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

            if high - low < 1e-6:
                break

        if optimal_alpha is None and len(alphas_tried) > 0:
            idx = int(np.argmin(
                np.abs(np.array(max_volumes) - self.target_volume)
            ))
            optimal_alpha = float(alphas_tried[idx])

        self.alpha = optimal_alpha
        self.alpha_history = alphas_tried
        self.volume_history = max_volumes

    # ---------------------------
    # Alpha filtering
    # ---------------------------

    def _filter_tetrahedra(self):
        keep_mask = self.circ_radii <= self.alpha
        self.kept_tetra = self.simplices[keep_mask]
        self.kept_tetra_pts = self.centers[self.kept_tetra]

    def build_alpha_shape(self):
        """Public alpha-shape construction."""
        if self.delaunay is None:
            raise RuntimeError("Call compute_delaunay() first.")

        if self.alpha is None:
            raise RuntimeError("Alpha not defined.")

        self._filter_tetrahedra()

    # ---------------------------
    # Cavities
    # ---------------------------

    def compute_cavities(self):

        if self.kept_tetra is None:
            raise RuntimeError("Call build_alpha_shape() first.")

        face_dict = {}

        for tidx, tet in enumerate(self.kept_tetra):
            for face in combinations(sorted(tet), 3):
                face_dict.setdefault(face, []).append(tidx)

        adjacency = np.zeros(
            (len(self.kept_tetra), len(self.kept_tetra)),
            dtype=bool
        )

        for tlist in face_dict.values():
            if len(tlist) == 2:
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

            face_count = {}
            for tet in tets:
                for face in combinations(sorted(tet), 3):
                    face_count[face] = face_count.get(face, 0) + 1

            boundary_faces = [
                f for f, count in face_count.items() if count == 1
            ]

            self.cavity_areas[c] = np.sum(
                [self._triangle_area(self.centers[list(f)])
                 for f in boundary_faces]
            )

            self.cavity_radii[c] = (
                3 * self.cavity_volumes[c] / (4 * np.pi)
            ) ** (1/3)

            self.cavity_centroids[c] = np.mean(
                t_pts.reshape(-1, 3), axis=0
            )
