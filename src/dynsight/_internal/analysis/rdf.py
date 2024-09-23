from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from ovito.io import import_file
from ovito.modifiers import (
    CoordinationAnalysisModifier,
    DeleteSelectedModifier,
    ExpressionSelectionModifier,
    LoadTrajectoryModifier,
    TimeAveragingModifier,
)
from scipy.signal import find_peaks

OVITO_GUI_MODE = 0
class RadialDistributionFunction:
    """Object for computing and analyzing the Radial Distribution Function.

    The Radial Distribution Function (RDF) describes how particle density
    varies as a function of the distance.

    Parameters:
        trajectory (list[Path]):
            List of file paths including trajectory and optional topology file.

        pipeline (Pipeline):
            OVITO pipeline used for importing and modifying the
            trajectory data.

        pair_distances (np.ndarray):
            Array of distances between particle pairs.

        rdf (np.ndarray):
            Array of RDF values corresponding to the pair distances.

        minima_points (tuple):
            Tuple containing arrays of minima point distances and RDF values.
    """

    def __init__(
        self,
        trajectory_file: Path,
        topology_file: Path | None = None,
        xyz_cols: list[str] | None = None,
    ) -> None:
        """Initialize the RDF object and import the trajectory data.

        Parameters:
            trajectory_file (Path):
                Path to the trajectory file (e.g. `.xyz` or `.xtc`).

            topology_file (Path | None, optional):
                Path to the topology file if required (e.g. `.gro`).

            xyz_cols (list[str] | None, optional):
                List of column names for XYZ format
                files (e.g., `["Particle Type", "X", "Y", "Z"]`).
        """
        # Check number of args and setup the trajectory files
        if topology_file is None:
            self.trajectory = [Path(trajectory_file)]
        else:
            self.trajectory = [Path(topology_file), Path(trajectory_file)]

        # Setup the pipeline
        if topology_file is None:
            self.pipeline = import_file(self.trajectory, columns=xyz_cols)
        else:
            self.pipeline = import_file(self.trajectory[0])
            timeframes = LoadTrajectoryModifier()
            timeframes.source.load(str(self.trajectory[1]))
            self.pipeline.modifiers.append(timeframes)

    def compute(
        self,
        cutoff: float,
        bins: int = 200,
        remove_atoms: str | None = None,
        step: int = 1,
    ) -> None:
        """Compute the Radial Distribution Function (RDF).

        Parameters:
            cutoff (float):
                Maximum distance for calculating the RDF.

            bins (int):
                Number of bins for the RDF calculation, Default is 200.

            remove_atoms (str | None, optional):
                Selection OVITO expression for atoms
                to be removed (e.g., 'ParticleType == 1').

            step (int, optional):
                Frequency of time sampling for averaging
                the RDF. Default is 1.

        Important:
            If you're using the `remove_atoms` argument to remove certain
            particles from the RDF calculation, you need to use the
            OVITO's expression syntax for selecting particles.

            Example expressions:
                - `"ParticleType == 1"`: Selects all particles of type 1.

                - `"Position.X > 0"`: Selects particles with an x-coordinate
                    greater than zero.

                - `"ParticleIdentifier % 2 == 0"`: Selects particles with even
                    particle identifiers.

            Be sure to have this type of information stored in your
            simulation file before using this command.

            More information on OVITO's expression syntax can be found here:
            https://www.ovito.org/manual/reference/pipelines/modifiers/expression_select.html
        """
        # Delete selected atoms (OVITO commands)
        if remove_atoms is not None:
            selection_modifier = ExpressionSelectionModifier(
                expression=remove_atoms
            )
            self.pipeline.modifiers.append(selection_modifier)
            self.pipeline.modifiers.append(DeleteSelectedModifier())

        # Compute RDF
        coord_modifier = CoordinationAnalysisModifier(
            cutoff=cutoff, number_of_bins=bins
        )
        self.pipeline.modifiers.append(coord_modifier)

        # Time averaging
        averaging_modifier = TimeAveragingModifier(
            operate_on="table:coordination-rdf", sampling_frequency=step
        )
        self.pipeline.modifiers.append(averaging_modifier)

        # Compile the pipeline
        data = self.pipeline.compute()
        total_rdf = data.tables["coordination-rdf[average]"].xy()
        self.pair_distances = total_rdf[:, 0]
        self.rdf = total_rdf[:, 1]

    def find_minima_points(self, prominence: float) -> None:
        """Find local minima in the RDF.

        Parameters:
            prominence (float):
                Minimum prominence of peaks in the RDF
                for identifying minima points. Tune this parameter to obtain
                better performance.
        """
        inverted_rdf = -self.rdf
        peaks, properties = find_peaks(inverted_rdf, prominence=prominence)
        self.minima_points = [self.pair_distances[peaks], self.rdf[peaks]]

    def plot_rdf(self, minpoints: bool = False) -> None:
        """Plot the RDF curve with optional marking of the minimum points.

        Parameters:
            minpoints (bool, optional):
                If `True`, the local minimum points
                are highlighted on the RDF plot. Default is `False`.
        """
        plt.plot(self.pair_distances, self.rdf, label="RDF", color="black")
        if minpoints:
            plt.scatter(
                self.minima_points[0],
                self.minima_points[1],
                color="red",
                label="Minimum points",
            )
        plt.title("Radial Distribution Function")
        plt.xlabel("Pair separation distance")
        plt.ylabel("g(r)")
        plt.legend()
        plt.tight_layout()
        plt.show()
