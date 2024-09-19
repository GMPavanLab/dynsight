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


class RDF:
    """Object for computing and analyzing the Radial Distribution Function.

    The Radial Distribution Function (RDF) describes how particle density
    varies as a function of the distance from a reference particle.
    """

    def __init__(
        self,
        trajectory_file: Path,
        topology_file: Path | None = None,
        xyz_cols: list[str] | None = None,
    ) -> None:
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

    def compute_rdf(
        self,
        cutoff: float,
        bins: int,
        remove_atoms: str | None = None,
        step: int = 1,
    ) -> None:
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
        inverted_rdf = -self.rdf
        peaks, properties = find_peaks(inverted_rdf, prominence=prominence)
        self.minima_points = [self.pair_distances[peaks], self.rdf[peaks]]

    def rdf_plot(self, minpoints: bool) -> None:
        """Plot the RDF curve with optional marking of the minimum points.

        Args:
            minpoints (bool): If True, the local minimum points are
            highlighted on the RDF plot.

        Raises:
            ValueError: If RDF has not been computed yet or if minimum points
            have not been computed when requested.
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
