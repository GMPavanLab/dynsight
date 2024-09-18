from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from ovito.io import import_file
from ovito.modifiers import (
    CoordinationAnalysisModifier,
    DeleteSelectedModifier,
    ExpressionSelectionModifier,
    LoadTrajectoryModifier,
    TimeAveragingModifier,
)
from scipy.signal import argrelextrema


class RDF:
    """Radial Distribution Function."""

    def __init__(self, topo_file: Path, traj_file: Path) -> None:
        """Initialization funciton for RDF object."""
        self.topo_file = topo_file
        self.traj_file = traj_file
        self.pipeline = None
        self.min_points = None
        self.rdf_bins = None
        self.rdf = None

    def compute_rdf(
        self, cutoff: float, bins: int, selection: str
    ) -> np.ndarray:
        """RDF computation."""
        # Set up the OVITO pipeline
        self.pipeline = import_file(self.topo_file)
        trajectory = LoadTrajectoryModifier()
        trajectory.source.load(self.traj_file)
        self.pipeline.modifiers.append(trajectory)

        selection_modifier = ExpressionSelectionModifier(expression=selection)
        self.pipeline.modifiers.append(selection_modifier)
        self.pipeline.modifiers.append(DeleteSelectedModifier())

        # Add coordination analysis modifier
        coord_modifier = CoordinationAnalysisModifier(
            cutoff=cutoff, number_of_bins=bins
        )
        self.pipeline.modifiers.append(coord_modifier)

        # Time averaging
        averaging_modifier = TimeAveragingModifier(
            operate_on="table:coordination-rdf"
        )
        self.pipeline.modifiers.append(averaging_modifier)

        # Compute the RDF
        data = self.pipeline.compute()
        total_rdf = data.tables["coordination-rdf[average]"].xy()
        self.rdf_bins = total_rdf[:, 0]
        self.rdf = total_rdf[:, 1]

    def find_minpoints(self) -> np.ndarray:
        """Find minumum points in RDF."""
        if self.rdf is None:
            raise_message = "RDF has not been computed yet."
            raise ValueError(raise_message)
        self.min_points = argrelextrema(self.rdf, np.less)

    def rdf_plot(self, minpoints: bool) -> None:
        """Plot the RDF."""
        if self.rdf is None:
            raise_message = "RDF has not been computed yet."
            raise ValueError(raise_message)
        if minpoints and (self.min_points is None):
            raise_message = "Minumum points have not been computed yet."
            raise ValueError(raise_message)
        plt.plot(self.rdf_bins, self.rdf, label="RDF", color="black")
        if minpoints:
            plt.scatter(
                self.rdf_bins[self.min_points[0]],
                self.rdf[self.min_points],
                color="red",
                label="Minimum points",
            )
        plt.title("Radial Distribution Function")
        plt.xlabel("Radius")
        plt.ylabel("g(r)")
        plt.legend()
        plt.show()
