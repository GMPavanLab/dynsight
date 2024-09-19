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
    """Object for computing and analyzing the Radial Distribution Function.

    The Radial Distribution Function (RDF) describes how particle density
    varies as a function of the distance from a reference particle.
    """

    def __init__(self) -> None:
        """Initialization function for RDF object.

        Attributes:
            pipeline (ovito.pipeline.Pipeline | None): OVITO pipeline object
            that handles data import and processing.
            min_points (np.ndarray | None): Array storing the indices of
            the local minima in the RDF data.
            rdf_bins (np.ndarray | None): Array of bin centers corresponding
            to the RDF data.
            rdf (np.ndarray | None): Computed RDF values.
        """
        self.pipeline = None
        self.min_points = None
        self.rdf_bins = None
        self.rdf = None

    def read_from_xyz(self, input_file: Path, columns: tuple[str]) -> None:
        """Read trajectory data from an XYZ file.

        Args:
            input_file (Path): Path to the input XYZ file containing
            atomic positions.
            columns (tuple[str]): Column names for XYZ data to map to the
            correct attributes.

        Raises:
            FileNotFoundError: If the specified file does not exist.
        """
        self.pipeline = import_file(Path(input_file), columns=columns)

    def read_from_gromacs(self, topo_file: Path, traj_file: Path) -> None:
        """Read trajectory from GROMACS .gro and .xtc files.

        Args:
            topo_file (Path): Path to the topology file (.gro) used for
            GROMACS simulations.
            traj_file (Path): Path to the trajectory file (.xtc) used for
            GROMACS simulations.

        Raises:
            FileNotFoundError: If the specified files do not exist.
        """
        self.pipeline = import_file(Path(topo_file))
        trajectory = LoadTrajectoryModifier()
        trajectory.source.load(Path(traj_file))
        self.pipeline.modifiers.append(trajectory)

    def compute_rdf(
        self,
        cutoff: float,
        bins: int,
        remove_atoms: str | None = None,
        step: int = 1,
    ) -> np.ndarray:
        """Compute the RDF using the loaded trajectory.

        Args:
            cutoff (float): The maximum distance (cutoff) for calculating the
            RDF.
            bins (int): Number of bins to divide the radial distances into.
            remove_atoms (str | None, optional): A selection
            string (OVITO expression language) to define which atoms should
            be removed before RDF calculation.
            step (int, optional): Sampling frequency (in timesteps) for
            time-averaging the RDF. Default is 1.

        Raises:
            ValueError: If no trajectory is loaded in the pipeline.
        """
        if self.pipeline is None:
            raise_message = "Unloaded trajectory."
            raise ValueError(raise_message)

        if remove_atoms is not None:
            selection_modifier = ExpressionSelectionModifier(
                expression=remove_atoms
            )
            self.pipeline.modifiers.append(selection_modifier)
            self.pipeline.modifiers.append(DeleteSelectedModifier())

        coord_modifier = CoordinationAnalysisModifier(
            cutoff=cutoff, number_of_bins=bins
        )
        self.pipeline.modifiers.append(coord_modifier)

        averaging_modifier = TimeAveragingModifier(
            operate_on="table:coordination-rdf",
            sampling_frequency=step,
        )
        self.pipeline.modifiers.append(averaging_modifier)

        data = self.pipeline.compute()
        total_rdf = data.tables["coordination-rdf[average]"].xy()
        self.rdf_bins = total_rdf[:, 0]
        self.rdf = total_rdf[:, 1]

    def find_minpoints(self) -> np.ndarray:
        """Find local minima points in the RDF data.

        This method uses scipy's `argrelextrema` function to find the
        indices of local minima in the RDF.

        Raises:
            ValueError: If the RDF has not been computed yet.
        """
        if self.rdf is None:
            raise_message = "RDF has not been computed yet."
            raise ValueError(raise_message)
        self.min_points = argrelextrema(self.rdf, np.less)

    def rdf_plot(self, minpoints: bool) -> None:
        """Plot the RDF curve with optional marking of the minimum points.

        Args:
            minpoints (bool): If True, the local minimum points are
            highlighted on the RDF plot.

        Raises:
            ValueError: If RDF has not been computed yet or if minimum points
            have not been computed when requested.
        """
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
        plt.xlabel("Pair separation distance")
        plt.ylabel("g(r)")
        plt.legend()
        plt.tight_layout()
        plt.show()
