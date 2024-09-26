from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import numpy as np

from ovito.io import import_file
from ovito.modifiers import (
    CoordinationAnalysisModifier,
    DeleteSelectedModifier,
    ExpressionSelectionModifier,
    LoadTrajectoryModifier,
    TimeAveragingModifier,
)


class RadialDistributionFunction:
    """Object for computing the Radial Distribution Function.

    The Radial Distribution Function (RDF) measures the probability of finding
    a particle at a certain distance provided. The RDF is normalized by
    the average number density of particles.

    To compute the Radial Distribution Function OVITO API has been used,
    more information about the theory adopted can be
    found `here <https://www.ovito.org/manual/reference/pipelines/modifiers/coordination_analysis.html>`_.

    Parameters:
        cutoff:
            Maximum distance for calculating the function.
        bins:
            Number of bins that will be used for the RDF calculation.
        frequency:
            Frequency at which frames are taken to compute the RDF. If not
            specified, all frames will be used and their RDFs will
            be avereged.
    """

    def __init__(
        self, cutoff: float, bins: int = 200, frequency: int = 1
    ) -> None:
        """Initialize the RDF object and sets the calculation parameter."""
        self._cutoff = cutoff
        self._bins = bins
        self._frequency = frequency

    def compute(
        self,
        trajectory_file: Path | str,
        topology_file: Path | str | None = None,
        xyz_cols: list[str] | None = None,
        remove_atoms: str | None = None,
    ) -> np.ndarray[float, Any]:
        """Computes the Radial Distribution Function.

        Parameters:
            trajectory_file:
                Trajectory file path (e.g. `.xyz` or `.xtc`).
            topology_file:
                Topology file path if required (e.g. `.gro`).
            remove_atoms:
                OVITO expression selection for atoms
                to be removed.
            xyz_cols:
                List of column names for XYZ format
                files (e.g., `["Particle Type", "X", "Y", "Z"]`).

        Returns:
            A NumPy array of shape (`bins`, 2) containing pair distances
            in the first column and their respective RDF value in the
            second column.

        Important:
            If you're using `remove_atoms` to exclude certain
            particles from the RDF calculation, you need to use the
            `OVITO's expression syntax <https://www.ovito.org/manual/reference/pipelines/modifiers/expression_select.html>`_.

            Example expressions:
                - `"ParticleType == 1"`: Selects all particles of type 1.

                - `"Position.X > 0"`: Selects particles with an x-coordinate
                    greater than zero.

            Be sure to have this type of data stored in your
            simulation file before using this command.
        """
        if topology_file is None:
            trajectory = [Path(trajectory_file)]
        else:
            trajectory = [Path(topology_file), Path(trajectory_file)]

        if topology_file is None:
            pipeline = import_file(trajectory, columns=xyz_cols)
        else:
            pipeline = import_file(trajectory[0])
            timeframes = LoadTrajectoryModifier()
            timeframes.source.load(trajectory[1])
            pipeline.modifiers.append(timeframes)
        if remove_atoms is not None:
            selection_modifier = ExpressionSelectionModifier(
                expression=remove_atoms
            )
            pipeline.modifiers.append(selection_modifier)
            pipeline.modifiers.append(DeleteSelectedModifier())
        coord_modifier = CoordinationAnalysisModifier(
            cutoff=self._cutoff, number_of_bins=self._bins
        )
        pipeline.modifiers.append(coord_modifier)
        averaging_modifier = TimeAveragingModifier(
            operate_on="table:coordination-rdf",
            sampling_frequency=self._frequency,
        )
        pipeline.modifiers.append(averaging_modifier)
        workflow_res = pipeline.compute()
        return workflow_res.tables["coordination-rdf[average]"].xy()
