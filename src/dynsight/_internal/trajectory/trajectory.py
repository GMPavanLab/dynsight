from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import MDAnalysis
    import numpy as np
    from numpy.typing import NDArray
    from tropea_clustering._internal.first_classes import StateUni

import dynsight


@dataclass
class Insight:
    """Contains an analysis perfomed on a many-body trajectory."""

    dataset: NDArray[np.float64]
    r_cut: float

    def spatial_average(
        self,
        trajectory: Trj,
        r_cut: float,
        selection: str = "all",
        num_processes: int = 1,
    ) -> Insight:
        averaged_dataset = dynsight.analysis.spatialaverage(
            universe=trajectory.universe,
            descriptor_array=self.dataset,
            selection=selection,
            cutoff=r_cut,
            num_processes=num_processes,
        )
        return Insight(averaged_dataset, r_cut)

    def get_onion(self, delta_t: int) -> OnionInsight:
        reshaped_data = dynsight.onion.helpers.reshape_from_nt(
            self.dataset, delta_t
        )
        state_list, labels = dynsight.onion.onion_uni(reshaped_data)
        return OnionInsight(
            dataset=self.dataset,
            r_cut=self.r_cut,
            labels=labels,
            delta_t=delta_t,
            state_list=state_list,
            reshaped_data=reshaped_data,
        )

    def save(self, file_name: str) -> None:
        file_path = Path(f"{file_name}.pkl")
        with Path.open(file_path, "wb") as file:
            pickle.dump(self, file)


@dataclass
class ClusterInsight(Insight):
    """Contains a clustering analysis."""

    labels: NDArray[np.int64]


@dataclass
class OnionInsight(ClusterInsight):
    """Contains a onion-clustering analysis."""

    delta_t: int
    state_list: list[StateUni]
    reshaped_data: NDArray[np.float64]

    def plot_output(self, file_name: str) -> None:
        dynsight.onion.plot.plot_output_uni(
            file_name,
            self.reshaped_data,
            self.dataset.shape[0],
            self.state_list,
        )

    def plot_one_trj(self, particle_id: int, file_name: str) -> None:
        dynsight.onion.plot.plot_one_trj_uni(
            file_name,
            particle_id,
            self.reshaped_data,
            self.dataset.shape[0],
            self.labels,
        )

    def plot_medoids(self, file_name: str) -> None:
        dynsight.onion.plot.plot_medoids_uni(
            file_name,
            self.reshaped_data,
            self.labels,
        )

    def plot_state_populations(self, file_name: str) -> None:
        dynsight.onion.plot.plot_state_populations(
            file_name,
            self.dataset.shape[0],
            self.delta_t,
            self.labels,
        )

    def plot_sankey(self, file_name: str, frame_list: list[int]) -> None:
        dynsight.onion.plot.plot_sankey(
            file_name,
            self.labels,
            self.dataset.shape[0],
            frame_list,
        )


@dataclass
class Trj:
    """Contains a many-body trajectory."""

    universe: MDAnalysis.Universe
    dt: float = field(init=False)

    def __post_init__(self) -> None:
        self.dt = self.universe.trajectory.dt

    def get_lens(self, r_cut: float) -> Insight:
        neigcounts = dynsight.lens.list_neighbours_along_trajectory(
            input_universe=self.universe,
            cutoff=r_cut,
        )
        lens, *_ = dynsight.lens.neighbour_change_in_time(neigcounts)
        return Insight(lens, r_cut)

    def save(self, file_name: str) -> None:
        file_path = Path(f"{file_name}.pkl")
        with Path.open(file_path, "wb") as file:
            pickle.dump(self, file)
