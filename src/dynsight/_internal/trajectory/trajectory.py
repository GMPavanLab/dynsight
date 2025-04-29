from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import MDAnalysis
    import numpy as np
    from numpy.typing import NDArray
    from tropea_clustering._internal.first_classes import StateMulti, StateUni

import dynsight

UNIVAR_DIM = 2


@dataclass
class Insight:
    """Contains an analysis perfomed on a trajectory.

    Attributes:
        dataset: the values of a single-particle descriptor. Has shape
            (n_particles, n_frames) or (n_particles, n_frames, n_dims).
        r_cut: the spatial cutoff with the descriptor has been computed.
    """

    dataset: NDArray[np.float64]
    r_cut: float

    def spatial_average(
        self,
        trajectory: Trj,
        r_cut: float,
        selection: str = "all",
        num_processes: int = 1,
    ) -> Insight:
        """Average the descripotor over the neighboring particles."""
        averaged_dataset = dynsight.analysis.spatialaverage(
            universe=trajectory.universe,
            descriptor_array=self.dataset,
            selection=selection,
            cutoff=r_cut,
            num_processes=num_processes,
        )
        return Insight(averaged_dataset, self.r_cut)

    def get_onion(self, delta_t: int) -> OnionInsight:
        """Perform onion clustering."""
        if self.dataset.ndim == UNIVAR_DIM:
            reshaped_data = dynsight.onion.helpers.reshape_from_nt(
                self.dataset, delta_t
            )
            onion_clust = dynsight.onion.OnionUni()
        else:
            reshaped_data = dynsight.onion.helpers.reshape_from_dnt(
                self.dataset.transpose(2, 0, 1), delta_t
            )
            onion_clust = dynsight.onion.OnionMulti(ndims=self.dataset.ndim)

        onion_clust.fit(reshaped_data)

        return OnionInsight(
            dataset=self.dataset,
            r_cut=self.r_cut,
            labels=onion_clust.labels_,
            delta_t=delta_t,
            state_list=onion_clust.state_list_,
            reshaped_data=reshaped_data,
        )

    def dump_insight(self, file_name: Path) -> None:
        """Save a copy of the Insight object."""
        file_path = file_name.with_suffix(".pkl")
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
    state_list: list[StateUni] | list[StateMulti]
    reshaped_data: NDArray[np.float64]

    def plot_output(self, file_name: Path) -> None:
        """Plot the overall onion clustering result."""
        if self.dataset.ndim == UNIVAR_DIM:
            dynsight.onion.plot.plot_output_uni(
                file_name,
                self.reshaped_data,
                self.dataset.shape[0],
                self.state_list,
            )
        else:
            dynsight.onion.plot.plot_output_multi(
                file_name,
                self.dataset,
                self.state_list,
                self.labels,
                self.delta_t,
            )

    def plot_one_trj(self, file_name: Path, particle_id: int) -> None:
        """Plot one particle's trajectory colored according to clustering."""
        if self.dataset.ndim == UNIVAR_DIM:
            dynsight.onion.plot.plot_one_trj_uni(
                file_name,
                particle_id,
                self.reshaped_data,
                self.dataset.shape[0],
                self.labels,
            )
        else:
            dynsight.onion.plot.plot_one_trj_multi(
                file_name,
                particle_id,
                self.delta_t,
                self.dataset,
                self.labels,
            )

    def plot_medoids(self, file_name: Path) -> None:
        """Plot the average sequence of each onion cluster."""
        if self.dataset.ndim == UNIVAR_DIM:
            dynsight.onion.plot.plot_medoids_uni(
                file_name,
                self.reshaped_data,
                self.labels,
            )
        else:
            dynsight.onion.plot.plot_medoids_multi(
                file_name,
                self.delta_t,
                self.dataset,
                self.labels,
            )

    def plot_state_populations(self, file_name: Path) -> None:
        """Plot each state's population along the trajectory."""
        dynsight.onion.plot.plot_state_populations(
            file_name,
            self.dataset.shape[0],
            self.delta_t,
            self.labels,
        )

    def plot_sankey(self, file_name: Path, frame_list: list[int]) -> None:
        """Plot the Sankey diagram of the onion clustering."""
        dynsight.onion.plot.plot_sankey(
            file_name,
            self.labels,
            self.dataset.shape[0],
            frame_list,
        )


@dataclass
class Trj:
    """Contains a trajectory."""

    universe: MDAnalysis.Universe

    def get_lens(self, r_cut: float) -> Insight:
        """Compute LENS on the trajectory."""
        neigcounts = dynsight.lens.list_neighbours_along_trajectory(
            input_universe=self.universe,
            cutoff=r_cut,
        )
        lens, *_ = dynsight.lens.neighbour_change_in_time(neigcounts)
        return Insight(lens, r_cut)

    def get_soap(
        self, r_cut: float, n_max: int, l_max: int, n_core: int
    ) -> Insight:
        """Compute SOAP on the trajectory."""
        soap = dynsight.soap.saponify_trajectory(
            self.universe,
            soaprcut=r_cut,
            soapnmax=n_max,
            soaplmax=l_max,
            n_core=n_core,
        )
        return Insight(soap, r_cut)

    def get_timesoap(
        self, r_cut: float, n_max: int, l_max: int, n_core: int
    ) -> Insight:
        """Compute timeSOAP on the trajectory."""
        soap = dynsight.soap.saponify_trajectory(
            self.universe,
            soaprcut=r_cut,
            soapnmax=n_max,
            soaplmax=l_max,
            n_core=n_core,
        )
        tsoap = dynsight.soap.timesoap(soap)
        return Insight(tsoap, r_cut)

    def dump_trj(self, file_name: Path) -> None:
        """Save a copy of the Trj object."""
        file_path = file_name.with_suffix(".pkl")
        with Path.open(file_path, "wb") as file:
            pickle.dump(self, file)
