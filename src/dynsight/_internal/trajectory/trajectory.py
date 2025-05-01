from __future__ import annotations

import json
from dataclasses import dataclass, field, fields
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from pathlib import Path

    import MDAnalysis
    from numpy.typing import NDArray
    from tropea_clustering._internal.first_classes import StateMulti, StateUni

import dynsight

UNIVAR_DIM = 2


@dataclass(frozen=True)
class Insight:
    """Contains an analysis perfomed on a trajectory.

    Attributes:
        dataset: the values of a single-particle descriptor. Has shape
            (n_particles, n_frames) or (n_particles, n_frames, n_dims).
    """

    dataset: NDArray[np.float64]
    meta: dict[str, Any] = field(default_factory=dict)

    def dump_to_json(self, file_path: Path) -> None:
        """Save the Insight object as .json file."""
        data = self.__dict__
        data["dataset"] = data["dataset"].tolist()
        with file_path.open("w") as file:
            json.dump(data, file, indent=4)

    @classmethod
    def load_from_json(cls, file_path: Path) -> Insight:
        """Load the Insight object from .json file."""
        with file_path.open("r") as file:
            data = json.load(file)

        if "dataset" not in data:
            msg = "'dataset' key not found in JSON file."
            raise ValueError(msg)

        return cls(
            dataset=np.array(data.get("dataset")),
            meta=data.get("meta"),
        )

    def spatial_average(
        self,
        trajectory: Trj,
        r_cut: float,
        selection: str = "all",
        num_processes: int = 1,
    ) -> Insight:
        """Average the descripotor over the neighboring particles.

        The returned Insight contains the following meta:
            * sp_av_r_cut: the r_cut value used for the computation.
        """
        averaged_dataset = dynsight.analysis.spatialaverage(
            universe=trajectory.universe,
            descriptor_array=self.dataset,
            selection=selection,
            cutoff=r_cut,
            num_processes=num_processes,
        )
        return Insight(
            dataset=averaged_dataset,
            meta={"sp_av_r_cut": r_cut},
        )

    def get_onion(self, delta_t: int) -> OnionInsight:
        """Perform onion clustering.

        The returned OnionInsight contains the following meta:
            * delta_t: the delta_t value used for the clustering.
        """
        if self.dataset.ndim == UNIVAR_DIM:
            reshaped_data = dynsight.onion.helpers.reshape_from_nt(
                self.dataset, delta_t
            )
            onion_clust = dynsight.onion.OnionUni()
        else:
            reshaped_data = dynsight.onion.helpers.reshape_from_dnt(
                self.dataset.transpose(2, 0, 1), delta_t
            )
            onion_clust = dynsight.onion.OnionMulti(
                ndims=self.dataset.ndim - 1
            )

        onion_clust.fit(reshaped_data)

        return OnionInsight(
            labels=onion_clust.labels_,
            state_list=onion_clust.state_list_,
            reshaped_data=reshaped_data,
            meta={"delta_t": delta_t},
        )


@dataclass(frozen=True)
class ClusterInsight:
    """Contains a clustering analysis."""

    labels: NDArray[np.int64]

    def dump_to_json(self, file_path: Path) -> None:
        """Save the ClusterInsight object as .json file."""
        data = self.__dict__
        data["labels"] = data["labels"].tolist()
        with file_path.open("w") as file:
            json.dump(data, file, indent=4)

    @classmethod
    def load_from_json(cls, file_path: Path) -> ClusterInsight:
        """Load the ClusterInsight object from .json file."""
        with file_path.open("r") as file:
            data = json.load(file)
        if "labels" not in data:
            msg = "'labels' key not found in JSON file."
            raise ValueError(msg)
        return cls(labels=np.array(data.get("labels")))


@dataclass(frozen=True)
class OnionInsight(ClusterInsight):
    """Contains a onion-clustering analysis."""

    state_list: list[StateUni] | list[StateMulti]
    reshaped_data: NDArray[np.float64]
    meta: dict[str, Any] = field(default_factory=dict)

    def dump_to_json(self, file_path: Path) -> None:
        """Save the OnionInsight object as .json file."""
        data = self.__dict__
        data["labels"] = data["labels"].tolist()
        new_state_list = []
        for state in data["state_list"]:
            tmp = {}
            for f in fields(state):
                value = getattr(state, f.name)
                if isinstance(value, np.ndarray):
                    tmp[f.name] = value.tolist()
                else:
                    tmp[f.name] = value
            new_state_list.append(tmp)
        data["state_list"] = new_state_list
        data["reshaped_data"] = data["reshaped_data"].tolist()
        with file_path.open("w") as file:
            json.dump(data, file, indent=4)

    @classmethod
    def load_from_json(cls, file_path: Path) -> OnionInsight:
        """Load the OnionInsight object from .json file."""
        with file_path.open("r") as file:
            data = json.load(file)
        if "state_list" not in data:
            msg = "'state_list' key not found in JSON file."
            raise ValueError(msg)
        return cls(
            labels=np.array(data.get("labels")),
            state_list=data.get("state_list"),
            reshaped_data=np.array(data.get("reshaped_data")),
            meta=data.get("meta"),
        )

    def plot_output(self, file_path: Path, data_insight: Insight) -> None:
        """Plot the overall onion clustering result."""
        if data_insight.dataset.ndim == UNIVAR_DIM:
            dynsight.onion.plot.plot_output_uni(
                file_path,
                self.reshaped_data,
                data_insight.dataset.shape[0],
                self.state_list,
            )
        else:
            dynsight.onion.plot.plot_output_multi(
                file_path,
                data_insight.dataset,
                self.state_list,
                self.labels,
                self.meta["delta_t"],
            )

    def plot_one_trj(
        self,
        file_path: Path,
        data_insight: Insight,
        particle_id: int,
    ) -> None:
        """Plot one particle's trajectory colored according to clustering."""
        if data_insight.dataset.ndim == UNIVAR_DIM:
            dynsight.onion.plot.plot_one_trj_uni(
                file_path,
                particle_id,
                self.reshaped_data,
                data_insight.dataset.shape[0],
                self.labels,
            )
        else:
            dynsight.onion.plot.plot_one_trj_multi(
                file_path,
                particle_id,
                self.meta["delta_t"],
                data_insight.dataset,
                self.labels,
            )

    def plot_medoids(self, file_path: Path, data_insight: Insight) -> None:
        """Plot the average sequence of each onion cluster."""
        if data_insight.dataset.ndim == UNIVAR_DIM:
            dynsight.onion.plot.plot_medoids_uni(
                file_path,
                self.reshaped_data,
                self.labels,
            )
        else:
            dynsight.onion.plot.plot_medoids_multi(
                file_path,
                self.meta["delta_t"],
                data_insight.dataset,
                self.labels,
            )

    def plot_state_populations(
        self,
        file_path: Path,
        data_insight: Insight,
    ) -> None:
        """Plot each state's population along the trajectory."""
        dynsight.onion.plot.plot_state_populations(
            file_path,
            data_insight.dataset.shape[0],
            self.meta["delta_t"],
            self.labels,
        )

    def plot_sankey(
        self,
        file_path: Path,
        data_insight: Insight,
        frame_list: list[int],
    ) -> None:
        """Plot the Sankey diagram of the onion clustering."""
        dynsight.onion.plot.plot_sankey(
            file_path,
            self.labels,
            data_insight.dataset.shape[0],
            frame_list,
        )


@dataclass(frozen=True)
class Trj:
    """Contains a trajectory."""

    universe: MDAnalysis.Universe = field()

    @classmethod
    def init_from_universe(cls, universe: MDAnalysis.Universe) -> Trj:
        """Initialize Trj object from MDAnalysis.Universe."""
        return Trj(universe)

    def get_lens(self, r_cut: float, neigh_count: bool = False) -> Insight:
        """Compute LENS on the trajectory.

        The returned Insight contains the following meta:
            * r_cut: the r_cut value used for the LENS calculation.
        """
        neigcounts = dynsight.lens.list_neighbours_along_trajectory(
            input_universe=self.universe,
            cutoff=r_cut,
        )
        lens, nn, *_ = dynsight.lens.neighbour_change_in_time(neigcounts)
        if neigh_count:
            return Insight(
                dataset=nn.astype(np.float64),
                meta={"r_cut": r_cut, "neigh_count": neigh_count},
            )
        return Insight(
            dataset=lens,
            meta={"r_cut": r_cut, "neigh_count": neigh_count},
        )

    def get_soap(
        self,
        r_cut: float,
        n_max: int,
        l_max: int,
        respect_pbc: bool = True,
        centers: str = "all",
        n_core: int = 1,
    ) -> Insight:
        """Compute SOAP on the trajectory.

        The returned Insight contains the following meta:
            * r_cut: the r_cut value used for the SOAP calculation.
            * n_max: the n_max value used for the SOAP calculation.
            * l_max: the l_max value used for the SOAP calculation.
            * respect_pbc: bust be True if trajectory has PBC
            * centers: selection of atoms used as centers for the SOAP
                calculation.
        """
        soap = dynsight.soap.saponify_trajectory(
            self.universe,
            soaprcut=r_cut,
            soapnmax=n_max,
            soaplmax=l_max,
            soap_respectpbc=respect_pbc,
            centers=centers,
            n_core=n_core,
        )
        attr_dict = {
            "r_cut": r_cut,
            "n_max": n_max,
            "l_max": l_max,
            "respect_pbc": respect_pbc,
            "centers": centers,
        }
        return Insight(dataset=soap, meta=attr_dict)

    def get_timesoap(
        self,
        r_cut: float,
        n_max: int,
        l_max: int,
        respect_pbc: bool = True,
        centers: str = "all",
        delay: int = 1,
        n_core: int = 1,
    ) -> Insight:
        """Compute timeSOAP on the trajectory.

        The returned Insight contains the following meta:
            * r_cut: the r_cut value used for the timeSOAP calculation.
            * n_max: the n_max value used for the timeSOAP calculation.
            * l_max: the l_max value used for the timeSOAP calculation.
            * respect_pbc: bust be True if trajectory has PBC
            * centers: selection of atoms used as centers for the SOAP
                calculation.
            * delay: the delay between frames on which timeSOAP is computed.
        """
        soap = dynsight.soap.saponify_trajectory(
            self.universe,
            soaprcut=r_cut,
            soapnmax=n_max,
            soaplmax=l_max,
            soap_respectpbc=respect_pbc,
            centers=centers,
            n_core=n_core,
        )
        tsoap = dynsight.soap.timesoap(soap, delay=delay)
        attr_dict = {
            "r_cut": r_cut,
            "n_max": n_max,
            "l_max": l_max,
            "respect_pbc": respect_pbc,
            "centers": centers,
            "delay": delay,
        }
        return Insight(dataset=tsoap, meta=attr_dict)
