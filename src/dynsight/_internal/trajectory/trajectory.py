from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

if TYPE_CHECKING:
    from pathlib import Path

    from numpy.typing import NDArray
    from tropea_clustering._internal.first_classes import StateMulti, StateUni

import MDAnalysis

import dynsight

UNIVAR_DIM = 2


@dataclass(frozen=True)
class Insight:
    """Contains an analysis perfomed on a trajectory.

    Attributes:
        dataset: The values of a some trajectory's descriptor.
        meta: A dictionary containing the relevant parameters.
    """

    dataset: NDArray[np.float64]
    meta: dict[str, Any] = field(default_factory=dict)

    def dump_to_json(self, file_path: Path) -> None:
        """Save the Insight object as .json file."""
        data = asdict(self)
        data["dataset"] = data["dataset"].tolist()
        with file_path.open("w") as file:
            json.dump(data, file, indent=4)

    @classmethod
    def load_from_json(cls, file_path: Path) -> Insight:
        """Load the Insight object from .json file.

        Raises:
            ValueError if the input file does not have a key "dataset".
        """
        with file_path.open("r") as file:
            data = json.load(file)

        if "dataset" not in data:
            msg = "'dataset' key not found in JSON file."
            raise ValueError(msg)

        return cls(
            dataset=np.array(data.get("dataset"), dtype=np.float64),
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

        The returned Insight contains the following meta: sp_av_r_cut,
        selection.
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
            meta={"sp_av_r_cut": r_cut, "selection": selection},
        )

    def get_onion(
        self,
        delta_t: int,
        bins: str | int = "auto",
        number_of_sigmas: float = 2.0,
    ) -> OnionInsight:
        """Perform onion clustering.

        The returned OnionInsight contains the following meta: delta_t, bins,
        number_of_sigma.
        """
        if self.dataset.ndim == UNIVAR_DIM:
            reshaped_data = dynsight.onion.helpers.reshape_from_nt(
                self.dataset, delta_t
            )
            onion_clust = dynsight.onion.OnionUni(
                bins=bins,
                number_of_sigmas=number_of_sigmas,
            )
        else:
            reshaped_data = dynsight.onion.helpers.reshape_from_dnt(
                self.dataset.transpose(2, 0, 1), delta_t
            )
            onion_clust = dynsight.onion.OnionMulti(
                ndims=self.dataset.ndim - 1,
                bins=bins,
                number_of_sigmas=number_of_sigmas,
            )

        onion_clust.fit(reshaped_data)

        return OnionInsight(
            labels=onion_clust.labels_,
            state_list=onion_clust.state_list_,
            reshaped_data=reshaped_data,
            meta={
                "delta_t": delta_t,
                "bins": bins,
                "number_of_sigmas": number_of_sigmas,
            },
        )


@dataclass(frozen=True)
class ClusterInsight:
    """Contains a clustering analysis.

    Attributes:
        labels: The labels assigned by the clustering algorithm.
    """

    labels: NDArray[np.int64]

    def dump_to_json(self, file_path: Path) -> None:
        """Save the ClusterInsight object as .json file."""
        data = asdict(self)
        data["labels"] = data["labels"].tolist()
        with file_path.open("w") as file:
            json.dump(data, file, indent=4)

    @classmethod
    def load_from_json(cls, file_path: Path) -> ClusterInsight:
        """Load the ClusterInsight object from .json file.

        Raises:
            ValueError if the input file does not have a key "labels".
        """
        with file_path.open("r") as file:
            data = json.load(file)
        if "labels" not in data:
            msg = "'labels' key not found in JSON file."
            raise ValueError(msg)
        return cls(labels=np.array(data.get("labels"), dtype=np.int64))


@dataclass(frozen=True)
class OnionInsight(ClusterInsight):
    """Contains an onion-clustering analysis.

    Attributes:
        labels: The labels assigned by the clustering algorithm.
        state_list: List of the onion-clustering Gaussian states.
        reshaped_data: The input data reshaped for onion-clustering.
        meta: A dictionary containing the relevant parameters.
    """

    state_list: list[StateUni] | list[StateMulti]
    reshaped_data: NDArray[np.float64]
    meta: dict[str, Any] = field(default_factory=dict)

    def dump_to_json(self, file_path: Path) -> None:
        """Save the OnionInsight object as .json file."""
        data = asdict(self)
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
        """Load the OnionInsight object from .json file.

        Raises:
            ValueError if the input file does not have a key "state_list".
        """
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
    """Contains a trajectory.

    Attributes:
        universe: a MDAnalysis.Universe containing the trajectory.

    .. warning::

        This class is under development. The name and type of the "universe"
        attribute may change in the future.
    """

    universe: MDAnalysis.Universe = field()

    @classmethod
    def init_from_universe(cls, universe: MDAnalysis.Universe) -> Trj:
        """Initialize Trj object from MDAnalysis.Universe."""
        return Trj(universe)

    @classmethod
    def init_from_xyz(cls, traj_file: Path, dt: float) -> Trj:
        """Initialize Trj object from .xyz file."""
        universe = MDAnalysis.Universe(traj_file, dt=dt)
        return Trj(universe)

    @classmethod
    def init_from_xtc(cls, traj_file: Path, topo_file: Path) -> Trj:
        """Initialize Trj object from .gro and .xtc files."""
        universe = MDAnalysis.Universe(topo_file, traj_file)
        return Trj(universe)

    def get_coordinates(self, selection: str) -> NDArray[np.float64]:
        """Returns the coordinates as an array.

        The array has shape (n_frames, n_atoms, n_coordinates).
        """
        atoms = self.universe.select_atoms(selection)
        return np.array(
            [atoms.positions.copy() for ts in self.universe.trajectory]
        )

    def get_lens(self, r_cut: float, neigh_count: bool = False) -> Insight:
        """Compute LENS on the trajectory.

        The returned Insight contains the following meta: r_cut, neigh_count.
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

        The returned Insight contains the following meta: r_cut, n_max, l_max,
        respect_pbc, centers.
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

        The returned Insight contains the following meta: r_cut, n_max, l_max,
        respect_pbc, centers, delay.
        """
        soap = self.get_soap(
            r_cut=r_cut,
            n_max=n_max,
            l_max=l_max,
            respect_pbc=respect_pbc,
            centers=centers,
            n_core=n_core,
        )
        tsoap = dynsight.soap.timesoap(soap.dataset, delay=delay)
        attr_dict = soap.meta
        attr_dict.update({"delay": delay})
        return Insight(dataset=tsoap, meta=attr_dict)

    def get_rdf(
        self,
        distances_range: list[float],
        s1: str = "all",
        s2: str = "all",
        exclusion_block: list[int] | None = None,
        nbins: int = 200,
        norm: Literal["rdf", "density", "none"] = "rdf",
        start: int | None = None,
        stop: int | None = None,
        step: int = 1,
    ) -> Insight:
        """Compute the radial distribution function g(r).

        The returned Insight contains the following meta: distances_range, s1,
        s2, exclusion_block, nbins, norm, start, stop, step.
        """
        bins, rdf = dynsight.analysis.compute_rdf(
            universe=self.universe,
            distances_range=distances_range,
            s1=s1,
            s2=s2,
            exclusion_block=exclusion_block,
            nbins=nbins,
            norm=norm,
            start=start,
            stop=stop,
            step=step,
        )
        dataset = np.array([bins, rdf])
        attr_dict = {
            "distances_range": distances_range,
            "s1": s1,
            "s2": s2,
            "exclusion_block": exclusion_block,
            "nbins": nbins,
            "norm": norm,
            "start": start,
            "stop": stop,
            "step": step,
        }
        return Insight(dataset=dataset, meta=attr_dict)
