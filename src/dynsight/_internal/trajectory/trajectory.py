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
from MDAnalysis.coordinates.memory import MemoryReader

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
        trj: Trj,
        r_cut: float,
        selection: str = "all",
        num_processes: int = 1,
    ) -> Insight:
        """Average the descripotor over the neighboring particles.

        The returned Insight contains the following meta: sp_av_r_cut,
        selection.
        """
        averaged_dataset = dynsight.analysis.spatialaverage(
            universe=trj.universe,
            descriptor_array=self.dataset,
            selection=selection,
            cutoff=r_cut,
            trajslice=trj.trajslice,
            num_processes=num_processes,
        )
        return Insight(
            dataset=averaged_dataset,
            meta={"sp_av_r_cut": r_cut, "selection": selection},
        )

    def get_time_correlation(
        self,
        max_delay: int | None = None,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Self time correlation function of the time-series signal."""
        return dynsight.analysis.self_time_correlation(
            self.dataset,
            max_delay,
        )

    def get_angular_velocity(self, delay: int) -> Insight:
        """Computes the angular displacement of a vectorial descriptor."""
        if self.dataset.ndim != UNIVAR_DIM + 1:
            msg = "dataset.ndim != 3."
            raise ValueError(msg)
        theta = dynsight.soap.timesoap(self.dataset, delay=delay)
        attr_dict = self.meta.copy()
        attr_dict.update({"delay": delay})
        return Insight(dataset=theta, meta=attr_dict)

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

    def get_onion_smooth(
        self,
        delta_t: int,
        bins: str | int = "auto",
        number_of_sigmas: float = 3.0,
        max_area_overlap: float = 0.8,
    ) -> OnionSmoothInsight:
        """Perform smooth onion clustering.

        The returned OnionInsight contains the following meta: delta_t, bins,
        number_of_sigma, max_area_overlap.
        """
        if self.dataset.ndim == UNIVAR_DIM:
            onion_clust = dynsight.onion.OnionUniSmooth(
                delta_t=delta_t,
                bins=bins,
                number_of_sigmas=number_of_sigmas,
                max_area_overlap=max_area_overlap,
            )
        else:
            onion_clust = dynsight.onion.OnionMultiSmooth(
                delta_t=delta_t,
                bins=bins,
                number_of_sigmas=number_of_sigmas,
            )

        onion_clust.fit(self.dataset)

        return OnionSmoothInsight(
            labels=onion_clust.labels_,
            state_list=onion_clust.state_list_,
            meta={
                "delta_t": delta_t,
                "bins": bins,
                "number_of_sigmas": number_of_sigmas,
                "max_area_overlap": max_area_overlap,
            },
        )

    def get_onion_analysis(
        self,
        delta_t_min: int = 1,
        delta_t_max: int | None = None,
        delta_t_num: int = 20,
        fig1_path: Path | None = None,
        fig2_path: Path | None = None,
        bins: str | int = "auto",
        number_of_sigmas: float = 3.0,
        max_area_overlap: float = 0.8,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
        """Perform the full onion time resolution analysis.

        Note: this method uses the "onion smooth" functions (see documentation
        for details).

        Parameters:
            delta_t_min: Smaller value for delta_t_list.

            delta_t_max: Larger value for delta_t_list,

            delta_t_num: Number of values in delta_t_list,

            fig1_path: If is not None, the time resolution analysis plot is
                saved in this location.

            fig2_path: If is not None, the populations fractions plot is
                saved in this location.

            bins: The 'bins' parameter for onion clustering.

            number_of_sigmas: The 'number_of_sigmas' parameter for onion
                clustering.

            max_area_overlap: The 'max_area_overlap' parameter for onion
                clustering.

        Returns:
            delta_t_list: The list of delta_t used.

            n_clust: The number of clusters at each delta_t.

            unclass_frac: The fraction of unclassified data at each delta_t.
        """
        if delta_t_max is None:
            delta_t_max = self.dataset.shape[1]
        delta_t_list = np.unique(
            np.geomspace(delta_t_min, delta_t_max, delta_t_num, dtype=int)
        )
        n_clust = np.zeros(delta_t_list.size, dtype=int)
        unclass_frac = np.zeros(delta_t_list.size)
        list_of_pop = []

        for i, delta_t in enumerate(delta_t_list):
            on_cl = self.get_onion_smooth(
                delta_t,
                bins,
                number_of_sigmas,
                max_area_overlap,
            )
            n_clust[i] = len(on_cl.state_list)
            unclass_frac[i] = np.sum(on_cl.labels == -1) / self.dataset.size
            list_of_pop.append(
                [
                    np.sum(on_cl.labels == i) / self.dataset.size
                    for i in np.unique(on_cl.labels)
                ]
            )

        tra = np.array([delta_t_list, n_clust, unclass_frac]).T
        if fig1_path is not None:
            dynsight.onion.plot_smooth.plot_time_res_analysis(fig1_path, tra)
        if fig2_path is not None:
            dynsight.onion.plot_smooth.plot_pop_fractions(
                fig2_path, list_of_pop, tra
            )

        return delta_t_list, n_clust, unclass_frac


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
        data = {
            "labels": self.labels.tolist(),
            "reshaped_data": self.reshaped_data.tolist(),
            "meta": self.meta,
        }

        new_state_list = []
        for state in self.state_list:
            tmp = {}
            for f in fields(state):
                value = getattr(state, f.name)
                if isinstance(value, np.ndarray):
                    tmp[f.name] = value.tolist()
                else:
                    tmp[f.name] = value
            new_state_list.append(tmp)

        data["state_list"] = new_state_list
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
class OnionSmoothInsight(ClusterInsight):
    """Contains a smooth onion-clustering analysis.

    Attributes:
        labels: The labels assigned by the clustering algorithm.
        state_list: List of the onion-clustering Gaussian states.
        meta: A dictionary containing the relevant parameters.
    """

    state_list: list[StateUni] | list[StateMulti]
    meta: dict[str, Any] = field(default_factory=dict)

    def dump_to_json(self, file_path: Path) -> None:
        """Save the OnionSmoothInsight object as .json file."""
        data = {
            "labels": self.labels.tolist(),
            "meta": self.meta,
        }

        new_state_list = []
        for state in self.state_list:
            tmp = {}
            for f in fields(state):
                value = getattr(state, f.name)
                if isinstance(value, np.ndarray):
                    tmp[f.name] = value.tolist()
                else:
                    tmp[f.name] = value
            new_state_list.append(tmp)

        data["state_list"] = new_state_list
        with file_path.open("w") as file:
            json.dump(data, file, indent=4)

    @classmethod
    def load_from_json(cls, file_path: Path) -> OnionSmoothInsight:
        """Load the OnionSmoothInsight object from .json file.

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
            meta=data.get("meta"),
        )

    def plot_output(self, file_path: Path, data_insight: Insight) -> None:
        """Plot the overall onion clustering result."""
        if data_insight.dataset.ndim == UNIVAR_DIM:
            dynsight.onion.plot_smooth.plot_output_uni(
                file_path,
                data_insight.dataset,
                self.state_list,
            )
        else:
            dynsight.onion.plot_smooth.plot_output_multi(
                file_path,
                data_insight.dataset,
                self.state_list,
                self.labels,
            )

    def plot_one_trj(
        self,
        file_path: Path,
        data_insight: Insight,
        particle_id: int,
    ) -> None:
        """Plot one particle's trajectory colored according to clustering."""
        if data_insight.dataset.ndim == UNIVAR_DIM:
            dynsight.onion.plot_smooth.plot_one_trj_uni(
                file_path,
                particle_id,
                data_insight.dataset,
                self.labels,
            )
        else:
            dynsight.onion.plot_smooth.plot_one_trj_multi(
                file_path,
                particle_id,
                data_insight.dataset,
                self.labels,
            )

    def plot_state_populations(
        self,
        file_path: Path,
    ) -> None:
        """Plot each state's population along the trajectory."""
        dynsight.onion.plot_smooth.plot_state_populations(
            file_path,
            self.labels,
        )

    def plot_sankey(
        self,
        file_path: Path,
        frame_list: list[int],
    ) -> None:
        """Plot the Sankey diagram of the onion clustering."""
        dynsight.onion.plot_smooth.plot_sankey(
            file_path,
            self.labels,
            frame_list,
        )

    def dump_colored_trj(self, trj: Trj, file_path: Path) -> None:
        """Save an .xyz file with the clustering labels for each atom."""
        trajslice = slice(None) if trj.trajslice is None else trj.trajslice

        n_frames = len(trj.universe.trajectory[trajslice])
        n_atoms = len(trj.universe.atoms)
        lab_new = self.labels + 2

        if self.labels.shape != (n_atoms, n_frames):
            msg = (
                f"Shape mismatch: Trj should have {self.labels.shape[0]} "
                f"atoms, {self.labels.shape[0]} frames, but has {n_atoms} "
                f"atoms, {n_frames} frames."
            )
            raise ValueError(msg)

        with file_path.open("w") as f:
            for i, ts in enumerate(trj.universe.trajectory[trajslice]):
                f.write(f"{n_atoms}\n")
                f.write(f"Frame {i}\n")
                for atom_idx in range(n_atoms):
                    label = str(lab_new[atom_idx, i])
                    x, y, z = ts.positions[atom_idx]
                    f.write(
                        f"{trj.universe.atoms[atom_idx].name} {x:.5f}"
                        f" {y:.5f} {z:.5f} {label}\n"
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
    trajslice: slice | None = None

    @classmethod
    def init_from_universe(cls, universe: MDAnalysis.Universe) -> Trj:
        """Initialize Trj object from MDAnalysis.Universe.

        See https://docs.mdanalysis.org/2.9.0/documentation_pages/core/universe.html#MDAnalysis.core.universe.Universe.
        """
        return Trj(universe)

    @classmethod
    def init_from_xyz(cls, traj_file: Path, dt: float) -> Trj:
        """Initialize Trj object from .xyz file.

        See https://docs.mdanalysis.org/2.9.0/documentation_pages/core/universe.html#MDAnalysis.core.universe.Universe.

        Parameters:
        dt: the trajectory's time-step.
        """
        universe = MDAnalysis.Universe(traj_file, dt=dt)
        return Trj(universe)

    @classmethod
    def init_from_xtc(cls, traj_file: Path, topo_file: Path) -> Trj:
        """Initialize Trj object from .gro and .xtc files.

        See https://docs.mdanalysis.org/2.9.0/documentation_pages/core/universe.html#MDAnalysis.core.universe.Universe.
        """
        universe = MDAnalysis.Universe(topo_file, traj_file)
        return Trj(universe)

    def get_coordinates(self, selection: str) -> NDArray[np.float64]:
        """Returns the coordinates as an array.

        The array has shape (n_frames, n_atoms, n_coordinates).
        """
        atoms = self.universe.select_atoms(selection)
        trajslice = slice(None) if self.trajslice is None else self.trajslice

        return np.array(
            [
                atoms.positions.copy()
                for ts in self.universe.trajectory[trajslice]
            ]
        )

    def with_slice(self, trajslice: slice | None) -> Trj:
        """Returns a Trj with a different frames' slice."""
        return Trj(self.universe, trajslice=trajslice)

    def get_slice(self, start: int, stop: int, step: int) -> Trj:
        """Returns a Trj with a subset of frames.

        .. warning::

            This function could fill up the memory in case of large
            trajectories and it's deprecated. Use Trj.with_slice() instead.
        """
        n_atoms = self.universe.atoms.n_atoms

        # Get array of positions from all but the last frame
        frame_indices = list(range(start, stop, step))
        coords = np.empty((len(frame_indices), n_atoms, 3), dtype=np.float32)
        for i, ts in enumerate(self.universe.trajectory[start:stop:step]):
            coords[i] = ts.positions

        mem_reader = MemoryReader(coords, order="fac")
        u_new = MDAnalysis.Universe(topology=self.universe._topology)  # noqa: SLF001
        u_new.trajectory = mem_reader

        return Trj(u_new)

    def get_coord_number(
        self,
        r_cut: float,
        selection: str = "all",
    ) -> Insight:
        """Compute coordination number on the trajectory.

        The returned Insight contains the following meta: r_cut, selection.
        """
        neigcounts = dynsight.lens.list_neighbours_along_trajectory(
            input_universe=self.universe,
            cutoff=r_cut,
            selection=selection,
            trajslice=self.trajslice,
        )
        _, nn, *_ = dynsight.lens.neighbour_change_in_time(neigcounts)
        return Insight(
            dataset=nn.astype(np.float64),
            meta={"r_cut": r_cut, "selection": selection},
        )

    def get_lens(self, r_cut: float, selection: str = "all") -> Insight:
        """Compute LENS on the trajectory.

        The returned Insight contains the following meta: r_cut, selection.
        """
        neigcounts = dynsight.lens.list_neighbours_along_trajectory(
            input_universe=self.universe,
            cutoff=r_cut,
            selection=selection,
            trajslice=self.trajslice,
        )
        lens, *_ = dynsight.lens.neighbour_change_in_time(neigcounts)
        return Insight(
            dataset=lens[:, 1:],
            meta={"r_cut": r_cut, "selection": selection},
        )

    def get_soap(
        self,
        r_cut: float,
        n_max: int,
        l_max: int,
        selection: str = "all",
        centers: str = "all",
        respect_pbc: bool = True,
        n_core: int = 1,
    ) -> Insight:
        """Compute SOAP on the trajectory.

        The returned Insight contains the following meta: r_cut, n_max, l_max,
        respect_pbc, centers, selection.
        """
        soap = dynsight.soap.saponify_trajectory(
            self.universe,
            soaprcut=r_cut,
            soapnmax=n_max,
            soaplmax=l_max,
            selection=selection,
            soap_respectpbc=respect_pbc,
            centers=centers,
            n_core=n_core,
            trajslice=self.trajslice,
        )
        attr_dict = {
            "r_cut": r_cut,
            "n_max": n_max,
            "l_max": l_max,
            "respect_pbc": respect_pbc,
            "selection": selection,
            "centers": centers,
        }
        return Insight(dataset=soap, meta=attr_dict)

    def get_timesoap(
        self,
        r_cut: float,
        n_max: int,
        l_max: int,
        selection: str = "all",
        centers: str = "all",
        respect_pbc: bool = True,
        delay: int = 1,
        n_core: int = 1,
    ) -> Insight:
        """Compute timeSOAP on the trajectory.

        The returned Insight contains the following meta: r_cut, n_max, l_max,
        respect_pbc, centers, selection, delay.

        If return_soap = True, returns also an Insight with the SOAP dataset.
        """
        soap = self.get_soap(
            r_cut=r_cut,
            n_max=n_max,
            l_max=l_max,
            respect_pbc=respect_pbc,
            selection=selection,
            centers=centers,
            n_core=n_core,
        )
        return soap.get_angular_velocity(delay=delay)

    def get_rdf(
        self,
        distances_range: list[float],
        s1: str = "all",
        s2: str = "all",
        exclusion_block: list[int] | None = None,
        nbins: int = 200,
        norm: Literal["rdf", "density", "none"] = "rdf",
    ) -> Insight:
        """Compute the radial distribution function g(r).

        See https://docs.mdanalysis.org/1.1.1/documentation_pages/analysis/rdf.html.

        The returned Insight contains the following meta: distances_range, s1,
        s2, exclusion_block, nbins, norm.
        """
        trajslice = slice(None) if self.trajslice is None else self.trajslice
        bins, rdf = dynsight.analysis.compute_rdf(
            universe=self.universe,
            distances_range=distances_range,
            s1=s1,
            s2=s2,
            exclusion_block=exclusion_block,
            nbins=nbins,
            norm=norm,
            start=trajslice.start,
            stop=trajslice.stop,
            step=trajslice.step,
        )
        dataset = np.array([bins, rdf])
        attr_dict = {
            "distances_range": distances_range,
            "s1": s1,
            "s2": s2,
            "exclusion_block": exclusion_block,
            "nbins": nbins,
            "norm": norm,
        }
        return Insight(dataset=dataset, meta=attr_dict)
