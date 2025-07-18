from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from pathlib import Path

    from numpy.typing import NDArray
    from tropea_clustering._internal.first_classes import StateMulti, StateUni

    from dynsight.trajectory import Insight, Trj


import dynsight
from dynsight.logs import logger

UNIVAR_DIM = 2


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
        logger.log(f"ClusterInsight saved to {file_path}.")

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
            logger.log(msg)
            raise ValueError(msg)

        logger.log(f"ClusterInsight loaded from {file_path}.")
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
        logger.log(f"OnionInsight saved to {file_path}.")

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
            logger.log(msg)
            raise ValueError(msg)

        logger.log(f"OnionInsight loaded from {file_path}.")
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
        attr_dict = {"file_path": file_path}
        logger.log(f"plot_output() with args {attr_dict}.")

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

        attr_dict = {"file_path": file_path, "particle_id": particle_id}
        logger.log(f"plot_one_trj() with args {attr_dict}.")

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
        attr_dict = {"file_path": file_path}
        logger.log(f"plot_medoids() with args {attr_dict}.")

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
        attr_dict = {"file_path": file_path}
        logger.log(f"plot_state_populations() with args {attr_dict}.")

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

        attr_dict = {"file_path": file_path, "frame_list": frame_list}
        logger.log(f"plot_state_populations() with args {attr_dict}.")


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
        logger.log(f"OnionSmoothInsight saved to {file_path}.")

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
            logger.log(msg)
            raise ValueError(msg)

        logger.log(f"OnionSmoothInsight loaded from {file_path}.")
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
        attr_dict = {"file_path": file_path}
        logger.log(f"plot_output() with args {attr_dict}.")

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
        attr_dict = {"file_path": file_path, "particle_id": particle_id}
        logger.log(f"plot_one_trj() with args {attr_dict}.")

    def plot_state_populations(
        self,
        file_path: Path,
    ) -> None:
        """Plot each state's population along the trajectory."""
        dynsight.onion.plot_smooth.plot_state_populations(
            file_path,
            self.labels,
        )
        attr_dict = {"file_path": file_path}
        logger.log(f"plot_state_populations() with args {attr_dict}.")

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
        attr_dict = {"file_path": file_path, "frame_list": frame_list}
        logger.log(f"plot_state_populations() with args {attr_dict}.")

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
            logger.log(msg)
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
        logger.log(f"Colored trj saved to {file_path}.")
