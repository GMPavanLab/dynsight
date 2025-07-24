from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

if TYPE_CHECKING:
    from pathlib import Path

    from numpy.typing import NDArray

    from dynsight.trajectory import Trj

import dynsight
from dynsight.logs import logger
from dynsight.trajectory import OnionInsight, OnionSmoothInsight

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
        """Save the Insight to a JSON file and  .npy file."""
        # Save dataset as .npy
        npy_path = file_path.with_suffix(".npy")
        np.save(npy_path, self.dataset.astype(np.float64))

        # Prepare JSON data
        json_data = {
            "dataset_file": npy_path.name,
            "meta": self.meta,
        }

        with file_path.open("w") as file:
            json.dump(json_data, file, indent=4)
        logger.log(f"Insight saved to {file_path} and dataset to {npy_path}.")

    @classmethod
    def load_from_json(
        cls,
        file_path: Path,
        mmap_mode: Literal["r", "r+", "w+", "c"] | None = None,
    ) -> Insight:
        """Load the Insight object from .json file.

        Parameters:
            file_path:
                Path to the .json file.
            mmap_mode:
                If given, used as np.load(..., mmap_mode=mmap_mode) for memory
                mapping.

        Raises:
            ValueError: if required keys are missing.
        """
        with file_path.open("r") as file:
            data = json.load(file)

        dataset_file = data.get("dataset_file")
        if not dataset_file:
            msg = "'dataset_file' key not found in JSON file."
            logger.log(msg)
            raise ValueError(msg)

        dataset_path = file_path.with_name(dataset_file)
        dataset = np.load(dataset_path, mmap_mode=mmap_mode)

        logger.log(
            f"Insight loaded from {file_path}, dataset from {dataset_path}."
        )

        return cls(
            dataset=dataset,
            meta=data.get("meta", {}),
        )

    def spatial_average(
        self,
        trj: Trj,
        r_cut: float,
        selection: str = "all",
        num_processes: int = 1,
    ) -> Insight:
        """Average the descriptor over the neighboring particles.

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
        attr_dict = {"sp_av_r_cut": r_cut, "selection": selection}

        logger.log(f"Computed spatial average with args {attr_dict}.")
        return Insight(
            dataset=averaged_dataset,
            meta=attr_dict,
        )

    def get_time_correlation(
        self,
        max_delay: int | None = None,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Self time correlation function of the time-series signal."""
        attr_dict = {"max_delay": max_delay}
        logger.log(
            f"Computed time corrleation function with args {attr_dict}."
        )
        return dynsight.analysis.self_time_correlation(
            self.dataset,
            max_delay,
        )

    def get_angular_velocity(self, delay: int = 1) -> Insight:
        """Computes the angular displacement of a vectorial descriptor.

        Raises:
            ValueError if the dataset does not have the right dimensions.
        """
        if self.dataset.ndim != UNIVAR_DIM + 1:
            msg = "dataset.ndim != 3."
            logger.log(msg)
            raise ValueError(msg)
        theta = dynsight.soap.timesoap(self.dataset, delay=delay)
        attr_dict = self.meta.copy()
        attr_dict.update({"delay": delay})

        logger.log(f"Computed angular velocity with args {attr_dict}.")
        return Insight(dataset=theta, meta=attr_dict)

    def get_tica(
        self,
        lag_time: int,
        tica_dim: int,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], Insight]:
        """Perform tICA on trajectories from a many-body system.

        The attributes "lag_time" and "tica_dim" are added to the meta.

        Raises:
            ValueError if the dataset does not have the right dimensions.
        """
        if self.dataset.ndim != UNIVAR_DIM + 1:
            msg = "dataset.ndim != 3."
            logger.log(msg)
            raise ValueError(msg)

        relax_times, coeffs, tica = dynsight.descriptors.many_body_tica(
            self.dataset,
            lag_time=lag_time,
            tica_dim=tica_dim,
        )

        attr_dict = self.meta.copy()
        attr_dict.update({"lag_time": lag_time, "tica_dim": tica_dim})

        logger.log(f"Computed many-body tICA with args {attr_dict}.")

        return relax_times, coeffs, Insight(dataset=tica, meta=attr_dict)

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
        attr_dict = {
            "delta_t": delta_t,
            "bins": bins,
            "number_of_sigmas": number_of_sigmas,
        }

        logger.log(f"Performed onion clustering with args {attr_dict}.")
        return OnionInsight(
            labels=onion_clust.labels_,
            state_list=onion_clust.state_list_,
            reshaped_data=reshaped_data,
            meta=attr_dict,
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
        attr_dict = {
            "delta_t": delta_t,
            "bins": bins,
            "number_of_sigmas": number_of_sigmas,
            "max_area_overlap": max_area_overlap,
        }

        logger.log(f"Performed onion clustering smooth with args {attr_dict}.")
        return OnionSmoothInsight(
            labels=onion_clust.labels_,
            state_list=onion_clust.state_list_,
            meta=attr_dict,
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
            delta_t_min:
                Smaller value for delta_t_list.
            delta_t_max:
                Larger value for delta_t_list.
            delta_t_num:
                Number of values in delta_t_list.
            fig1_path:
                If is not None, the time resolution analysis plot is saved in
                this location.
            fig2_path:
                If is not None, the populations fractions plot is saved in
                this location.
            bins:
                The 'bins' parameter for onion clustering.
            number_of_sigmas:
                The 'number_of_sigmas' parameter for onion clustering.
            max_area_overlap:
                The 'max_area_overlap' parameter for onion clustering.

        Returns:
            tuple:
                * delta_t_list: The list of ∆t used.
                * n_clust: The number of clusters at each ∆t.
                * unclass_frac: The fraction of unclassified data at each ∆t.
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

        attr_dict = {
            "delta_t_min": delta_t_min,
            "delta_t_max": delta_t_max,
            "delta_t_num": delta_t_num,
            "fig1_path": fig1_path,
            "fig2_path": fig2_path,
            "bins": bins,
            "number_of_sigmas": number_of_sigmas,
            "max_area_overlap": max_area_overlap,
        }

        logger.log(
            f"Performed full onion clustering analysis with args {attr_dict}."
        )
        return delta_t_list, n_clust, unclass_frac
