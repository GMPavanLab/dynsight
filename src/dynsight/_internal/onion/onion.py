"""onion-clustering package."""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Any

import numpy as np
from onion_clustering import main, main_2d


class StateMulti:
    """Stores the info regarding a multivariate states."""

    def __init__(
        self,
        mean: np.ndarray[float, Any],
        stddev: np.ndarray[float, Any],
        perc: float,
    ) -> None:
        """Creates the state."""
        self.mean = mean
        self.stddev = stddev
        self.perc = perc


class OnionUni:
    """Class to perform onion-clustering on univariate timeseries."""

    def __init__(
        self,
        path_to_input: str,
        tau_w: int,
        output_path: Path | None = None,
        t_smooth: int = 1,
        t_delay: int = 0,
        t_conv: float = 1,
        t_units: str = "frames",
        example_id: int = 0,
        num_tau_w: int = 20,
        min_tau_w: int = 2,
        max_tau_w: int = -1,
        min_t_smooth: int = 1,
        max_t_smooth: int = 5,
        step_t_smooth: int = 1,
        bins: int = -1,
    ) -> None:
        """Takes as input the parameters and perform the analysis.

        Parameters:
            path_to_input (str):
                the path to the file containing the input time-series.
            tau_w (int):
                the length of the time window (in number of frames).
            output_path (Path | None):
                the folder where to save all the output. If None, creates
                a folder called onion_output and runs the analysis there.
            t_smooth (int, optional):
                the length of the smoothing window (in number of frames) for
                the moving average. A value of t_smooth = 1 correspond to no
                smoothing. Default is 1.
            t_delay (int, optional):
                is for ignoring the first tau_delay frames of the trajectory.
                Default is 0.
            t_conv (int, optional):
                converts number of frames in time units. Default is 1.
            t_units (str, optional):
                a string indicating the time units. Default is 'frames'.
            example_id (int, optional):
                plots the trajectory of the molecule with this ID, colored
                according to the identified states. Default is 0.
            num_tau_w (int, optional):
                the number of different tau_window values tested.
                Default is 20.
            min_tau_w (int, optional):
                the smaller tau_window value tested. It has to be larger
                that 1. Default is 2.
            max_tau_w (int, optional):
                the larger tau_window value tested. It has to be larger
                that 2. Default is the trj length.
            min_t_smooth (int, optional):
                the smaller t_smooth value tested. It has to be larger
                that 0. Default is 1.
            max_t_smooth (int, optional):
                the larger t_smooth value tested. It has to be larger
                that 0. Default is 5.
            step_t_smooth (int, optional):
                the step in the t_smooth values tested. It has to be larger
                that 0. Default is 1.
            bins (int, optional):
                the number of bins used to compute histograms. This should
                be used only if all the fits fail with the automatic binning.
        """
        ### Input parameters ###
        self._path_to_input = path_to_input
        self._tau_w = tau_w
        self._t_smooth = t_smooth
        self._t_delay = t_delay
        self._t_conv = t_conv
        self._t_units = t_units
        self._example_id = example_id
        self._num_tau_w = num_tau_w
        self._min_tau_w = min_tau_w
        self._max_tau_w = max_tau_w
        self._min_t_smooth = min_t_smooth
        self._max_t_smooth = max_t_smooth
        self._step_t_smooth = step_t_smooth
        self._bins = bins

        if output_path is None:
            self._output_path = Path("./onion_output")
        else:
            self._output_path = output_path

        ### Clustering output ###
        self._number_of_states: np.ndarray[int, Any]
        self._fraction_0: np.ndarray[float, Any]
        self._state_list: np.ndarray[float, Any]
        self._clustering: np.ndarray[int, Any]

    def run(self) -> None:
        """Run the code.

        Creates the files necessary for input and then runs the clustering.
        The process will change directory to run the analysis, then it will
        come back to the cwd. All the output files will be found in
        cwd/onion_output, or output_path if specified.
        """
        if self._output_path.exists():
            shutil.rmtree(self._output_path)
        self._output_path.mkdir()

        ### Create the 'data_directory.txt' file ###
        file_path = self._output_path / "data_directory.txt"
        with file_path.open(mode="w") as file:
            file.write(self._path_to_input)

        ### Create the 'input_parameter.txt' file ###
        file_path = self._output_path / "input_parameters.txt"
        with file_path.open(mode="w") as file:
            file.write(f"tau_window\t{self._tau_w}\n")
            file.write(f"t_smooth\t{self._t_smooth}\n")
            file.write(f"t_delay\t{self._t_delay}\n")
            file.write(f"t_conv\t{self._t_conv}\n")
            file.write(f"t_units\t{self._t_units}\n")
            file.write(f"example_ID\t{self._example_id}\n")
            file.write(f"num_tau_w\t{self._num_tau_w}\n")
            file.write(f"min_tau_w\t{self._min_tau_w}\n")
            file.write(f"min_t_smooth\t{self._min_t_smooth}\n")
            file.write(f"max_t_smooth\t{self._max_t_smooth}\n")
            file.write(f"step_t_smooth\t{self._step_t_smooth}\n")
            if self._max_tau_w != -1:
                file.write(f"max_tau_w\t{self._max_tau_w}\n")
            if self._bins != -1:
                file.write(f"bins\t{self._bins}\n")

        original_dir = Path.cwd()
        os.chdir(self._output_path)
        try:
            self._clustering = main.main()
            self._number_of_states = np.loadtxt(
                "number_of_states.txt", dtype=int
            )
            self._fraction_0 = np.loadtxt("fraction_0.txt")
            self._state_list = np.loadtxt("final_states.txt")
        finally:
            os.chdir(original_dir)

    def get_number_of_states(self) -> np.ndarray[int, Any]:
        """Returns the number_of_states matrix.

        Each row is a different tau_window, first element is tau_window
        then each column is a different t_smooth.
        """
        return self._number_of_states

    def get_fraction_0(self) -> np.ndarray[float, Any]:
        """Returns the fraction_0 matrix.

        Each row is a different tau_window, first element is tau_window
        then each column is a different t_smooth.
        """
        return self._fraction_0

    def get_state_list(self) -> np.ndarray[float, Any]:
        """Returns the list of states.

        For each state, the 4 components are, respectively,
        mean, standard deviation, area and percentage of data points.
        """
        return self._state_list

    def get_clustering(self) -> np.ndarray[int, Any]:
        """Returns the single point clustering.

        Each row is a particle, each column a simulation frame.
        """
        return self._clustering


class OnionMulti:
    """Class to perform onion-clustering on multivariate timeseries."""

    def __init__(
        self,
        path_to_input: str,
        tau_w: int,
        output_path: Path | None = None,
        t_smooth: int = 1,
        t_delay: int = 0,
        t_conv: float = 1,
        t_units: str = "frames",
        example_id: int = 0,
        num_tau_w: int = 20,
        min_tau_w: int = 2,
        max_tau_w: int = -1,
        min_t_smooth: int = 1,
        max_t_smooth: int = 5,
        step_t_smooth: int = 1,
        bins: int = -1,
    ) -> None:
        """Takes as input the parameters and perform the analysis.

        Parameters:
            path_to_input (list[str]):
                list of path to the files containing the components of the
                input time-series.
            tau_w (int):
                the length of the time window (in number of frames).
            output_path (Path | None):
                the folder where to save all the output. If None, creates
                a folder called onion_output and runs the analysis there.
            t_smooth (int, optional):
                the length of the smoothing window (in number of frames) for
                the moving average. A value of t_smooth = 1 correspond to no
                smoothing. Default is 1.
            t_delay (int, optional):
                is for ignoring the first tau_delay frames of the trajectory.
                Default is 0.
            t_conv (int, optional):
                converts number of frames in time units. Default is 1.
            t_units (str, optional):
                a string indicating the time units. Default is 'frames'.
            example_id (int, optional):
                plots the trajectory of the molecule with this ID, colored
                according to the identified states. Default is 0.
            num_tau_w (int, optional):
                the number of different tau_window values tested.
                Default is 20.
            min_tau_w (int, optional):
                the smaller tau_window value tested. It has to be larger
                that 1. Default is 2.
            max_tau_w (int, optional):
                the larger tau_window value tested. It has to be larger
                that 2. Default is the largest possible window.
            min_t_smooth (int, optional):
                the smaller t_smooth value tested. It has to be larger
                that 0. Default is 1.
            max_t_smooth (int, optional):
                the larger t_smooth value tested. It has to be larger
                that 0. Default is 5.
            step_t_smooth (int, optional):
                the step in the t_smooth values tested. It has to be larger
                that 0. Default is 1.
            bins (int, optional):
                the number of bins used to compute histograms. This should
                be used only if all the fits fail with the automatic binning.
        """
        ### Input parameters ###
        self._path_to_input = path_to_input
        self._tau_w = tau_w
        self._t_smooth = t_smooth
        self._t_delay = t_delay
        self._t_conv = t_conv
        self._t_units = t_units
        self._example_id = example_id
        self._num_tau_w = num_tau_w
        self._min_tau_w = min_tau_w
        self._max_tau_w = max_tau_w
        self._min_t_smooth = min_t_smooth
        self._max_t_smooth = max_t_smooth
        self._step_t_smooth = step_t_smooth
        self._bins = bins

        if output_path is None:
            self._output_path = Path("./onion_output")
        else:
            self._output_path = output_path

        ### Clustering output ###
        self._number_of_states: np.ndarray[int, Any]
        self._fraction_0: np.ndarray[float, Any]
        self._state_list: list[StateMulti]
        self._clustering: np.ndarray[int, Any]

    def run(self) -> None:  # noqa: PLR0915
        """Run the code.

        Creates the files necessary for input and then runs the clustering.
        The process will change directory to run the analysis, then it will
        come back to the cwd. All the output files will be found in
        cwd/onion_output, or output_path if specified.
        """
        if self._output_path.exists():
            shutil.rmtree(self._output_path)
        self._output_path.mkdir()

        ### Create the 'data_directory.txt' file ###
        file_path = self._output_path / "data_directory.txt"
        with file_path.open(mode="w") as file:
            file.write(self._path_to_input)

        ### Create the 'input_parameter.txt' file ###
        file_path = self._output_path / "input_parameters.txt"
        with file_path.open(mode="w") as file:
            file.write(f"tau_window\t{self._tau_w}\n")
            file.write(f"t_smooth\t{self._t_smooth}\n")
            file.write(f"t_delay\t{self._t_delay}\n")
            file.write(f"t_conv\t{self._t_conv}\n")
            file.write(f"t_units\t{self._t_units}\n")
            file.write(f"example_ID\t{self._example_id}\n")
            file.write(f"num_tau_w\t{self._num_tau_w}\n")
            file.write(f"min_tau_w\t{self._min_tau_w}\n")
            file.write(f"min_t_smooth\t{self._min_t_smooth}\n")
            file.write(f"max_t_smooth\t{self._max_t_smooth}\n")
            file.write(f"step_t_smooth\t{self._step_t_smooth}\n")
            if self._max_tau_w != -1:
                file.write(f"max_tau_w\t{self._max_tau_w}\n")
            if self._bins != -1:
                file.write(f"bins\t{self._bins}\n")

        original_dir = Path.cwd()
        os.chdir(self._output_path)
        try:
            self._clustering = main_2d.main()
            self._number_of_states = np.loadtxt(
                "number_of_states.txt", dtype=int
            )
            self._fraction_0 = np.loadtxt("fraction_0.txt")

            file_path = Path("final_states.txt")
            with file_path.open(mode="r") as file:
                state_list = []
                num_cols_in_2d = 5
                num_cols_in_3d = 7
                for line in file:
                    line1 = (
                        line.replace("[", "").replace("]", "").replace(",", "")
                    )
                    line2 = line1.strip().split(" ")
                    if len(line2) == num_cols_in_2d:
                        mean = np.array(
                            [float(line2[0][1:]), float(line2[1][:-1])]
                        )
                        stddev = np.array(
                            [float(line2[2][1:]), float(line2[3][:-1])]
                        )
                        perc = float(line2[4])
                        state_list.append(StateMulti(mean, stddev, perc))
                    elif len(line2) == num_cols_in_3d:
                        mean = np.array(
                            [
                                float(line2[0][1:]),
                                float(line2[1]),
                                float(line2[2][:-1]),
                            ]
                        )
                        stddev = np.array(
                            [
                                float(line2[3][1:]),
                                float(line2[4]),
                                float(line2[5][:-1]),
                            ]
                        )
                        perc = float(line2[6])
                        state_list.append(StateMulti(mean, stddev, perc))
                self._state_list = state_list
        finally:
            os.chdir(original_dir)

    def get_number_of_states(self) -> np.ndarray[int, Any]:
        """Returns the number_of_states matrix.

        Each row is a different tau_window, first element is tau_window
        then each column is a different t_smooth.
        """
        return self._number_of_states

    def get_fraction_0(self) -> np.ndarray[float, Any]:
        """Returns the fraction_0 matrix.

        Each row is a different tau_window, first element is tau_window
        then each column is a different t_smooth.
        """
        return self._fraction_0

    def get_state_list(self) -> list[StateMulti]:
        """Returns the list of states.

        For each state, the 3 components are, respectively,
        mean, standard deviation and percentage of data points.
        """
        return self._state_list

    def get_clustering(self) -> np.ndarray[int, Any]:
        """Returns the single point clustering.

        Each row is a particle, each column a simulation frame.
        """
        return self._clustering
