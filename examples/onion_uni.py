"""Example script for running dynsight.onion.onion_uni."""

import matplotlib.pyplot as plt
import numpy as np

from dynsight import onion
from dynsight.utilities import find_extrema_points


def main() -> None:
    """Run the example.

    The data clustered are the LENS values computed over a TIP4P/ICE
    MD trajectory at the solid/liquid coexistence point. The trajectory
    is sampled every 0.1 ns, and there are 2048 water molecules.
    For more information about this dataset or the LENS descriptor, see
    https://doi.org/10.1073/pnas.2300565120.

    Use `git clone git@github.com:matteobecchi/onion_example_files.git`
    to download example datasets.
    """
    ### Set the path to where the example files are located
    path_to_input_data = "onion_example_files/data/univariate_time-series.npy"

    ### Load the input data - it's an array of shape (n_particles, n_frames)
    ### The first LENS frame has to be removed because it's always zero
    input_data = np.load(path_to_input_data)[:, 1:]
    n_frames = input_data.shape[1]

    """ STEP 0: STATIC CLUSTERING
    Before using Onion Clustering, a simple pattern recognition analysis can
    be performed with dynsight, identifying the maxima of the data
    distribution. This analysis ignores the time correlations withing the
    data. The value of 'prominance' tunes the sensibility to the data
    histogram roughness. Plot the histogram to set the best value.

    The results is an array which lists the (x, y) value of each peak.
    """
    counts, bins = np.histogram(
        input_data.flatten(),
        bins=50,
        density=True,
    )
    _ = find_extrema_points(
        x_axis=bins[1:],
        y_axis=counts,
        extrema_type="max",
        prominence=0.2,
    )

    """ STEP 1: CLUSTERING WITH A SINGLE TIME RESOLUTION
    Chose the time resolution --> the length of the windows in which the
    time-series will be divided. This is the minimum lifetime required for
    a state to be considered stable."""
    tau_window = 5
    n_windows = int(n_frames / tau_window)  # Number of windows

    ### The input array needs to be (n_parrticles * n_windows, tau_window)
    ### because each window is trerated as a single data-point
    reshaped_data = onion.helpers.reshape_from_nt(input_data, tau_window)

    ### onion_uni() returns the list of states and the label for each
    ### signal window
    state_list, labels = onion.onion_uni(reshaped_data)

    ### These functions are examples of how to visualize the results
    onion.plot.plot_output_uni(
        "Fig1.png", reshaped_data, n_windows, state_list
    )
    onion.plot.plot_one_trj_uni(
        "Fig2.png", 1234, reshaped_data, labels, n_windows
    )
    onion.plot.plot_medoids_uni("Fig3.png", reshaped_data, labels)
    onion.plot.plot_state_populations("Fig4.png", n_windows, labels)
    onion.plot.plot_sankey("Fig5.png", labels, n_windows, [10, 20, 30, 40])

    """ STEP 2: CLUSTERING THE WHOLE RANGE OF TIME RESOLUTIONS
    This allows to select the optimal time resolution for the analysis,
    avoiding an a priori choice."""
    tau_windows = np.unique(np.geomspace(2, n_frames, num=20, dtype=int))

    tra = np.zeros((len(tau_windows), 3))  # List of number of states and
    # ENV0 population for each tau_window
    list_of_pop = []  # List of the states' population for each tau_window

    for i, tau_window in enumerate(tau_windows):
        reshaped_data = onion.helpers.reshape_from_nt(input_data, tau_window)

        state_list, labels = onion.onion_uni(reshaped_data)

        pop_list = [state.perc for state in state_list]
        pop_list.insert(0, 1 - np.sum(np.array(pop_list)))  # Add ENV0 fraction
        list_of_pop.append(pop_list)

        tra[i][0] = tau_window
        tra[i][1] = len(state_list)
        tra[i][2] = pop_list[0]

    ### These functions are examples of how to visualize the results
    onion.plot.plot_time_res_analysis("Fig6.png", tra)
    onion.plot.plot_pop_fractions("Fig7.png", list_of_pop, tra)

    plt.show()


if __name__ == "__main__":
    main()
