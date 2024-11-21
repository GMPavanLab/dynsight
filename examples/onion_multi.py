"""Example script for running dynsight.onion.onion_multi."""

import matplotlib.pyplot as plt
import numpy as np

from dynsight import onion


def main() -> None:
    """Run the example.

    The data clustered are the MD trajectories of two molecules moving in
    a 2-dimensional free energy landscape, with four Gaussian minima, under
    Langevin dynsmics.

    Use git clone git@github.com:matteobecchi/onion_example_files.git
    to download example datasets.
    """
    ### Set the path to where the example files are located
    path_to_input_data = (
        "onion_example_files/data/multivariate_time-series.npy"
    )

    ### Load the input data -
    ### it's an array of shape (n_dims, n_particles, n_frames)
    input_data = np.load(path_to_input_data)
    n_frames = input_data.shape[2]

    """ STEP 1: CLUSTERING WITH A SINGLE TIME RESOLUTION
    Chose the time resolution --> the length of the windows in which the
    time-series will be divided. This is the minimum lifetime required for
    a state to be considered stable."""
    tau_window = 10
    bins = 25  # For mutlivariate clustering, setting bins is often important
    n_windows = int(n_frames / tau_window)  # Number of windows

    ### The input array has to be (n_parrticles * n_windows,
    ### tau_window * n_dims)
    ### because each window is trerated as a single data-point
    reshaped_data = onion.helpers.reshape_from_dnt(input_data, tau_window)

    ### onion_multi() returns the list of states and the label for each
    ### signal window
    state_list, labels = onion.onion_multi(reshaped_data, bins=bins)

    ### These functions are examples of how to visualize the results
    onion.plot.plot_output_multi(
        "Fig1.png", input_data, state_list, labels, tau_window
    )
    onion.plot.plot_one_trj_multi(
        "Fig2.png", 0, tau_window, input_data, labels
    )
    onion.plot.plot_medoids_multi("Fig3.png", tau_window, input_data, labels)
    onion.plot.plot_state_populations("Fig4.png", n_windows, labels)
    onion.plot.plot_sankey("Fig5.png", labels, n_windows, [100, 200, 300, 400])

    """ STEP 2: CLUSTERING THE WHOLE RANGE OF TIME RESOLUTIONS
    This allows to select the optimal time resolution for the analysis,
    avoiding an a priori choice."""
    tau_window_list = np.geomspace(3, 10000, 20, dtype=int)

    tra = np.zeros((len(tau_window_list), 3))  # List of number of states and
    # ENV0 population for each tau_window
    pop_list = []  # List of the states' population for each tau_window

    for i, tau_window in enumerate(tau_window_list):
        reshaped_data = onion.helpers.reshape_from_dnt(input_data, tau_window)

        state_list, labels = onion.onion_multi(reshaped_data, bins=bins)

        list_pop = [state.perc for state in state_list]
        list_pop.insert(0, 1 - np.sum(np.array(list_pop)))  # Add ENV0 fraction

        tra[i][0] = tau_window
        tra[i][1] = len(state_list)
        tra[i][2] = list_pop[0]
        pop_list.append(list_pop)

    ### These functions are examples of how to visualize the results
    onion.plot.plot_time_res_analysis("Fig6.png", tra)
    onion.plot.plot_pop_fractions("Fig7.png", pop_list, tra)

    plt.show()


if __name__ == "__main__":
    main()
