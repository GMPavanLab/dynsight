"""Example script for running dynsight.onion.onion_multi."""

import matplotlib.pyplot as plt
import numpy as np

from dynsight.onion import onion_multi
from dynsight.onion.plot import (
    plot_medoids_multi,
    plot_one_trj_multi,
    plot_output_multi,
    plot_pop_fractions,
    plot_sankey,
    plot_state_populations,
    plot_time_res_analysis,
)


def main() -> None:
    """Run the example.

    Use git clone git@github.com:matteobecchi/onion_example_files.git
    to download example datasets.
    """
    path_to_input_data = (
        "onion_example_files/data/multivariate_time-series.npy"
    )

    ### Load the input data -
    ### it's an array of shape (n_dims, n_particles, n_frames)
    input_data = np.load(path_to_input_data)
    n_particles = input_data.shape[1]
    n_frames = input_data.shape[2]

    ### CLUSTERING WITH A SINGLE TIME RESOLUTION ###
    ### Chose the time resolution --> the length of the windows in which the
    ### time-series will be divided
    tau_window = 10
    bins = 25  # For mutlivariate clustering, setting bins is often important
    n_windows = int(n_frames / tau_window)  # Number of windows

    ### The input array has to be (n_parrticles * n_windows,
    ### tau_window * n_dims)
    ### because each window is trerated as a single data-point
    reshaped_data = np.reshape(input_data, (n_particles * n_windows, -1))

    ### onion_multi() returns the list of states and the label for each
    ### signal window
    state_list, labels = onion_multi(reshaped_data, bins=bins)

    ### These functions are examples of how to visualize the results
    plot_output_multi("Fig1.png", input_data, state_list, labels, tau_window)
    plot_one_trj_multi("Fig2.png", 0, tau_window, input_data, labels)
    plot_medoids_multi("Fig3.png", tau_window, input_data, labels)
    plot_state_populations("Fig4.png", n_windows, labels)
    plot_sankey("Fig5.png", labels, n_windows, [100, 200, 300, 400])

    ### CLUSTERING THE WHOLE RANGE OF TIME RESOLUTIONS ###
    tau_window_list = np.geomspace(3, 10000, 20, dtype=int)

    tra = np.zeros((len(tau_window_list), 3))  # List of number of states and
    # ENV0 population for each tau_window
    pop_list = []  # List of the states' population for each tau_window

    for i, tau_window in enumerate(tau_window_list):
        n_windows = int(n_frames / tau_window)
        excess_frames = n_frames - n_windows * tau_window

        if excess_frames > 0:
            reshaped_data = np.reshape(
                input_data[:, :, :-excess_frames],
                (n_particles * n_windows, -1),
            )
        else:
            reshaped_data = np.reshape(
                input_data, (n_particles * n_windows, -1)
            )

        state_list, labels = onion_multi(reshaped_data, bins=bins)

        list_pop = [state.perc for state in state_list]
        list_pop.insert(0, 1 - np.sum(np.array(list_pop)))

        tra[i][0] = tau_window
        tra[i][1] = len(state_list)
        tra[i][2] = list_pop[0]
        pop_list.append(list_pop)

    ### These functions are examples of how to visualize the results
    plot_time_res_analysis("Fig6.png", tra)
    plot_pop_fractions("Fig7.png", pop_list)

    plt.show()


if __name__ == "__main__":
    main()
