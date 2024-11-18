"""Example script for running dynsight.onion.onion_uni."""

import matplotlib.pyplot as plt
import numpy as np

import dynsight


def main() -> None:
    """Run the example.

    Use `git clone git@github.com:matteobecchi/onion_example_files.git`
    to download example datasets.
    """
    path_to_input_data = "onion_example_files/data/univariate_time-series.npy"

    ### Load the input data - it's an array of shape (n_particles, n_frames)
    input_data = np.load(path_to_input_data)[:, 1:]
    n_frames = input_data.shape[1]

    ### CLUSTERING WITH A SINGLE TIME RESOLUTION ###
    ### Chose the time resolution --> the length of the windows in which the
    ### time-series will be divided
    tau_window = 5
    n_windows = int(n_frames / tau_window)  # Number of windows

    ### The input array needs to be (n_parrticles * n_windows, tau_window)
    ### because each window is trerated as a single data-point
    reshaped_data = dynsight.onion.helpers.reshape_from_nt(
        input_data, tau_window
    )

    ### onion_uni() returns the list of states and the label for each
    ### signal window
    state_list, labels = dynsight.onion.onion_uni(reshaped_data)

    ### These functions are examples of how to visualize the results
    dynsight.onion.plot.plot_output_uni(
        "Fig1.png", reshaped_data, n_windows, state_list
    )
    dynsight.onion.plot.plot_one_trj_uni(
        "Fig2.png", 1234, reshaped_data, labels, n_windows
    )
    dynsight.onion.plot.plot_medoids_uni("Fig3.png", reshaped_data, labels)
    dynsight.onion.plot.plot_state_populations("Fig4.png", n_windows, labels)
    dynsight.onion.plot.plot_sankey(
        "Fig5.png", labels, n_windows, [10, 20, 30, 40]
    )

    ### CLUSTERING THE WHOLE RANGE OF TIME RESOLUTIONS ###
    tau_windows = np.unique(np.geomspace(2, 499, num=20, dtype=int))

    tra = np.zeros((len(tau_windows), 3))  # List of number of states and
    # ENV0 population for each tau_window
    list_of_pop = []  # List of the states' population for each tau_window

    for i, tau_window in enumerate(tau_windows):
        reshaped_data = dynsight.onion.helpers.reshape_from_nt(
            input_data, tau_window
        )

        state_list, labels = dynsight.onion.onion_uni(reshaped_data)

        pop_list = [state.perc for state in state_list]
        pop_list.insert(0, 1 - np.sum(np.array(pop_list)))
        list_of_pop.append(pop_list)

        tra[i][0] = tau_window
        tra[i][1] = len(state_list)
        tra[i][2] = pop_list[0]

    ### These functions are examples of how to visualize the results
    dynsight.onion.plot.plot_time_res_analysis("Fig6.png", tra)
    dynsight.onion.plot.plot_pop_fractions("Fig7.png", list_of_pop, tra)

    plt.show()


if __name__ == "__main__":
    main()
