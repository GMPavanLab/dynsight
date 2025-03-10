"""Example script for running dynsight.onion.onion_uni."""

import matplotlib.pyplot as plt
import numpy as np

from dynsight import onion


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
    n_particles = input_data.shape[0]
    n_frames = input_data.shape[1]

    """ STEP 1: CLUSTERING WITH A SINGLE TIME RESOLUTION
    Chose the time resolution --> the length of the windows in which the
    time-series will be divided. This is the minimum lifetime required for
    a state to be considered stable."""
    delta_t = 5

    ### The input array needs to be (n_particles * n_windows, delta_t)
    ### because each window is trerated as a single data-point
    reshaped_data = onion.helpers.reshape_from_nt(input_data, delta_t)

    ### onion_uni() returns the list of states and the label for each
    ### signal window
    state_list, labels = onion.onion_uni(reshaped_data)

    ### These functions are examples of how to visualize the results
    onion.plot.plot_output_uni(
        "Fig1.png", reshaped_data, n_particles, state_list
    )
    onion.plot.plot_one_trj_uni(
        "Fig2.png", 1234, reshaped_data, n_particles, labels
    )
    onion.plot.plot_medoids_uni("Fig3.png", reshaped_data, labels)
    onion.plot.plot_state_populations("Fig4.png", n_particles, delta_t, labels)
    onion.plot.plot_sankey("Fig5.png", labels, n_particles, [10, 20, 30, 40])

    """ STEP 2: CLUSTERING THE WHOLE RANGE OF TIME RESOLUTIONS
    This allows to select the optimal time resolution for the analysis,
    avoiding an a priori choice."""
    all_delta_t = np.unique(np.geomspace(2, n_frames, num=20, dtype=int))

    tra = np.zeros((len(all_delta_t), 3))  # List of number of states and
    # ENV0 population for each delta_t
    list_of_pop = []  # List of the states' population for each delta_t

    for i, delta_t in enumerate(all_delta_t):
        reshaped_data = onion.helpers.reshape_from_nt(input_data, delta_t)

        state_list, labels = onion.onion_uni(reshaped_data)

        pop_list = [state.perc for state in state_list]
        pop_list.insert(0, 1 - np.sum(np.array(pop_list)))  # Add ENV0 fraction
        list_of_pop.append(pop_list)

        tra[i][0] = delta_t
        tra[i][1] = len(state_list)
        tra[i][2] = pop_list[0]

    ### These functions are examples of how to visualize the results
    onion.plot.plot_time_res_analysis("Fig6.png", tra)
    onion.plot.plot_pop_fractions("Fig7.png", list_of_pop, tra)

    plt.show()


if __name__ == "__main__":
    main()
