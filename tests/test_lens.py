from pathlib import Path

import dynsight
import h5py
import numpy as np

"""
Test description:tests if a LENS calculation yields the same
                    values (of LENS and nn) as a control calculation
                    at different r_cut.

Control file path: tests/systems/2_particles.hdf5

Dynsyght function tested: dynsight.lens.list_neighbours_along_trajectory()
                          dynsight.lens.neighbour_change_in_time()

r_cuts checked: 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5
"""


def test_lens_signals() -> None:
    # i/o files
    input_file = "tests/systems/2_particles.hdf5"
    output_file = "tests/systems/2_particles_test.hdf5"

    # trajectory name
    traj_name = "2_particles"
    trajectory = slice(0, 20)

    # r_cuts
    lens_cutoffs = [2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5]

    # create universe for lens calculation
    with h5py.File(input_file, "r") as file:
        tgroup = file["Trajectories"][traj_name]
        universe = dynsight.hdf5er.create_universe_from_slice(
            tgroup, trajectory
        )

    # LENS (and nn) calculation for different r_cuts
    for i in range(len(lens_cutoffs)):
        neig_counts = dynsight.lens.list_neighbours_along_trajectory(
            universe, cutoff=lens_cutoffs[i]
        )
        lens, nn, *_ = dynsight.lens.neighbour_change_in_time(neig_counts)

        # test array
        test_lens_nn = np.array([lens, nn])
        # check array
        with h5py.File(input_file, "r") as in_file:
            check_lens_nn = np.array(in_file[f"LENS_{i}"][f"LENS_{i}"])
        # output file
        with h5py.File(output_file, "w") as out_file:
            out_file.create_group(f"LENS_test_{i}")
            out_file[f"LENS_test_{i}"].create_dataset(
                f"LENS_test_{i}", data=test_lens_nn
            )

        # check if control and test array are equal
        assert np.array_equal(check_lens_nn, test_lens_nn), (
            f"LENS analyses provided different values "
            f"compared to the control system "
            f"for r_cut: {lens_cutoffs[i]} (results: {output_file})."
        )
        # if test passed remove test_lens_nn array from test folder
        Path(output_file).unlink()
