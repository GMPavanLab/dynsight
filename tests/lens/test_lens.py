from pathlib import Path

import h5py
import numpy as np

import dynsight


def test_lens_signals() -> None:
    """Test the consistency of LENS calculations with a control calculation.

    This test verifies that the LENS calculation (LENS and nn) yields the same
    values as a control calculation at different r_cut.

    Control file path:
        - tests/systems/2_particles.hdf5

    Dynsight function tested:
        - dynsight.lens.list_neighbours_along_trajectory()
        - dynsight.lens.neighbour_change_in_time()

    r_cuts checked:
        - [2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5]
    """
    # Define input and output files
    original_dir = Path(__file__).absolute().parent
    input_file = original_dir / "../systems/2_particles.hdf5"
    output_file = original_dir / "../2_particles_test.hdf5"

    # Define trajectory parameters
    traj_name = "2_particles"
    trajectory = slice(0, 20)

    # Define r_cuts
    lens_cutoffs = [2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5]

    # Create universe for lens calculation
    with h5py.File(input_file, "r") as file:
        tgroup = file["Trajectories"][traj_name]
        universe = dynsight.hdf5er.create_universe_from_slice(
            tgroup, trajectory
        )

    # Run LENS (and nn) calculation for different r_cuts
    for i in range(len(lens_cutoffs)):
        neig_counts = dynsight.lens.list_neighbours_along_trajectory(
            universe, cutoff=lens_cutoffs[i]
        )
        lens, nn, *_ = dynsight.lens.neighbour_change_in_time(neig_counts)

        # Define test array
        test_lens_nn = np.array([lens, nn])
        # Define check array
        with h5py.File(input_file, "r") as in_file:
            check_lens_nn = np.array(in_file[f"LENS_{i}"][f"LENS_{i}"])
        # Define output file
        with h5py.File(output_file, "w") as out_file:
            out_file.create_group(f"LENS_test_{i}")
            out_file[f"LENS_test_{i}"].create_dataset(
                f"LENS_test_{i}", data=test_lens_nn
            )

        # Check if control and test array are equal
        assert np.array_equal(check_lens_nn, test_lens_nn), (
            f"LENS analyses provided different values "
            f"compared to the control system "
            f"for r_cut: {lens_cutoffs[i]} (results: {output_file})."
        )
        # If test passed remove test_lens_nn array from test folder
        output_file.unlink()
