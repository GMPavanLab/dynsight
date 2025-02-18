from pathlib import Path

import MDAnalysis
import numpy as np

import dynsight


def test_lens_signals() -> None:
    """Test the consistency of LENS calculations with a control calculation.

    * Original author: Martina Crippa
    * Mantainer: Matteo Becchi

    This test verifies that the LENS calculation (LENS and nn) yields the same
    values as a control calculation at different r_cut.

    Control file path:
        - tests/systems/2_particles.xyz

    Dynsight function tested:
        - dynsight.lens.list_neighbours_along_trajectory()
        - dynsight.lens.neighbour_change_in_time()

    r_cuts checked:
        - [2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5]
    """
    # Define input and output files
    original_dir = Path(__file__).absolute().parent
    input_file = original_dir / "../systems/2_particles.xyz"
    output_file = original_dir / "../2_particles_test.hdf5"

    check_file = np.load(original_dir / "../systems/LENS.npz")

    # Define r_cuts
    lens_cutoffs = [2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5]

    # Create universe for lens calculation
    universe = MDAnalysis.Universe(input_file, dt=1)

    # Run LENS (and nn) calculation for different r_cuts
    for i in range(len(lens_cutoffs)):
        neig_counts = dynsight.lens.list_neighbours_along_trajectory(
            universe, cutoff=lens_cutoffs[i]
        )
        lens, nn, *_ = dynsight.lens.neighbour_change_in_time(neig_counts)

        # Define test array
        test_lens_nn = np.array([lens, nn])

        check_lens_nn = check_file[f"LENS_{i}"]

        # Check if control and test array are equal
        assert np.allclose(check_lens_nn, test_lens_nn), (
            f"LENS analyses provided different values "
            f"compared to the control system "
            f"for r_cut: {lens_cutoffs[i]} (results: {output_file})."
        )
