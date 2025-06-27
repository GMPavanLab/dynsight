from pathlib import Path

import MDAnalysis
import numpy as np

from dynsight.trajectory import Trj


# Define the actual test
def test_lens_signals() -> None:
    """Test the consistency of LENS calculations with a control calculation.

    * Original author: Martina Crippa

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
    check_file = np.load(original_dir / "../systems/LENS.npz")

    # Define r_cuts
    lens_cutoffs = [2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5]

    # Create universe for lens calculation
    universe = MDAnalysis.Universe(input_file, dt=1)
    example_trj = Trj(universe)

    # Run LENS (and nn) calculation for different r_cuts
    for i, r_cut in enumerate(lens_cutoffs):
        neigcounts, test_lens = example_trj.get_lens(r_cut=r_cut)
        test_lens_ds = np.array(
            [np.concatenate(([0.0], tmp)) for tmp in test_lens.dataset]
        )  # the inner LENS function has always 0.0 as first frame

        _, test_nn = example_trj.get_coord_number(
            r_cut=r_cut, neigcounts=neigcounts
        )
        test_array = [test_lens_ds, test_nn.dataset]

        check_lens_nn = check_file[f"LENS_{i}"]

        # Check if control and test array are equal
        assert np.allclose(check_lens_nn, test_array), (
            f"LENS analyses provided different values "
            f"compared to the control system "
            f"for r_cut: {r_cut}."
        )
