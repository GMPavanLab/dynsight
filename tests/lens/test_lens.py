"""Test the consistency of LENS calculations with a control calculation.

* Original author: Martina Crippa

This test verifies that the LENS calculation (LENS and nn) yields the same
values as a control calculation at different r_cut.

Control file path:
    - tests/systems/2_particles.xyz

Dynsight function tested:
    - dynsight.lens.list_neighbours_along_trajectory()
    - dynsight.lens.compute_lens_over_trj()

r_cuts checked:
    - [2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5]
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from dynsight.trajectory import Trj

LENS_CUTOFF = [2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5]

# ---------------- Fixtures ----------------


@pytest.fixture(scope="module")
def here() -> Path:
    return Path(__file__).parent


@pytest.fixture(scope="module")
def file_paths(here: Path) -> dict[str, Path]:
    return {
        "xyz": here / "../systems/2_particles.xyz",
        "check_file": here / "../systems/LENS.npz",
    }


@pytest.fixture(scope="module")
def trajectory(file_paths: dict[str, Path]) -> Trj:
    return Trj.init_from_xyz(file_paths["xyz"], dt=1)


# ---------------- Tests ----------------
def test_lens(
    trajectory: Trj,
    file_paths: dict[str, Path],
) -> None:
    """Test the consistency of LENS calculations."""
    check_file = np.load(file_paths["check_file"])

    # Run LENS (and NN) calculation for different r_cuts
    for i, r_cut in enumerate(LENS_CUTOFF):
        reference_array = check_file[f"LENS_{i}"]

        test_lens = trajectory.get_lens(r_cut=r_cut, respect_pbc=False)
        test_lens_ds = np.array(
            [np.concatenate(([0.0], tmp)) for tmp in test_lens.dataset]
        )  # the original LENS function gave always 0.0 as first frame

        assert np.allclose(reference_array[0], test_lens_ds), (
            "LENS analyses provided different values "
            f"compared to the control system for r_cut: {r_cut}."
        )


def test_nn(
    trajectory: Trj,
    file_paths: dict[str, Path],
) -> None:
    """Test the consistency of NN calculations."""
    check_file = np.load(file_paths["check_file"])

    # Run LENS (and NN) calculation for different r_cuts
    for i, r_cut in enumerate(LENS_CUTOFF):
        reference_array = check_file[f"LENS_{i}"]

        _, test_nn = trajectory.get_coord_number(
            r_cut=r_cut,
            respect_pbc=False,
        )
        test_nn_ds = test_nn.dataset

        assert np.allclose(reference_array[1], test_nn_ds), (
            "NN analyses provided different values "
            f"compared to the control system for r_cut: {r_cut}."
        )
