"""Pytest for dynsight.lens.compute_lens."""

from pathlib import Path

import MDAnalysis
import numpy as np
import pytest

from dynsight.trajectory import Trj
from dynsight.utilities import save_xyz_from_ndarray

from .case_data import LENSCaseData

# ---------------- Fixturesas --------------


@pytest.fixture
def trj_2d(tmp_path: Path) -> Trj:
    """Return a Trj for a bidimensional system."""
    rng = np.random.default_rng(42)
    coords = rng.random((100, 100, 3))
    coords[..., 2] = 0.0  # system is 2D
    traj_path = tmp_path / "random_2d.xyz"
    save_xyz_from_ndarray(traj_path, coords)

    trj = Trj.init_from_xyz(traj_path, dt=1.0)
    for ts in trj.universe.trajectory:  # Add box with thickness along z
        coords = trj.universe.atoms.positions
        mins = coords.min(axis=0)
        maxs = coords.max(axis=0)
        lengths = maxs - mins  # Lx, Ly, Lz
        lengths[2] = 0.5
        ts.dimensions = np.concatenate([lengths, np.array([90, 90, 90])])
    return trj


# ---------------- Tests ----------------


def test_lens(case_data: LENSCaseData) -> None:
    original_dir = Path(__file__).resolve().parent
    topology_file = original_dir / "../systems/balls_7_nvt.gro"
    trajectory_file = original_dir / "../systems/balls_7_nvt.xtc"
    expected_lens = original_dir / "test_lens" / case_data.expected_lens
    universe = MDAnalysis.Universe(topology_file, trajectory_file)

    example_trj = Trj(universe)

    test_lens = example_trj.get_lens(
        r_cut=case_data.r_cut,
        delay=case_data.delay,
        centers=case_data.centers,
        selection=case_data.selection,
        n_jobs=case_data.n_jobs,
    )

    if not expected_lens.exists():
        np.save(expected_lens, test_lens.dataset)
        pytest.fail(
            "LENS test files were not present. They have been created."
        )
    exp_lens = np.load(expected_lens)
    assert np.allclose(exp_lens, test_lens.dataset, atol=1e-6)


def test_lens_2d(trj_2d: Trj) -> None:
    """Test LENS and number of neighbors on a 2D system."""
    original_dir = Path(__file__).resolve().parent

    test_lens_pbc = trj_2d.get_lens(r_cut=0.1, respect_pbc=True)
    _ = trj_2d.get_coord_number(r_cut=0.1, respect_pbc=True)
    pbc_path = original_dir / "../systems/lens_2d_pbc.npy"
    if not pbc_path.exists():
        np.save(pbc_path, test_lens_pbc.dataset)
        pytest.fail(
            "LENS test files were not present. They have been created."
        )
    exp_lens = np.load(pbc_path)
    assert np.allclose(exp_lens, test_lens_pbc.dataset)

    test_lens_fbc = trj_2d.get_lens(r_cut=0.1, respect_pbc=False)
    _ = trj_2d.get_coord_number(r_cut=0.1, respect_pbc=False)
    fbc_path = original_dir / "../systems/lens_2d_fbc.npy"
    if not fbc_path.exists():
        np.save(fbc_path, test_lens_fbc.dataset)
        pytest.fail(
            "LENS test files were not present. They have been created."
        )
    exp_lens = np.load(fbc_path)
    assert np.allclose(exp_lens, test_lens_fbc.dataset)
