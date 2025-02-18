import tempfile
from pathlib import Path

import MDAnalysis
import numpy as np

from dynsight.analysis import spatialaverage


def test_spatialaverage() -> None:
    original_dir = Path(__file__).resolve().parent
    topology_file = original_dir / "../systems/coex/test_coex.gro"
    trajectory_file = original_dir / "../systems/coex/test_coex.xtc"
    expected_results = original_dir / "spavg/test_spavg.npy"

    u = MDAnalysis.Universe(topology_file, trajectory_file)
    atoms = u.select_atoms("type O")

    descriptor = np.zeros((2048, 6))
    for ts in u.trajectory:
        descriptor[:, ts.frame] = atoms.positions[:, 0]

    # Create a temporary file for test_arr
    with tempfile.NamedTemporaryFile(suffix=".npy", delete=False) as temp_file:
        temp_file_path = Path(temp_file.name)
        np.save(temp_file_path, descriptor)

    # Load the temporary file and run spatialaverage
    test_arr = spatialaverage(
        universe=u,
        descriptor_array=descriptor,
        selection="type O",
        cutoff=5.0,
        num_processes=1,
    )

    # Clean up temporary file
    temp_file_path.unlink()

    # Load expected results and compare
    expected_arr = np.load(expected_results)
    assert np.allclose(test_arr, expected_arr)
