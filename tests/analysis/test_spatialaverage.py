"""Pytest for dynsight.analysis.stapialaverage."""

from pathlib import Path

import numpy as np

from dynsight.trajectory import Insight, Trj


def test_spatialaverage() -> None:
    original_dir = Path(__file__).resolve().parent
    topology_file = original_dir / "../systems/coex/test_coex.gro"
    trajectory_file = original_dir / "../systems/coex/test_coex.xtc"
    expected_results = original_dir / "../analysis/spavg/test_spavg.npy"

    example_trj = Trj.init_from_xtc(trajectory_file, topology_file)
    descriptor = example_trj.get_coordinates("type O")[:, :, 0].T
    example_data = Insight(descriptor.astype(np.float64))

    aver_data = example_data.spatial_average(
        example_trj,
        r_cut=5.0,
        selection="type O",
        num_processes=1,
    )

    # Load expected results and compare
    expected_arr = np.load(expected_results)
    assert np.allclose(aver_data.dataset, expected_arr)
