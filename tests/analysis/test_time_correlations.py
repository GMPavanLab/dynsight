"""Pytest for dynsight.analysis.time_correlations."""

import numpy as np

import dynsight


# Define the actual test
def test_output_files() -> None:
    rng = np.random.default_rng(12345)

    # Generate random walks
    n_part = 5
    n_frames = 100
    random_walk = np.zeros((n_part, n_frames))
    x_pos = 0.0
    for i in range(n_part):
        for j in range(n_frames):
            displ = rng.random(1)
            x_pos += displ[0]
            random_walk[i][j] = x_pos

    # Test the self-correlation function
    walk_insight = dynsight.trajectory.Insight(random_walk)
    t_corr, std_dev = walk_insight.get_time_correlation()

    exp_t_corr = np.load("tests/analysis/tcorr/t_corr.npy")
    exp_std_dev = np.load("tests/analysis/tcorr/std_dev.npy")

    assert np.allclose(t_corr, exp_t_corr)
    assert np.allclose(std_dev, exp_std_dev)

    # Test the cross-correlation function
    t_corr, std_dev = dynsight.analysis.cross_time_correlation(random_walk)

    exp_t_corr = np.load("tests/analysis/tcorr/c_corr.npy")
    exp_std_dev = np.load("tests/analysis/tcorr/ctd_dev.npy")

    assert np.allclose(t_corr, exp_t_corr)
    assert np.allclose(std_dev, exp_std_dev)
