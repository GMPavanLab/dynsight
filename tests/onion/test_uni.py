"""Pytest for dynsight.onion.onion_uni."""

from pathlib import Path

import numpy as np

from dynsight.trajectory import Insight


# Define the actual test
def test_output_files() -> None:
    ### Create the input data ###
    rng = np.random.default_rng(12345)
    n_particles = 5
    n_steps = 1000
    random_walk = np.zeros((n_particles, n_steps))
    for i in range(n_particles):
        tmp = np.zeros(n_steps)
        for j in range(1, n_steps):
            d_x = rng.normal()
            x_new = tmp[j - 1] + d_x
            tmp[j] = x_new
        random_walk[i] = tmp
    data = Insight(random_walk)

    ### Perform onion clustering
    clustering = data.get_onion(delta_t=10)

    ### Define the paths to the expected output ###
    results_dir = Path("tests/onion/")
    expected_output_path = results_dir / "output_uni/labels.npy"

    ### Compare the contents of the expected and actual output ###
    expected_output = np.load(expected_output_path)
    assert np.allclose(expected_output, clustering.labels)
