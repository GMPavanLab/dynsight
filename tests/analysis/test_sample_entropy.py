"""Pytest for dynsight.analysis.compute_sample_entropy."""

import numpy as np
import pytest

import dynsight


# Define the actual test
def test_output_files() -> None:
    rng = np.random.default_rng(12345)

    random_data = rng.random(100)
    r_fact = float(0.5 * np.std(random_data))

    # Test the case where time-series are too short
    with pytest.raises(ValueError, match="Time-series too short"):
        _ = dynsight.analysis.sample_entropy(
            random_data,
            r_factor=r_fact,
            m_par=105,
        )

    # Test the case where distance threshold is too small
    with pytest.raises(RuntimeError, match="No matching sequences found."):
        _ = dynsight.analysis.sample_entropy(
            random_data,
            r_factor=0.0,
        )

    # Test the use of the function computing entropy
    data_sample_entropy = dynsight.analysis.sample_entropy(
        random_data,
        r_factor=r_fact,
    )

    expected_entropy = 1.3062516534463542
    if isinstance(data_sample_entropy, float):
        assert np.isclose(data_sample_entropy, expected_entropy)
