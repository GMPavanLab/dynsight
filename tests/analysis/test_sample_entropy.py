"""Pytest for dynsight.analysis.compute_sample_entropy."""

import numpy as np
import pytest
from numpy.typing import NDArray

import dynsight

# ----------------------- Fixtures -----------------------


@pytest.fixture
def random_data() -> NDArray[np.float64]:
    """Provides a reproducible random time series."""
    rng = np.random.default_rng(12345)
    return rng.random(100)


@pytest.fixture
def r_fact(random_data: NDArray[np.float64]) -> float:
    """Computes the r_factor as half the standard deviation."""
    return float(0.5 * np.std(random_data))


def test_too_short(
    random_data: NDArray[np.float64],
    r_fact: float,
) -> None:
    """Test that short time-series raise a ValueError."""
    with pytest.raises(ValueError, match="Time-series too short"):
        dynsight.analysis.sample_entropy(
            random_data,
            r_factor=r_fact,
            m_par=105,
        )


# ----------------------- Tests -----------------------


def test_too_small_rfact(random_data: NDArray[np.float64]) -> None:
    """Test that a too small r_factor raises a RuntimeError."""
    with pytest.raises(RuntimeError, match="No matching sequences found."):
        dynsight.analysis.sample_entropy(random_data, r_factor=0.0)


def test_sample_entropy(
    random_data: NDArray[np.float64], r_fact: float
) -> None:
    """Test that the computed sample entropy matches the expected value."""
    result = dynsight.analysis.sample_entropy(random_data, r_factor=r_fact)
    expected = 1.3062516534463542
    assert np.isclose(result, expected)
