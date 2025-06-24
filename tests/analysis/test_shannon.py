"""Pytest for dynsight.analysis.compute_entropy_gain."""

import numpy as np
import pytest
from numpy.typing import NDArray

import dynsight

# ----------------------- Fixtures -----------------------


@pytest.fixture
def rng() -> np.random.Generator:
    """Shared RNG for reproducible results."""
    return np.random.default_rng(12345)


@pytest.fixture
def data(rng: np.random.Generator) -> NDArray[np.float64]:
    """Random (100x100) array."""
    return rng.random((100, 100))


@pytest.fixture
def data_2d(rng: np.random.Generator) -> NDArray[np.float64]:
    """Random (100x2) array."""
    return rng.random((100, 2))


@pytest.fixture
def labels(rng: np.random.Generator) -> NDArray[np.int64]:
    """Valid integer labels for 100 samples."""
    return rng.integers(0, 5, (100,), dtype=np.int64)


@pytest.fixture
def bad_labels() -> NDArray[np.int64]:
    """Mismatched label shape."""
    rng = np.random.default_rng(12345)
    return rng.integers(0, 5, (200, 50), dtype=np.int64)


# ----------------------- Tests -----------------------


def test_wrong_input(
    data_2d: NDArray[np.float64],
    labels: NDArray[np.int64],
) -> None:
    """Check wrong input raises Errors."""
    with pytest.raises(ValueError, match="data is empty"):
        dynsight.analysis.compute_shannon(np.array([]), (0.0, 1.0), n_bins=20)
    with pytest.raises(ValueError, match="data is empty"):
        dynsight.analysis.compute_shannon_multi(
            np.array([]), [(0.0, 1.0), (0.0, 1.0)], n_bins=[20, 20]
        )
    with pytest.raises(
        ValueError,
        match="Mismatch between data dimensions, data_ranges, and n_bins",
    ):
        dynsight.analysis.compute_shannon_multi(
            data_2d, [(0.0, 1.0)], n_bins=[20]
        )
    with pytest.raises(
        RuntimeError,
        match=rf"data \(\({data_2d.shape}\)\) and labels "
        rf"\(\({labels[:50].shape}\)\) "
        r"must have same shape\[0\]",
    ):
        dynsight.analysis.compute_entropy_gain_multi(
            data_2d, labels[:50], n_bins=[20, 20]
        )


def test_bad_shape(
    data: NDArray[np.float64], bad_labels: NDArray[np.int64]
) -> None:
    """Bad label shape should raise RuntimeError."""
    with pytest.raises(RuntimeError):
        dynsight.analysis.compute_entropy_gain(data, bad_labels, n_bins=20)


def test_gain(data: NDArray[np.float64], labels: NDArray[np.int64]) -> None:
    """Check entropy gain value."""
    _, gain, *_ = dynsight.analysis.compute_entropy_gain(
        data, labels, n_bins=20
    )
    ref = 0.0010842808402454819
    assert np.isclose(gain, ref)


def test_gain_multi(
    data_2d: NDArray[np.float64],
    labels: NDArray[np.int64],
) -> None:
    """Check entropy gain value."""
    _, gain, *_ = dynsight.analysis.compute_entropy_gain_multi(
        data_2d,
        labels,
        n_bins=[20, 20],
    )
    ref = 0.32537575591706214
    assert np.isclose(gain, ref)
