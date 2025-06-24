"""Pytest for dynsight.onion onion clustering (uni- and multi-dimensional)."""

from pathlib import Path

import numpy as np
import pytest

from dynsight.trajectory import Insight

# ---------------- Fixtures ----------------


@pytest.fixture(scope="module")
def random_walk_uni() -> Insight:
    """Return a 1D random walk."""
    rng = np.random.default_rng(12345)
    n_particles = 5
    n_steps = 1000
    data = np.zeros((n_particles, n_steps), dtype=np.float64)
    for i in range(n_particles):
        for j in range(1, n_steps):
            data[i, j] = data[i, j - 1] + rng.normal()
    return Insight(data)


@pytest.fixture(scope="module")
def random_walk_multi() -> Insight:
    """Return a 2D random walk."""
    rng = np.random.default_rng(12345)
    n_particles = 5
    n_steps = 1000
    data = np.zeros((n_particles, n_steps, 2), dtype=np.float64)
    for i in range(n_particles):
        for j in range(1, n_steps):
            data[i, j, 0] = data[i, j - 1, 0] + rng.normal()
            data[i, j, 1] = data[i, j - 1, 1] + rng.normal()
    return Insight(data)


@pytest.fixture(scope="module")
def expected_labels_uni() -> Path:
    return Path("tests/onion/output_uni/labels.npy")


@pytest.fixture(scope="module")
def expected_labels_multi() -> Path:
    return Path("tests/onion/output_multi/labels.npy")


# ---------------- Tests ----------------


def test_onion_uni_clustering(
    random_walk_uni: Insight, expected_labels_uni: Path
) -> None:
    """Test onion clustering on uni-dimensional random walk data."""
    clustering = random_walk_uni.get_onion(delta_t=10)
    expected = np.load(expected_labels_uni)
    assert np.allclose(expected, clustering.labels)


def test_onion_multi_clustering(
    random_walk_multi: Insight, expected_labels_multi: Path
) -> None:
    """Test onion clustering on multi-dimensional random walk data."""
    clustering = random_walk_multi.get_onion(delta_t=10)
    expected = np.load(expected_labels_multi)
    assert np.allclose(expected, clustering.labels)
