"""Tests for dynsight.analysis.time_correlations."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pytest

if TYPE_CHECKING:
    from numpy.typing import NDArray

import dynsight

# ---------------- Fixtures ----------------


@pytest.fixture
def rng() -> np.random.Generator:
    """Seeded RNG for reproducibility."""
    return np.random.default_rng(12345)


@pytest.fixture
def walk(rng: np.random.Generator) -> NDArray[np.float64]:
    """Generate a simple random walk (5 particles, 100 frames)."""
    n_part, n_frames = 5, 100
    walk = np.zeros((n_part, n_frames), dtype=np.float64)
    x_pos = 0.0
    for i in range(n_part):
        for j in range(n_frames):
            x_pos += rng.random()
            walk[i, j] = x_pos
    return walk


@pytest.fixture
def ref_data() -> dict[str, NDArray[np.float64]]:
    """Load expected correlation data."""
    base = Path("tests/analysis/tcorr")
    return {
        "auto_corr": np.load(base / "t_corr.npy"),
        "auto_std": np.load(base / "std_dev.npy"),
        "cross_corr": np.load(base / "c_corr.npy"),
        "cross_std": np.load(base / "ctd_dev.npy"),
    }


# ---------------- Tests ----------------


def test_auto_corr(
    walk: NDArray[np.float64], ref_data: dict[str, NDArray[np.float64]]
) -> None:
    """Check self-correlation against reference."""
    insight = dynsight.trajectory.Insight(walk)
    corr, std = insight.get_time_correlation()
    assert np.allclose(corr, ref_data["auto_corr"])
    assert np.allclose(std, ref_data["auto_std"])


def test_cross_corr(
    walk: NDArray[np.float64], ref_data: dict[str, NDArray[np.float64]]
) -> None:
    """Check cross-correlation against reference."""
    corr, std = dynsight.analysis.cross_time_correlation(walk)
    assert np.allclose(corr, ref_data["cross_corr"])
    assert np.allclose(std, ref_data["cross_std"])
