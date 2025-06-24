"""Tests for dynsight.analysis.spatialaverage."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pytest

if TYPE_CHECKING:
    from numpy.typing import NDArray

from dynsight.trajectory import Insight, Trj

# ---------------- Fixtures ----------------


@pytest.fixture
def base_dir() -> Path:
    """Base path for test files."""
    return Path(__file__).resolve().parent


@pytest.fixture
def files(base_dir: Path) -> dict[str, Path]:
    """Paths to topology, trajectory, and expected result."""
    return {
        "top": base_dir / "../systems/coex/test_coex.gro",
        "xtc": base_dir / "../systems/coex/test_coex.xtc",
        "ref": base_dir / "../analysis/spavg/test_spavg.npy",
    }


@pytest.fixture
def trj(files: dict[str, Path]) -> Trj:
    """Initialized trajectory."""
    return Trj.init_from_xtc(files["xtc"], files["top"])


@pytest.fixture
def insight(trj: Trj) -> Insight:
    """Insight object using coordinates of type O."""
    coords: NDArray[np.float64] = trj.get_coordinates("type O")[
        :, :, 0
    ].T.astype(np.float64)
    return Insight(coords)


# ---------------- Test ----------------


def test_spavg(trj: Trj, insight: Insight, files: dict[str, Path]) -> None:
    """Test spatial_average against saved reference data."""
    out = insight.spatial_average(
        trj,
        r_cut=5.0,
        selection="type O",
        num_processes=1,
    )

    expected: NDArray[np.float64] = np.load(files["ref"])
    assert np.allclose(out.dataset, expected)
