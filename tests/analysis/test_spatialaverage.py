"""Tests for dynsight.analysis.spatialaverage."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pytest

if TYPE_CHECKING:
    from numpy.typing import NDArray

from dynsight.analysis import spatialaverage
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


@pytest.fixture
def trj_2d(base_dir: Path) -> Trj:
    """A 2D trajectory, whose LENS is one frame shorter than itself."""
    return Trj.init_from_xyz(
        traj_file=base_dir / "../../docs/source/_static/ex_test_files"
        "/trajectory.xyz",
        dt=1.0,
    )


# ---------------- Test ----------------


def test_spavg(trj: Trj, insight: Insight, files: dict[str, Path]) -> None:
    """Test spatial_average against saved reference data."""
    out = insight.spatial_average(
        trj,
        r_cut=5.0,
        selection="type O",
        n_jobs=1,
    )

    expected: NDArray[np.float64] = np.load(files["ref"])
    assert np.allclose(out.dataset, expected)


def test_spatial_average_frame_mismatch_raises_clearly(trj_2d: Trj) -> None:
    """LENS is one frame shorter than its trajectory: say so, don't crash."""
    descriptor = np.zeros((trj_2d.n_atoms, trj_2d.n_frames - 1))

    with pytest.raises(ValueError, match="Descriptor covers"):
        spatialaverage(
            universe=trj_2d.universe,
            descriptor_array=descriptor,
            selection="all",
            r_cut=3.0,
        )
