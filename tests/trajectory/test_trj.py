"""Pytest for dynsight.trajectory methods."""

from __future__ import annotations

from pathlib import Path

import MDAnalysis
import numpy as np
import pytest

from dynsight.logs import logger
from dynsight.trajectory import (
    ClusterInsight,
    Insight,
    OnionInsight,
    OnionSmoothInsight,
    Trj,
)

# ---------------- Fixtures ----------------


@pytest.fixture(scope="module")
def here() -> Path:
    return Path(__file__).parent


@pytest.fixture(scope="module")
def file_paths(here: Path) -> dict[str, Path]:
    return {
        "xyz": here / "../systems/2_particles.xyz",
        "gro": here / "../systems/balls_7_nvt.gro",
        "xtc": here / "../systems/balls_7_nvt.xtc",
        "files_dir": here / "files",
    }


@pytest.fixture(scope="module")
def universe(file_paths: dict[str, Path]) -> MDAnalysis.Universe:
    return MDAnalysis.Universe(file_paths["xyz"], dt=1)


# ---------------- Tests ----------------


def test_trj_inits(
    universe: MDAnalysis.Universe, file_paths: dict[str, Path]
) -> None:
    """Test initialization methods for Trj class."""
    n_frames_xyz = 21
    n_frames_xtc = 201

    trj_1 = Trj(universe)
    assert len(trj_1.universe.trajectory) == n_frames_xyz

    trj_2 = Trj.init_from_universe(universe)
    logger.get()
    assert len(trj_2.universe.trajectory) == n_frames_xyz

    trj_3 = Trj.init_from_xyz(file_paths["xyz"], dt=1)
    assert len(trj_3.universe.trajectory) == n_frames_xyz

    trj_4 = Trj.init_from_xtc(file_paths["xtc"], file_paths["gro"])
    assert len(trj_4.universe.trajectory) == n_frames_xtc


def test_insight(
    tmp_path: Path, file_paths: dict[str, Path], universe: MDAnalysis.Universe
) -> None:
    """Test Insight methods."""
    trj = Trj(universe)
    _, insight = trj.get_lens(10.0)

    # Insight dump/load
    json_file = tmp_path / "insight.json"
    insight.dump_to_json(json_file)
    loaded_insight = Insight.load_from_json(
        file_paths["files_dir"] / "ins_1_test.json"
    )
    assert loaded_insight is not None

    # ClusterInsight dump/load
    fake_labels = np.zeros((5, 5), dtype=int)
    cluster_ins = ClusterInsight(fake_labels)
    cluster_json = tmp_path / "cluster.json"
    cluster_ins.dump_to_json(cluster_json)
    loaded_cluster = ClusterInsight.load_from_json(
        file_paths["files_dir"] / "cl_ins_test.json"
    )
    assert loaded_cluster is not None

    # OnionInsight dump/load
    onion_ins = insight.get_onion(delta_t=5)
    onion_json = tmp_path / "onion.json"
    onion_ins.dump_to_json(onion_json)
    loaded_onion = OnionInsight.load_from_json(
        file_paths["files_dir"] / "on_ins_test.json"
    )
    assert loaded_onion is not None

    # OnionSmoothInsight dump/load
    onion_smooth_ins = insight.get_onion_smooth(delta_t=5)
    onion_smooth_json = tmp_path / "onion_smooth.json"
    onion_smooth_ins.dump_to_json(onion_smooth_json)
    loaded_onion_smooth = OnionSmoothInsight.load_from_json(onion_smooth_json)
    assert loaded_onion_smooth is not None


def test_onion_analysis(universe: MDAnalysis.Universe) -> None:
    """Test the onion clustering complete analysis tool."""
    trj = Trj(universe)
    _, insight = trj.get_lens(10.0)
    result = insight.get_onion_analysis()
    assert result is not None


def test_insight_load_errors(file_paths: dict[str, Path]) -> None:
    """Test insight load errors."""
    with pytest.raises(
        ValueError, match="'dataset' key not found in JSON file."
    ):
        _ = Insight.load_from_json(file_paths["files_dir"] / "empty.json")

    with pytest.raises(
        ValueError, match="'labels' key not found in JSON file."
    ):
        _ = ClusterInsight.load_from_json(
            file_paths["files_dir"] / "ins_1_test.json"
        )

    with pytest.raises(
        ValueError, match="'state_list' key not found in JSON file."
    ):
        _ = OnionInsight.load_from_json(
            file_paths["files_dir"] / "cl_ins_test.json"
        )
