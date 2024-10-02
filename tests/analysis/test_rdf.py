"""Pytest for dynsight.analysis.compute_rdf."""

from pathlib import Path

import MDAnalysis
import numpy as np

from dynsight.analysis import compute_rdf


def test_compute_rdf() -> None:
    original_dir = Path(__file__).resolve().parent
    topology_file = original_dir / "../systems/coex/test_coex.gro"
    trajectory_file = original_dir / "../systems/coex/test_coex.xtc"
    expected_bins = original_dir / "test_rdf/test_bins.npy"
    expected_rdf = original_dir / "test_rdf/test_rdf.npy"

    u = MDAnalysis.Universe(topology_file, trajectory_file)

    selection = "type O"

    test_bins, test_rdf = compute_rdf(
        universe=u,
        s1=selection,
        s2=selection,
        distances_range=[0.0, 5.0],
    )
    exp_bins = np.load(expected_bins)
    exp_rdf = np.load(expected_rdf)
    assert np.array_equal(exp_rdf, test_rdf)
    assert np.array_equal(exp_bins, test_bins)
