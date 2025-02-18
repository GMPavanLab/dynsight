"""Pytest for dynsight.analysis.compute_rdf."""

from pathlib import Path

import MDAnalysis
import numpy as np
import pytest

from dynsight.analysis import compute_rdf
from tests.analysis.rdf.case_data import RDFCaseData


def test_compute_rdf(case_data: RDFCaseData) -> None:
    original_dir = Path(__file__).resolve().parent
    topology_file = (
        original_dir / "../../systems/coex" / case_data.topology_filename
    )
    trajectory_file = (
        original_dir / "../../systems/coex" / case_data.trajectory_filename
    )
    expected_bins = original_dir / "test_rdf" / case_data.expected_bins
    expected_rdf = original_dir / "test_rdf" / case_data.expected_rdf

    u = MDAnalysis.Universe(topology_file, trajectory_file)

    selection = "type O"

    test_bins, test_rdf = compute_rdf(
        universe=u,
        s1=selection,
        s2=selection,
        distances_range=[0.0, 5.0],
        norm=case_data.norm,
    )

    if not expected_bins.exists() or not expected_rdf.exists():
        np.save(expected_bins, test_bins)
        np.save(expected_rdf, test_rdf)
        pytest.fail("RDF test files were not present. They have been created.")

    exp_bins = np.load(expected_bins)
    exp_rdf = np.load(expected_rdf)
    assert np.allclose(exp_rdf, test_rdf)
    assert np.allclose(exp_bins, test_bins)
