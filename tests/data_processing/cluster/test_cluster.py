"""Pytest for dynsight.lens.compute_lens."""

from pathlib import Path

import numpy as np
import pytest

from dynsight.data_processing import cleaning_cluster_population

from .case_data import CleanPopCaseData


def test_clean_pop_noexcl(case_data: CleanPopCaseData) -> None:
    original_dir = Path(__file__).resolve().parent
    expected_clean_pop = (
        original_dir / "test_cluster" / case_data.expected_clean_pop
    )

    labels = np.zeros((4, 10, 3), dtype=int)

    labels[:, :, 0] = np.array(
        [
            [0, 0, 0, 1, 1, 1, 2, 2, 2, 2],
            [0, 0, 0, 1, 1, 1, 2, 2, 2, 2],
            [0, 0, 0, 1, 1, 1, 2, 2, 2, 3],
            [0, 0, 0, 1, 1, 1, 2, 2, 2, 3],
        ]
    )

    labels[:, :, 1] = np.array(
        [
            [0, 0, 0, 0, 0, 1, 1, 1, 1, 4],
            [0, 0, 0, 0, 0, 1, 1, 1, 1, 4],
            [0, 0, 0, 0, 0, 1, 1, 1, 4, 4],
            [0, 0, 0, 0, 0, 1, 1, 1, 4, 4],
        ]
    )

    labels[:, :, 2] = np.array(
        [
            [0, 0, 9, 9, 0, 1, 1, 1, 1, 9],
            [0, 0, 0, 9, 0, 1, 1, 1, 1, 9],
            [0, 0, 0, 9, 0, 1, 1, 1, 1, 9],
            [0, 9, 0, 0, 0, 1, 1, 1, 1, 9],
        ]
    )

    test_clean_pop = cleaning_cluster_population(
        labels,
        threshold=case_data.threshold,
        assigned_env=case_data.assigned_env,
        excluded_env=case_data.excluded_env,
    )

    if not expected_clean_pop.exists():
        np.save(expected_clean_pop, test_clean_pop)
        pytest.fail(
            "Clean_pop test files were not present. They have been created."
        )
    exp_clean_pop = np.load(expected_clean_pop)
    assert np.array_equal(exp_clean_pop, test_clean_pop)
