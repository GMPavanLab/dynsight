import pytest

from tests.data_processing.cluster.case_data import CleanPopCaseData


@pytest.fixture(
    scope="session",
    params=(
        # Case 0: Cleaning 5%
        lambda name: CleanPopCaseData(
            expected_clean_pop="c0_clean_pop_th5_ass99_exNone.npy",
            threshold=0.05,
            assigned_env=99,
            excluded_env=None,
            name=name,
        ),
        # Case 1: Cleaning 15%
        lambda name: CleanPopCaseData(
            expected_clean_pop="c1_clean_pop_th15_ass99_exNone.npy",
            threshold=0.15,
            assigned_env=99,
            excluded_env=None,
            name=name,
        ),
        # Case 2: Cleaning 25%
        lambda name: CleanPopCaseData(
            expected_clean_pop="c2_clean_pop_th25_ass99_exNone.npy",
            threshold=0.25,
            assigned_env=99,
            excluded_env=None,
            name=name,
        ),
        # Case 3: Cleaning 25%, excluding 4
        lambda name: CleanPopCaseData(
            expected_clean_pop="c3_clean_pop_th25_ass99_ex4.npy",
            threshold=0.25,
            assigned_env=99,
            excluded_env=4,
            name=name,
        ),
        # Case 4: Cleaning 25%, excluding 3,4
        lambda name: CleanPopCaseData(
            expected_clean_pop="c4_clean_pop_th25_ass99_ex3-4.npy",
            threshold=0.25,
            assigned_env=99,
            excluded_env=[3, 4],
            name=name,
        ),
        # Case 5: Cleaning 25%, excluding 3,7
        lambda name: CleanPopCaseData(
            expected_clean_pop="c5_clean_pop_th25_ass99_ex3-7.npy",
            threshold=0.25,
            assigned_env=99,
            excluded_env=[3, 7],
            name=name,
        ),
        # Case 6: Cleaning 25%, excluding 3,4
        lambda name: CleanPopCaseData(
            expected_clean_pop="c6_clean_pop_th25_ass1_exNone.npy",
            threshold=0.25,
            assigned_env=1,
            excluded_env=None,
            name=name,
        ),
    ),
)
def case_data(request: pytest.FixtureRequest) -> CleanPopCaseData:
    return request.param(
        f"{request.fixturename}{request.param_index}",  # type: ignore [attr-defined]
    )
