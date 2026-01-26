import pytest

from tests.neighbors.case_data import NNCaseData


@pytest.fixture(
    scope="session",
    params=(
        # Case 0: Default case
        lambda name: NNCaseData(
            expected_nn="c0_nn_rc3_all_all_1.npy",
            r_cut=3,
            centers="all",
            selection="all",
            n_jobs=1,
            name=name,
        ),
        # Case 1: changing cutoff
        lambda name: NNCaseData(
            expected_nn="c1_nn_rc4_all_all_1.npy",
            r_cut=4,
            centers="all",
            selection="all",
            n_jobs=1,
            name=name,
        ),
        # Case 2: single center
        lambda name: NNCaseData(
            expected_nn="c2_nn_rc4_1_all_1.npy",
            r_cut=4,
            centers="id 1",
            selection="all",
            n_jobs=1,
            name=name,
        ),
        # Case 3: single selection
        lambda name: NNCaseData(
            expected_nn="c3_nn_rc4_all_1_1.npy",
            r_cut=4,
            centers="all",
            selection="id 1",
            n_jobs=1,
            name=name,
        ),
        # Case 4: parallel
        lambda name: NNCaseData(
            expected_nn="c4_nn_rc4_all_all_2.npy",
            r_cut=4,
            centers="all",
            selection="all",
            n_jobs=2,
            name=name,
        ),
    ),
)
def case_data(request: pytest.FixtureRequest) -> NNCaseData:
    return request.param(
        f"{request.fixturename}{request.param_index}",  # type: ignore [attr-defined]
    )
