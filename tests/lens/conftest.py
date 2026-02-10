import pytest

from tests.lens.case_data import LENSCaseData


@pytest.fixture(
    scope="session",
    params=(
        # Case 0: default case
        lambda name: LENSCaseData(
            expected_lens="c0_lens_rc3_d1_all_all_1.npy",
            r_cut=3,
            delay=1,
            centers="all",
            selection="all",
            n_jobs=1,
            name=name,
        ),
        # Case 1: changing cutoff
        lambda name: LENSCaseData(
            expected_lens="c1_lens_rc4_d1_all_all_1.npy",
            r_cut=4,
            delay=1,
            centers="all",
            selection="all",
            n_jobs=1,
            name=name,
        ),
        # Case 2: single center
        lambda name: LENSCaseData(
            expected_lens="c2_lens_rc4_d1_1_all_1.npy",
            r_cut=4,
            delay=1,
            centers="id 1",
            selection="all",
            n_jobs=1,
            name=name,
        ),
        # Case 3: single selection
        lambda name: LENSCaseData(
            expected_lens="c3_lens_rc4_d1_all_1_1.npy",
            r_cut=4,
            delay=1,
            centers="all",
            selection="id 1",
            n_jobs=1,
            name=name,
        ),
        # Case 4: parallel
        lambda name: LENSCaseData(
            expected_lens="c4_lens_rc4_d1_all_all_2.npy",
            r_cut=4,
            delay=1,
            centers="all",
            selection="all",
            n_jobs=2,
            name=name,
        ),
        # Case 5: changing delay
        lambda name: LENSCaseData(
            expected_lens="c5_lens_rc4_d2_all_all_1.npy",
            r_cut=4,
            delay=2,
            centers="all",
            selection="all",
            n_jobs=1,
            name=name,
        ),
    ),
)
def case_data(request: pytest.FixtureRequest) -> LENSCaseData:
    return request.param(
        f"{request.fixturename}{request.param_index}",  # type: ignore [attr-defined]
    )
