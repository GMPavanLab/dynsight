import pytest

from tests.tsoap.case_data import TimeSOAPCaseData


@pytest.fixture(
    scope="session",
    params=(
        # Case 0: Default case
        lambda name: TimeSOAPCaseData(
            expected_tsoap="c0_tsoap_rc8_d1.npy",
            r_c=8,
            delay=1,
            name=name,
        ),
        # Case 1: changing the cutoff radius
        lambda name: TimeSOAPCaseData(
            expected_tsoap="c1_tsoap_rc4_d1.npy",
            r_c=4,
            delay=1,
            name=name,
        ),
        # Case 2: changing the delay
        lambda name: TimeSOAPCaseData(
            expected_tsoap="c2_tsoap_rc4_d4.npy",
            r_c=8,
            delay=4,
            name=name,
        ),
    ),
)
def case_data(request: pytest.FixtureRequest) -> TimeSOAPCaseData:
    return request.param(
        f"{request.fixturename}{request.param_index}",  # type: ignore [attr-defined]
    )
