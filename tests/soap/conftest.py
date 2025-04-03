import pytest

from tests.soap.case_data import SOAPCaseData


@pytest.fixture(
    scope="session",
    params=(
        # Case 0: Default case
        lambda name: SOAPCaseData(
            expected_soap="c0_soap_rc8_l4_n4_pbc0_all.npy",
            r_c=8,
            l_max=4,
            n_max=4,
            respect_pbc=True,
            centers="all",
            name=name,
        ),
        # Case 1: changing the cutoff radius
        lambda name: SOAPCaseData(
            expected_soap="c1_soap_rc8_l6_n6_pbc1_noc5.npy",
            r_c=4,
            l_max=4,
            n_max=4,
            respect_pbc=True,
            centers="all",
            name=name,
        ),
        # Case 2: changing n and l parameters
        lambda name: SOAPCaseData(
            expected_soap="c2_soap_rc8_l6_n6_pbc1_all.npy",
            r_c=8,
            l_max=6,
            n_max=6,
            respect_pbc=True,
            centers="all",
            name=name,
        ),
        # Case 3: Excluding a center
        lambda name: SOAPCaseData(
            expected_soap="c3_soap_rc8_l4_n4_pbc1_noc5.npy",
            r_c=8,
            l_max=4,
            n_max=4,
            respect_pbc=True,
            centers="not name C5",
            name=name,
        ),
        # Case 4: Selecting specific centers
        lambda name: SOAPCaseData(
            expected_soap="c4_soap_rc8_l4_n4_pbc1_c3c6.npy",
            r_c=8,
            l_max=4,
            n_max=4,
            respect_pbc=True,
            centers="name C3 or name C6",
            name=name,
        ),
        # Case 5: Disabling PBC
        lambda name: SOAPCaseData(
            expected_soap="c5_soap_rc8_l4_n4_pbc1_all.npy",
            r_c=8,
            l_max=4,
            n_max=4,
            respect_pbc=False,
            centers="all",
            name=name,
        ),
    ),
)
def case_data(request: pytest.FixtureRequest) -> SOAPCaseData:
    return request.param(
        f"{request.fixturename}{request.param_index}",  # type: ignore [attr-defined]
    )
