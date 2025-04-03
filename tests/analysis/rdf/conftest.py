import pytest

from tests.analysis.rdf.case_data import RDFCaseData


@pytest.fixture(
    scope="session",
    params=(
        # Case 0: Default case
        lambda name: RDFCaseData(
            topology_filename="test_coex.gro",
            trajectory_filename="test_coex.xtc",
            expected_bins="test_bins_rdf.npy",
            expected_rdf="test_rdf_rdf.npy",
            norm="rdf",
            name=name,
        ),
        # Case 1: Density normalization
        lambda name: RDFCaseData(
            topology_filename="test_coex.gro",
            trajectory_filename="test_coex.xtc",
            expected_bins="test_bins_density.npy",
            expected_rdf="test_rdf_density.npy",
            norm="density",
            name=name,
        ),
        # Case 2: No normalization
        lambda name: RDFCaseData(
            topology_filename="test_coex.gro",
            trajectory_filename="test_coex.xtc",
            expected_bins="test_bins_none.npy",
            expected_rdf="test_rdf_none.npy",
            norm="none",
            name=name,
        ),
    ),
)
def case_data(request: pytest.FixtureRequest) -> RDFCaseData:
    return request.param(
        f"{request.fixturename}{request.param_index}",  # type: ignore [attr-defined]
    )
