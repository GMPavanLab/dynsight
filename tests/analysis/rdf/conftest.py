import pytest

from .case_data import RDFCaseData


@pytest.fixture(
    scope="session",
    params=(
        lambda name: RDFCaseData(
            topology_filename="test_coex.gro",
            trajectory_filename="test_coex.xtc",
            expected_bins="test_bins_rdf.npy",
            expected_rdf="test_rdf_rdf.npy",
            norm="rdf",
            name=name,
        ),
        lambda name: RDFCaseData(
            topology_filename="test_coex.gro",
            trajectory_filename="test_coex.xtc",
            expected_bins="test_bins_density.npy",
            expected_rdf="test_rdf_density.npy",
            norm="density",
            name=name,
        ),
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
