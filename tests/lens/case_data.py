from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class LENSCaseData:
    expected_lens: str
    r_cut: float
    delay: int
    centers: str
    selection: str
    n_jobs: int
    name: str
