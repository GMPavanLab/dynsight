from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class NNCaseData:
    expected_nn: str
    r_cut: float
    centers: str
    selection: str
    n_jobs: int
    name: str
