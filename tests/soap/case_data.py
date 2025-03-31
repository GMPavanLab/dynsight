from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class SOAPCaseData:
    expected_soap: str
    r_c: float
    l_max: int
    n_max: int
    respect_pbc: bool
    centers: str
    name: str
