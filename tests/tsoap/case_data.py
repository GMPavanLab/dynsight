from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class TimeSOAPCaseData:
    expected_tsoap: str
    r_c: float
    delay: int
    name: str
