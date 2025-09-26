from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class LENSCaseData:
    num_processes: int
    name: str
