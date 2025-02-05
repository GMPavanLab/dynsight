from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True, slots=True)
class RDFCaseData:
    topology_filename: str
    trajectory_filename: str
    expected_bins: str
    expected_rdf: str
    norm: Literal["rdf", "density", "none"]
    name: str
