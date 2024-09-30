"""analysis package."""

from dynsight._internal.analysis.rdf import compute_rdf
from dynsight._internal.analysis.shannon_entropy import (
    compute_entropy_gain,
)

__all__ = ["compute_entropy_gain", "compute_rdf"]
