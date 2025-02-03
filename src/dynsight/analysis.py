"""analysis package."""

from dynsight._internal.analysis.rdf import compute_rdf
from dynsight._internal.analysis.shannon_entropy import (
    compute_data_entropy,
    compute_entropy_gain,
)
from dynsight._internal.analysis.spatial_average import (
    spatialaverage,
)
from dynsight._internal.analysis.time_correlations import (
    cross_time_correlation,
    self_time_correlation,
)

__all__ = [
    "compute_data_entropy",
    "compute_entropy_gain",
    "compute_rdf",
    "cross_time_correlation",
    "self_time_correlation",
    "spatialaverage",
]
