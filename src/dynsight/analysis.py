"""analysis package."""

from dynsight._internal.analysis.entropy import (
    compute_data_entropy,
    compute_entropy_gain,
    compute_sample_entropy,
    sample_entropy,
)
from dynsight._internal.analysis.rdf import compute_rdf
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
    "compute_sample_entropy",
    "cross_time_correlation",
    "sample_entropy",
    "self_time_correlation",
    "spatialaverage",
]
