"""analysis package."""

from dynsight._internal.analysis.entropy import (
    compute_entropy_gain,
    compute_entropy_gain_multi,
    compute_kl_entropy,
    compute_negentropy,
    compute_shannon,
    compute_shannon_multi,
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
    "compute_entropy_gain",
    "compute_entropy_gain_multi",
    "compute_kl_entropy",
    "compute_negentropy",
    "compute_rdf",
    "compute_shannon",
    "compute_shannon_multi",
    "cross_time_correlation",
    "sample_entropy",
    "self_time_correlation",
    "spatialaverage",
]
