"""utilities package."""

from dynsight._internal.utilities.spatial_average import (
    spatialaverage,
)
from dynsight._internal.utilities.utilities import (
    find_extrema_points,
    normalize_array,
)

__all__ = [
    "find_extrema_points",
    "normalize_array",
    "spatialaverage",
]
