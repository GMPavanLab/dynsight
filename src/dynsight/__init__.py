"""dynsight package."""

from dynsight import (
    analysis,
    lens,
    onion,
    soap,
    utilities,
)

try:
    from dynsight import data_processing
except ModuleNotFoundError:
    data_processing = None

try:
    from dynsight import hdf5er
except ModuleNotFoundError:
    hdf5er = None

__all__ = [
    "analysis",
    "data_processing",
    "hdf5er",
    "lens",
    "onion",
    "soap",
    "utilities",
]
