"""dynsight package."""

import contextlib

from dynsight import (
    analysis,
    lens,
    onion,
    soap,
    utilities,
)

with contextlib.suppress(ModuleNotFoundError):
    from dynsight import data_processing  # Only if cpctools is installed

with contextlib.suppress(ModuleNotFoundError):
    from dynsight import hdf5er  # Only if cpctools is installed

__all__ = [
    "analysis",
    "data_processing",
    "hdf5er",
    "lens",
    "onion",
    "soap",
    "utilities",
]
