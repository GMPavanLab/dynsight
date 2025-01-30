"""dynsight package."""

import contextlib

from dynsight import (
    analysis,
    lens,
    onion,
    soap,
    utilities,
)

__all__ = [
    "analysis",
    "lens",
    "onion",
    "soap",
    "utilities",
]

with contextlib.suppress(ModuleNotFoundError):
    from dynsight import data_processing  # Only if cpctools is installed

    __all__ += ["data_processing"]

with contextlib.suppress(ModuleNotFoundError):
    from dynsight import hdf5er  # Only if cpctools is installed

    __all__ += ["hdf5er"]
