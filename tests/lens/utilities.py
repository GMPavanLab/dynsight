import re

import numba
import numpy as np


@numba.jit  # type: ignore[misc]
def is_sorted(a: np.ndarray, /) -> bool:  # type: ignore[type-arg]
    """Checks if an array is sorted.

    See https://stackoverflow.com/a/47004533.

    """
    for i in range(a.size - 1):  # noqa: SIM110
        if a[i + 1] < a[i]:
            return False
    return True


__PropertiesFinder = re.compile('Properties="{0,1}(.*?)"{0,1}(?: |$)', flags=0)
__LatticeFinder = re.compile('Lattice="(.*?)"', flags=0)
