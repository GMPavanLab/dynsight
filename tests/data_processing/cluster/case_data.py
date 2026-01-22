from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray


@dataclass(frozen=True, slots=True)
class CleanPopCaseData:
    name: str
    expected_clean_pop: str
    threshold: float
    assigned_env: int
    excluded_env: NDArray[np.int64] | None = None
