from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class CleanPopCaseData:
    name: str
    expected_clean_pop: str
    threshold: float
    assigned_env: int
    excluded_env: int | list[int] | None = None
