"""logging package."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from dynsight.trajectory import Insight

class Logger:
    """Creates and save human-readible log."""

    def __init__(self) -> None:
        self._log: list[str] = []
        self._registered_data: list["Insight"] = []

    def log(self, msg: str) -> None:
        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
        entry = f"[{timestamp}] {msg}"
        self._log.append(entry)

    def save(self, filename: Path) -> None:
        with filename.open("w", encoding="utf-8") as f:
            f.write("\n".join(self._log))

    def clear(self) -> None:
        self._log = []
        self._registered_data = []

    def get(self) -> str:
        return "\n".join(self._log)
    
    def register_data(self, insight: "Insight") -> None:
        self._registered_data.append(insight)
        self.log(f"Registered data: {insight}")
    
    def extract_datasets(
        self,
        output_dir: Path | str = Path("output"),
    ) -> list[Path]:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        saved_paths: list[Path] = []
        for index, insight in enumerate(self._registered_data, start=1):
            base_filename = f"dataset_{index}"
            if isinstance(insight.meta, dict) and insight.meta:
                sanitized_values = []

                for value in insight.meta.values():
                    value_str = str(value)
                    sanitized = "".join(
                        ch if ch.isalnum() or ch in {"-", "_"} else "_"
                        for ch in value_str
                    ).strip("_")
                    if sanitized:
                        sanitized_values.append(sanitized)

                if sanitized_values:
                    base_filename = "_".join(sanitized_values)

            filename = output_path / f"{base_filename}.npy"

            counter = 1
            candidate = filename
            while candidate.exists():
                candidate = output_path / f"{base_filename}_{counter}.npy"
                counter += 1
            filename = candidate

            np.save(filename, insight.dataset)
            saved_paths.append(filename)
            self.log(f"Dataset saved to {filename}.")

        self._registered_data = []

        return saved_paths

logger = Logger()
