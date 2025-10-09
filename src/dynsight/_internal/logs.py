"""logging package."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np

if TYPE_CHECKING:
    from dynsight.trajectory import Insight


class Logger:
    """Creates and save human-readible log."""

    def __init__(self) -> None:
        self._log: list[str] = []
        self._registered_data: list[Insight] = []

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

    def register_data(self, insight: Insight) -> None:
        self._registered_data.append(insight)
        self.log(f"Registered data: {insight}")

    def extract_datasets(
        self,
        output_dir: Path | str = Path("analysis_archive"),
    ) -> list[Path]:
        output_path = Path(output_dir)
        zip_parent = output_path.parent
        zip_parent.mkdir(parents=True, exist_ok=True)

        saved_paths: list[Path] = []
        dataset_files: list[Path] = []

        with TemporaryDirectory(
            dir=zip_parent, prefix=f"{output_path.name}_"
        ) as temp_dir:
            temp_path = Path(temp_dir)

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

                filename = temp_path / f"{base_filename}.npy"
                counter = 1
                while filename.exists():
                    filename = temp_path / f"{base_filename}_{counter}.npy"
                    counter += 1

                np.save(filename, insight.dataset)
                dataset_files.append(filename)
                self.log(f"Dataset prepared: {filename.name}.")

            self._registered_data = []

            if dataset_files:
                zip_filename = self._create_zip_archive(
                    dataset_files, output_path, zip_parent
                )
                saved_paths.append(zip_filename)
                self.log(f"Output directory zipped to {zip_filename}.")

        return saved_paths

    def _create_zip_archive(
        self,
        dataset_files: list[Path],
        output_path: Path,
        zip_parent: Path,
    ) -> Path:
        zip_base = output_path.name
        zip_filename = zip_parent / f"{zip_base}.zip"

        counter = 1
        while zip_filename.exists():
            zip_filename = zip_parent / f"{zip_base}_{counter}.zip"
            counter += 1

        with ZipFile(zip_filename, "w", compression=ZIP_DEFLATED) as archive:
            for file_path in sorted(dataset_files):
                archive.write(file_path, arcname=file_path.name)

        return zip_filename


logger = Logger()
