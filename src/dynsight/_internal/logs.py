"""logging package."""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np

if TYPE_CHECKING:
    from dynsight.trajectory import Insight

import logging

COLORS = {
    "DEBUG": "\033[36m",
    "INF": "\033[32m",
    "WRN": "\033[33m",
    "ERR": "\033[31m",
    "CRT": "\033[41m",
}
RESET = "\033[0m"

LEVEL_ALIASES = {
    "DEBUG": "DBG",
    "INFO": "INF",
    "WARNING": "WRN",
    "ERROR": "ERR",
    "CRITICAL": "CRT",
}


class ColorFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        alias = LEVEL_ALIASES.get(record.levelname, record.levelname)
        color = COLORS.get(alias, "")
        record.levelname = f"{color}{alias:^3}{RESET}"
        record.msg = f"{color}{record.msg}{RESET}"
        return super().format(record)


handler = logging.StreamHandler(sys.stdout)
formatter = ColorFormatter(
    fmt="%(asctime)s | %(levelname)s | %(message)s", datefmt="%Y-%m-%d %H:%M"
)
handler.setFormatter(formatter)

console = logging.getLogger(__name__)
console.setLevel(logging.DEBUG)
console.addHandler(handler)
console.propagate = False


class Logger:
    """Creates and save human-readible log."""

    def __init__(self) -> None:
        self._log: list[str] = []
        self._registered_data: list[Insight] = []

    def log(self, msg: str) -> None:
        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
        history_entry = f"[{timestamp}] {msg}"
        console.info(msg)
        self._log.append(history_entry)

    def save_history(self, filename: Path) -> None:
        with filename.open("w", encoding="utf-8") as f:
            f.write("\n".join(self._log))

    def clear_history(self) -> None:
        self._log = []
        self._registered_data = []

    def get(self) -> str:
        return "\n".join(self._log)

    def register_data(self, insight: Insight) -> None:
        for existing in self._registered_data:
            if existing.meta == insight.meta:
                console.warning("Insight already registered, skipping.")
                return

        insight_bytes = insight.dataset.nbytes
        self._registered_data.append(insight)

        total_bytes = sum(
            x.dataset.nbytes
            if isinstance(x.dataset, np.ndarray)
            else sys.getsizeof(x.dataset)
            for x in self._registered_data
        )

        bytes_in_kb = 1024
        bytes_in_mb = bytes_in_kb**2
        bytes_in_gb = bytes_in_kb**3

        def _format_size(bytes_val: int) -> str:
            gb = bytes_val / bytes_in_gb
            mb = bytes_val / bytes_in_mb
            kb = bytes_val / bytes_in_kb
            thr = 0.5
            if gb >= thr:
                return f"{gb:.2f} GB"
            if mb >= thr:
                return f"{mb:.2f} MB"
            return f"{kb:.2f} KB"

        added_size = _format_size(insight_bytes)
        total_size = _format_size(total_bytes)

        console.warning(f"Registering dataset with size: {added_size}.")
        console.warning(f"RAM occupied by dynsight datasets: {total_size}.")

    def extract_datasets(
        self,
        output_dir: Path | str = Path("analysis_archive"),
    ) -> None:
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
