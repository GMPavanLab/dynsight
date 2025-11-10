"""logging package."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING, Any
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


@dataclass
class RecordedDataset:
    meta: Any
    path: Path


class Logger:
    """Create and save human-readible log."""

    def __init__(
        self,
        *,
        auto_recording: bool = True,
    ) -> None:
        self._log: list[str] = []
        self._recorded_data: list[RecordedDataset] = []
        self._temp_dir: TemporaryDirectory[str] | None = None
        self._temp_path: Path | None = None
        self.auto_recording = auto_recording

    def configure(
        self,
        *,
        auto_recording: bool = True,
    ) -> None:
        """Adjusts the runtime configuration of the logger.

        Parameters:
            auto_recording:
                Enables or disables automatic dataset recording.
                When set to `True`, every processed dataset will be
                automatically saved into a temporary archive.
                When `False`, datasets must be explicitly registered
                via `register_data()`.
        """
        self.auto_recording = auto_recording
        state = "enabled" if auto_recording else "disabled"
        console.info(f"Automatic dataset recording {state}.")

    def log(self, msg: str) -> None:
        """Records an informational message to the log.

        Parameters:
            msg:
                The message to record.
        """
        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
        history_entry = f"[{timestamp}] {msg}"
        console.info(msg)
        self._log.append(history_entry)

    def save_history(self, filename: Path) -> None:
        """Saves the current log history to a text file.

        Parameters:
            filename:
                The file path where the log history will be written.
        """
        with filename.open("w", encoding="utf-8") as f:
            f.write("\n".join(self._log))

    def clear_history(self) -> None:
        """Clears the current log history and registered datasets."""
        self._log = []
        self._cleanup_temp_dir()
        self._recorded_data = []

    def get(self) -> str:
        """Retrieves the current log history as a string."""
        return "\n".join(self._log)

    def record_data(self, insight: Insight) -> None:
        """Record and save a dataset associated with an `Insight` instance.

        Parameters:
            insight:
                the `Insight` to be registered.
        """
        for existing in self._recorded_data:
            if existing.meta == insight.meta:
                console.warning("Insight already registered, skipping.")
                return

        insight_bytes = insight.dataset.nbytes

        temp_path = self._ensure_temp_dir()

        base_filename = self._build_base_filename(insight)
        filename = self._make_unique_filename(temp_path, base_filename)

        np.save(filename, insight.dataset)

        dataset_entry = RecordedDataset(meta=insight.meta, path=filename)
        self._recorded_data.append(dataset_entry)

        total_bytes = sum(
            entry.path.stat().st_size
            for entry in self._recorded_data
            if entry.path.exists()
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
        console.warning(f"Disk used by dynsight datasets: {total_size}.")
        self.log(f"Dataset prepared: {filename.name}.")

    def extract_datasets(
        self,
        output_dir: Path | str = Path("analysis_archive"),
    ) -> None:
        """Exports all recorded datasets into a ZIP archive.

        Parameters:
            output_dir:
                The directory where the ZIP archive will be saved.
        """
        if self._recorded_data == []:
            console.error("No datasets to extract.")
            return
        output_path = Path(output_dir)
        zip_parent = output_path.parent
        zip_parent.mkdir(parents=True, exist_ok=True)

        saved_paths: list[Path] = []
        dataset_files = [
            entry.path
            for entry in self._recorded_data
            if entry.path.exists()
        ]

        if dataset_files:
            zip_filename = self._create_zip_archive(
                dataset_files, output_path, zip_parent
            )
            saved_paths.append(zip_filename)
            self.log(f"Output directory zipped to {zip_filename}.")

        self._cleanup_temp_dir()
        self._recorded_data = []

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

    def _ensure_temp_dir(self) -> Path:
        if self._temp_dir is None or self._temp_path is None:
            self._temp_dir = TemporaryDirectory(prefix="analysis_archive_")
            self._temp_path = Path(self._temp_dir.name)
        return self._temp_path

    def _cleanup_temp_dir(self) -> None:
        if self._temp_dir is not None:
            self._temp_dir.cleanup()
        self._temp_dir = None
        self._temp_path = None

    def _build_base_filename(self, insight: Insight) -> str:
        base_filename = "dataset"
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
        return base_filename

    def _make_unique_filename(
        self, temp_path: Path, base_filename: str
    ) -> Path:
        filename = temp_path / f"{base_filename}.npy"
        counter = 1
        while filename.exists():
            filename = temp_path / f"{base_filename}_{counter}.npy"
            counter += 1
        return filename


logger = Logger()
