"""logging package."""

from datetime import datetime, timezone
from pathlib import Path


class Logger:
    """Creates and save human-readible log."""

    def __init__(self) -> None:
        self._log: list[str] = []

    def log(self, msg: str) -> None:
        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
        entry = f"[{timestamp}] {msg}"
        self._log.append(entry)

    def save(self, filename: Path) -> None:
        with filename.open("w", encoding="utf-8") as f:
            f.write("\n".join(self._log))

    def clear(self) -> None:
        self._log = []

    def get(self) -> str:
        return "\n".join(self._log)


logger = Logger()
