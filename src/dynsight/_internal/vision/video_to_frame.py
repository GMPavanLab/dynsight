from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import cv2

if TYPE_CHECKING:
    import pathlib

    import numpy as np


@dataclass
class Video:
    video_path: pathlib.Path
    frames: list[np.ndarray] = field(default_factory=list)
    _capture: cv2.VideoCapture = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._capture = cv2.VideoCapture(str(self.video_path))
        if not self._capture.isOpened():
            msg = f"Impossible to load the video: {self.video_path}"
            raise ValueError(msg)

    def __del__(self) -> None:
        if hasattr(self, "_capture") and self._capture.isOpened():
            self._capture.release()

    def count_frames(self) -> int:
        return int(self._capture.get(cv2.CAP_PROP_FRAME_COUNT))

    def resolution(self) -> tuple[int, int]:
        width = int(self._capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self._capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        return (width, height)

    def extract_frames(self, working_dir: pathlib.Path) -> None:
        frames_dir = working_dir / "frames"
        frames_dir.mkdir(exist_ok=True)
        self.frames.clear()

        self._capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
        total_frames = self.count_frames()

        for frame_idx in range(total_frames):
            ret, frame = self._capture.read()
            if not ret:
                break
            self.frames.append(frame)
            frame_filename = frames_dir / f"{frame_idx}.png"
            cv2.imwrite(str(frame_filename), frame)
