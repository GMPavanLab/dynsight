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

    def extract_frames(self, working_dir: pathlib.Path) -> list[np.ndarray]:
        frames_dir = working_dir / "Frames"
        frames_dir.mkdir(exist_ok=True)
        capture = cv2.VideoCapture(str(self.video_path))
        if not capture.isOpened():
            msg = f"Impossible to load the video: {self.video_path}"
            raise ValueError(msg)
        frame_idx = 0
        self.frames.clear()

        while True:
            ret, frame = capture.read()
            if not ret:
                break
            self.frames.append(frame)
            frame_filename = frames_dir / f"{frame_idx}.png"
            cv2.imwrite(str(frame_filename), frame)
            frame_idx += 1
        capture.release()
        return self.frames
