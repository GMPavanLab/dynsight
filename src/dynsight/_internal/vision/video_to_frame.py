from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import cv2

if TYPE_CHECKING:
    import pathlib

    import numpy as np


@dataclass
class Video:
    """Load a video file and provides utilities.

    Attributes:
        video_path:
            File path to the video.
    """

    video_path: pathlib.Path
    frames: list[np.ndarray] = field(default_factory=list)

    def count_frames(self) -> int:
        """Counts the total number of frames in the video.

        Opens the video file at `video_path` and retrieves the frame count
        from the video metadata.

        Returns:
            The number of frames in the video.

        Raises:
            ValueError:
                If the video cannot be loaded.
        """
        capture = cv2.VideoCapture(str(self.video_path))
        if not capture.isOpened():
            msg = f"Impossible to load the video: {self.video_path}"
            raise ValueError(msg)
        count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        capture.release()
        return count

    def resolution(self) -> tuple[int, int]:
        """Retrieves the width and height of the video frames.

        Opens the video file at `video_path`, reads its properties, and returns
        the frame width and height in pixels. Raises a ValueError if the video
        cannot be loaded.

        Returns:
            A tuple `(width, height)` representing the frame dimensions.

        Raises:
            ValueError:
                If the video cannot be loaded.
        """
        capture = cv2.VideoCapture(str(self.video_path))
        if not capture.isOpened():
            msg = f"Impossible to load the video: {self.video_path}"
            raise ValueError(msg)
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        return (width, height)

    def extract_frames(self, working_dir: pathlib.Path) -> list[np.ndarray]:
        """Extracts all frames from the video and saves them as PNG images.

        If it doesn't exist, creates a `frames` subdirectory inside
        `working_dir', reads each frame from the video at `video_path`, appends
        it to the `frames` list, and writes it to disk.
        Clears any previously stored frames in memory before extraction.

        Parameters:
            working_dir: Directory in which to create a `frames` folder and
            save extracted PNG images.

        Returns:
            List of all frames extracted as NumPy arrays.

        Raises:
            ValueError:
                If the video cannot be loaded.
        """
        frames_dir = working_dir / "frames"
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
