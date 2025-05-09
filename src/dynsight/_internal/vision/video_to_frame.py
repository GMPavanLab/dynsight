from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import cv2

if TYPE_CHECKING:
    import pathlib


@dataclass
class Video:
    """Load a video file and provide utilities.

    This class loads a video from a file path and provides methods
    to retrieve video information and extract frames.

    * Author: Simone Martino

    .. caution::
        This part of the code is still under development and may
        contain errors.

    Parameters:
        video_path: Path to the video file.

    """

    video_path: pathlib.Path
    # OpenCV video capture object.
    _capture: cv2.VideoCapture = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Load the the video."""
        self._capture = cv2.VideoCapture(str(self.video_path))
        if not self._capture.isOpened():
            msg = f"Impossible to load the video: {self.video_path}"
            raise ValueError(msg)

    def __del__(self) -> None:
        """Close the the video."""
        if hasattr(self, "_capture") and self._capture.isOpened():
            self._capture.release()

    def count_frames(self) -> int:
        """Counts the total number of frames in the video.

        Returns:
            The number of frames in the video.
        """
        return int(self._capture.get(cv2.CAP_PROP_FRAME_COUNT))

    def resolution(self) -> tuple[int, int]:
        """Retrieves video width and height in pixels.

        Returns:
            A tuple `(width, height)` representing the frame dimensions.
        """
        width = int(self._capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self._capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        return (width, height)

    def extract_frames(self, working_dir: pathlib.Path) -> None:
        """Extracts all frames from the video and saves them as PNG images.

        If it doesn't exist, creates a ``frames`` subdirectory inside
        ``working_dir``, reads each frame from the video and
        writes them to disk.

        Parameters:
            working_dir: Directory in which to create a `frames` folder and
            save extracted PNG images.

        """
        frames_dir = working_dir / "frames"
        frames_dir.mkdir(exist_ok=True)

        self._capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
        total_frames = self.count_frames()

        for frame_idx in range(total_frames):
            _, frame = self._capture.read()
            frame_filename = frames_dir / f"{frame_idx}.png"
            cv2.imwrite(str(frame_filename), frame)
