from __future__ import annotations

import pathlib
import tkinter as tk
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from .vision_gui import VisionGUI

if TYPE_CHECKING:
    from .video_to_frame import Video


class Detect:
    def __init__(
        self,
        input_video: Video,
        project_folder: pathlib.Path = Path("output_folder"),
    ) -> None:
        self.project_folder = project_folder
        self.frames_dir = project_folder / "frames"
        if not (self.frames_dir.exists() and self.frames_dir.is_dir()):
            input_video.extract_frames(project_folder)

    def synthesize(
        self,
        n_epochs: int = 2,
        dataset_dimension: int = 1000,
        reference_img_path: None | pathlib.Path = None,
        training_set_fraction: float = 0.7,
        collage_size: tuple = (1080, 1080),
        sample_from: Literal["gui"] = "gui",
    ) -> None:
        if sample_from == "gui":
            if reference_img_path is None:
                reference_img_path = self.frames_dir / "0.png"
            root = tk.Tk()
            VisionGUI(
                master=root,
                image_path=reference_img_path,
                destination_folder=self.project_folder,
            )
            root.mainloop()
            training_items_path = self.project_folder / "training_items"
            if not (
                training_items_path.exists() and training_items_path.is_dir()
            ):
                msg = "'training_items' folder not created or not found"
                raise ValueError(msg)
