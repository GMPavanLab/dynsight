from __future__ import annotations

import pathlib
import tkinter as tk
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image


@dataclass
class Box:
    id: int
    center_x: float
    center_y: float
    width: float
    height: float
    abs_coords: tuple[float, float, float, float]


class VisionGUI:
    """GUI for interactively labeling images by drawing bounding boxes."""

    def __init__(
        self,
        master: tk.Tk,
        image_path: pathlib.Path,
        destination_folder: pathlib.Path = Path(__file__).parent,
    ) -> None:
        self.master = master
        self.image_path = image_path
        self.destination_folder = destination_folder
        self.master.title("Dynsight: Label tool")
        try:
            self.image = tk.PhotoImage(file=image_path)
        except Exception as e:
            msg = f"Error loading image: {e}"
            raise ValueError(msg) from e
        # Setup the main grid
        self.master.rowconfigure(index=0, weight=1)
        # Image
        self.master.columnconfigure(index=0, weight=1)
        # Sidebar
        self.master.columnconfigure(index=1, weight=1)

        # Image canvas
        self.canvas = tk.Canvas(
            master=self.master,
            width=self.image.width(),
            height=self.image.height(),
            cursor="crosshair",
        )
        self.canvas.grid(row=0, column=0, sticky="nsew")
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.image)
        # Rulers
        self.h_line = self.canvas.create_line(
            0,  # x0
            0,  # y0
            self.image.width(),  # x1
            0,  # y1
            fill="blue",
            dash=(2, 2),
            width=3,
        )
        self.v_line = self.canvas.create_line(
            0,  # x0
            0,  # y0
            0,  # x1
            self.image.height(),  # y1
            fill="blue",
            dash=(2, 2),
            width=3,
        )
        # Sidebar
        self.sidebar = tk.Frame(
            master=self.master,
            width=150,
            padx=10,
            pady=10,
        )
        self.sidebar.grid(row=0, column=1, sticky="ns")
        self.sidebar.grid_propagate(flag=False)

        # Buttons
        self.submit_button = tk.Button(
            self.sidebar,
            text="Submit",
            command=self._submit,
        )
        self.submit_button.pack(pady=10, fill="x")

        self.undo_button = tk.Button(
            self.sidebar,
            text="Undo",
            command=self._undo,
        )
        self.undo_button.pack(pady=10, fill="x")

        self.close_button = tk.Button(
            self.sidebar,
            text="Close",
            command=self._close,
        )
        self.close_button.pack(pady=10, fill="x")

        # Labelling variables
        self.start_x = 0.0
        self.start_y = 0.0
        self.current_box = 0
        self.boxes: list[Box] = []

        # Mouse bindings
        self.canvas.bind("<Button-1>", self._on_click_press)
        self.canvas.bind("<ButtonRelease-1>", self._on_click_release)
        self.canvas.bind("<B1-Motion>", self._on_mouse_drag)
        self.canvas.bind("<Motion>", self._follow_mouse)

    # Mouse functions
    def _on_click_press(self, event: tk.Event[Any]) -> None:
        """Starts drawing the box on mouse press."""
        self.start_x, self.start_y = event.x, event.y
        self.current_box = self.canvas.create_rectangle(
            self.start_x,  # x0
            self.start_y,  # y0
            self.start_x,  # x1
            self.start_y,  # y1
            outline="red",
            width=3,
        )

    def _on_click_release(self, event: tk.Event[Any]) -> None:
        """Finalize the box on mouse release."""
        x2, y2 = event.x, event.y
        x1, y1 = self.start_x, self.start_y
        abs_coords = (min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2))
        center_x = (x1 + x2) / (2 * self.image.width())
        center_y = (y1 + y2) / (2 * self.image.height())
        width_rel = abs(x2 - x1) / self.image.width()
        height_rel = abs(y2 - y1) / self.image.height()
        box_info = Box(
            id=self.current_box,
            center_x=center_x,
            center_y=center_y,
            width=width_rel,
            height=height_rel,
            abs_coords=abs_coords,
        )
        self.boxes.append(box_info)
        self.current_box = 0

    def _on_mouse_drag(self, event: tk.Event[Any]) -> None:
        """Update the box coordinates while dragging the mouse."""
        sel_x, sel_y = event.x, event.y
        self.canvas.coords(
            self.current_box,  # ID
            self.start_x,  # x0
            self.start_y,  # y0
            sel_x,  # x1
            sel_y,  # y1
        )
        # Sync rulers too
        self.canvas.coords(
            self.h_line,  # ID
            0,  # x0
            sel_y,  # y0
            self.image.width(),  # x1
            sel_y,  # y1
        )
        self.canvas.coords(
            self.v_line,  # ID
            sel_x,  # x0
            0,  # y0
            sel_x,  # x1
            self.image.height(),  # y1
        )

    def _follow_mouse(self, event: tk.Event[Any]) -> None:
        """Sync guide lines position with mouse movement."""
        x, y = event.x, event.y
        self.canvas.coords(
            self.h_line,  # ID
            0,  # x0
            y,  # y0
            self.image.width(),  # x1
            y,  # y1
        )
        self.canvas.coords(
            self.v_line,  # ID
            x,  # x0
            0,  # y0
            x,  # x1
            self.image.height(),  # y1
        )

    # Button functions
    def _submit(self) -> None:
        """Save the cropped images."""
        cropped_img_folder = self.destination_folder / "training_items"
        cropped_img_folder.mkdir(exist_ok=True)
        pil_image = Image.open(self.image_path)
        for i, box in enumerate(self.boxes):
            abs_coords = box.abs_coords
            cropped_image = pil_image.crop(abs_coords)
            save_path = cropped_img_folder / f"{i}.png"
            cropped_image.save(save_path)
        self.master.destroy()

    def _undo(self) -> None:
        """Remove the last drawn box."""
        if self.boxes:
            last_box = self.boxes.pop()
            self.canvas.delete(last_box.id)

    def _close(self) -> None:
        """Close without saving."""
        self.master.destroy()
