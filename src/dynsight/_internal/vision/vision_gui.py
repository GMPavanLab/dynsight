from __future__ import annotations

import pathlib
import tkinter as tk
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image, ImageTk
from PIL.Image import Resampling


@dataclass
class Box:
    id: int
    center_x: float
    center_y: float
    width: float
    height: float
    abs_coords: tuple[int, int, int, int]


class VisionGUI:
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

        screen_width = self.master.winfo_screenwidth()
        screen_height = self.master.winfo_screenheight()
        self.master.geometry(f"{screen_width // 2}x{screen_height // 2}")

        pil_image = Image.open(self.image_path)
        self.original_image = pil_image
        self.original_width, self.original_height = pil_image.size

        self.scale = 1.0
        self.canvas_image_id: int | None = None

        self.master.rowconfigure(index=0, weight=1)
        self.master.columnconfigure(index=0, weight=1)
        self.master.columnconfigure(index=1, weight=0)

        self.canvas = tk.Canvas(master=self.master, cursor="crosshair")
        self.canvas.grid(row=0, column=0, sticky="nsew")

        self.sidebar = tk.Frame(
            master=self.master,
            width=150,
            padx=10,
            pady=10,
        )
        self.sidebar.grid(row=0, column=1, sticky="ns")
        self.sidebar.grid_propagate(flag=False)

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

        self.start_x = 0
        self.start_y = 0
        self.current_box = 0
        self.boxes: list[Box] = []

        self.h_line = self.canvas.create_line(
            0,
            0,
            0,
            0,
            fill="blue",
            dash=(2, 2),
            width=3,
        )
        self.v_line = self.canvas.create_line(
            0,
            0,
            0,
            0,
            fill="blue",
            dash=(2, 2),
            width=3,
        )

        self.canvas.bind("<Button-1>", self._on_click_press)
        self.canvas.bind("<ButtonRelease-1>", self._on_click_release)
        self.canvas.bind("<B1-Motion>", self._on_mouse_drag)
        self.canvas.bind("<Motion>", self._follow_mouse)
        self.master.bind("<Configure>", self._resize)

        self._resize()

    def _resize(self, _: tk.Event[Any] | None = None) -> None:
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        size_thr = 10
        if canvas_width < size_thr or canvas_height < size_thr:
            return

        self.scale = min(
            canvas_width / self.original_width,
            canvas_height / self.original_height,
        )
        resized = self.original_image.resize(
            (
                int(self.original_width * self.scale),
                int(self.original_height * self.scale),
            ),
            Resampling.LANCZOS,
        )
        self.tk_image = ImageTk.PhotoImage(resized)
        self.canvas.delete("all")
        self.canvas_image_id = self.canvas.create_image(
            0,
            0,
            anchor=tk.NW,
            image=self.tk_image,
        )

        # Re-draw boxes
        for box in self.boxes:
            x1, y1, x2, y2 = [int(v * self.scale) for v in box.abs_coords]
            box.id = self.canvas.create_rectangle(
                x1,
                y1,
                x2,
                y2,
                outline="red",
                width=3,
            )

        # Recreate guide lines
        self.h_line = self.canvas.create_line(
            0,
            0,
            canvas_width,
            0,
            fill="blue",
            dash=(2, 2),
            width=3,
        )
        self.v_line = self.canvas.create_line(
            0,
            0,
            0,
            canvas_height,
            fill="blue",
            dash=(2, 2),
            width=3,
        )

    def _on_click_press(self, event: tk.Event[Any]) -> None:
        self.start_x, self.start_y = event.x, event.y
        self.current_box = self.canvas.create_rectangle(
            self.start_x,
            self.start_y,
            self.start_x,
            self.start_y,
            outline="red",
            width=3,
        )

    def _on_click_release(self, event: tk.Event[Any]) -> None:
        x2, y2 = event.x, event.y
        x1, y1 = self.start_x, self.start_y
        x1_orig = int(min(x1, x2) / self.scale)
        y1_orig = int(min(y1, y2) / self.scale)
        x2_orig = int(max(x1, x2) / self.scale)
        y2_orig = int(max(y1, y2) / self.scale)

        abs_coords = (x1_orig, y1_orig, x2_orig, y2_orig)

        center_x = (x1_orig + x2_orig) / (2 * self.original_width)
        center_y = (y1_orig + y2_orig) / (2 * self.original_height)
        width_rel = abs(x2_orig - x1_orig) / self.original_width
        height_rel = abs(y2_orig - y1_orig) / self.original_height

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
        self._resize()

    def _on_mouse_drag(self, event: tk.Event[Any]) -> None:
        sel_x, sel_y = event.x, event.y
        self.canvas.coords(
            self.current_box,
            self.start_x,
            self.start_y,
            sel_x,
            sel_y,
        )
        self.canvas.coords(
            self.h_line,
            0,
            sel_y,
            self.canvas.winfo_width(),
            sel_y,
        )
        self.canvas.coords(
            self.v_line,
            sel_x,
            0,
            sel_x,
            self.canvas.winfo_height(),
        )

    def _follow_mouse(self, event: tk.Event[Any]) -> None:
        x, y = event.x, event.y
        self.canvas.coords(self.h_line, 0, y, self.canvas.winfo_width(), y)
        self.canvas.coords(self.v_line, x, 0, x, self.canvas.winfo_height())

    def _submit(self) -> None:
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
        if self.boxes:
            last_box = self.boxes.pop()
            self.canvas.delete(last_box.id)
            self._resize()

    def _close(self) -> None:
        self.master.destroy()
