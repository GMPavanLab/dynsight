import pathlib
import tkinter as tk
from pathlib import Path

from PIL import Image


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
            x0=0,
            y0=0,
            x1=0,
            y1=self.image.height(),
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
        self.start_x = None
        self.start_y = None
        self.current_box = None
        self.boxes = []

        # Mouse bindings
        self.canvas.bind("<Button-1>", self._on_click_press())
        self.canvas.bind("<ButtonRelease-1>", self._on_click_release())
        self.canvas.bind("<B1-Motion>", self._on_mouse_drag())
        self.canvas.bind("<Motion>", self._follow_mouse())

    # Mouse functions
    def _on_click_press(self, event: tk.Event) -> None:
        """Starts drawing the box on mouse press."""
        self.start_x, self.start_y = event.x, event.y
        self.current_box = self.canvas.create_rectangle(
            x0=self.start_x,
            y0=self.start_y,
            x1=self.start_x,
            y1=self.start_y,
            outline="red",
            width=3,
        )

    def _on_click_release(self, event: tk.Event) -> None:
        """Finalize the box on mouse release."""
        x2, y2 = event.x, event.y
        x1, y1 = self.start_x, self.start_y
        abs_coords = (min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2))
        center_x = (x1 + x2) / (2 * self.image.width())
        center_y = (y1 + y2) / (2 * self.image.height())
        width_rel = abs(x2 - x1) / self.image.width()
        height_rel = abs(y2 - y1) / self.image.height()
        box_info = {
            "id": self.current_box,
            "center_x": center_x,
            "center_y": center_y,
            "width": width_rel,
            "height": height_rel,
            "abs_coords": abs_coords,
        }
        self.boxes.append(box_info)
        self.current_box = None

    def _on_mouse_drag(self, event: tk.Event) -> None:
        """Update the box coordinates while dragging the mouse."""
        sel_x, sel_y = event.x, event.y
        self.canvas.coords(
            tagOrId=self.current_box,
            x1=self.start_x,
            y1=self.start_y,
            x2=sel_x,
            y2=sel_y,
        )
        # Sync rulers too
        self.canvas.coords(
            tagOrId=self.h_line,
            x1=0,
            y1=sel_y,
            x2=self.image.width(),
            y2=sel_y,
        )
        self.canvas.coords(
            tagOrId=self.v_line,
            x1=sel_x,
            y1=0,
            x2=sel_x,
            y2=self.image.height(),
        )

    def _follow_mouse(self, event: tk.Event) -> None:
        """Sync guide lines position with mouse movement."""
        x, y = event.x, event.y
        self.canvas.coords(
            tagOrId=self.h_line,
            x1=0,
            y1=y,
            x2=self.image.width(),
            y2=y,
        )
        self.canvas.coords(
            tagOrId=self.v_line,
            x1=x,
            y1=0,
            x2=x,
            y2=self.image.height(),
        )

    # Button functions
    def _submit(self) -> None:
        """Save the cropped images."""
        pil_image = Image.open(self.image_path)
        for i, box in enumerate(self.boxes):
            abs_coords = box["abs_coords"]
            cropped_image = pil_image.crop(abs_coords)
            save_path = self.destination_folder / f"training_items/{i + 1}.png"
            cropped_image.save(save_path)
        self.master.quit()

    def _undo(self) -> None:
        """Remove the last drawn box."""
        if self.boxes:
            last_box = self.boxes.pop()
            self.canvas.delete(last_box["id"])

    def _close(self) -> None:
        """Close without saving."""
        self.master.quit()
