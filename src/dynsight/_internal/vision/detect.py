from __future__ import annotations

import pathlib
import random
import tkinter as tk
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import matplotlib.pyplot as plt
import numpy as np
import yaml
from PIL import Image
from scipy.optimize import curve_fit
from scipy.stats import norm
from ultralytics import YOLO

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
        self.video_size = input_video.resolution()
        self.n_frames = input_video.count_frame()

    def synthesize(
        self,
        dataset_dimension: int = 1000,
        reference_img_path: None | pathlib.Path = None,
        validation_set_fraction: float = 0.2,
        collage_size: tuple = (1080, 1080),
        collage_max_repeats: int = 30,
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

        # Build synthetic dataset
        syn_dataset_path = self.project_folder / "synthetic_dataset"

        images_train_dir = syn_dataset_path / "images" / "train"
        images_val_dir = syn_dataset_path / "images" / "val"
        labels_train_dir = syn_dataset_path / "labels" / "train"
        labels_val_dir = syn_dataset_path / "labels" / "val"

        syn_dataset_path.mkdir(exist_ok=True)
        for d in [
            images_train_dir,
            images_val_dir,
            labels_train_dir,
            labels_val_dir,
        ]:
            d.mkdir(exist_ok=True, parents=True)

        num_val = int(dataset_dimension * validation_set_fraction)
        num_train = dataset_dimension - num_val
        remaining = dataset_dimension - (num_train + num_val)
        num_train += remaining

        assignements = ["train"] * num_train + ["val"] * num_val
        random.shuffle(assignements)

        for i in range(1, dataset_dimension + 1):
            collage, label_lines = self._create_collage(
                images_folder=training_items_path,
                width=collage_size[0],
                height=collage_size[1],
                max_repeats=collage_max_repeats,
            )
            subset = assignements[i - 1]
            if subset == "train":
                image_save_path = images_train_dir / f"{i}.png"
                label_save_path = labels_train_dir / f"{i}.txt"
            else:
                image_save_path = images_val_dir / f"{i}.png"
                label_save_path = labels_val_dir / f"{i}.txt"

            collage.save(image_save_path)
            with label_save_path.open("w") as f:
                for line in label_lines:
                    f.write(line + "\n")
            # YAML file
            yaml_file_name = self.project_folder / "training_options.yaml"
            yaml_config_data = {
                "path": str(syn_dataset_path.resolve()),
                "train": "images/train",
                "val": "images/val",
                "nc": 1,
                "names": ["obj"],
            }
            with Path.open(yaml_file_name, "w") as file:
                yaml.dump(yaml_config_data, file, sort_keys=False)

    def train(
        self,
        yaml_file: pathlib.Path,
        initial_model: str | pathlib.Path = "yolo12x.pt",
        training_name: str | None = None,
        training_epochs: int = 100,
        training_patience: int = 100,
        batch_size: int = 16,
        workers: int = 8,
        device: int | str | list[int] | None = None,
    ) -> None:
        model = YOLO(initial_model)
        model.train(
            data=yaml_file,
            epochs=training_epochs,
            patience=training_patience,
            batch=batch_size,
            imgsz=self.video_size,
            workers=workers,
            project=self.project_folder / "models",
            name=training_name,
            device=device,
            plots=False,
        )

    def predict(
        self,
        model_path: str | pathlib.Path,
        detections_iou: float = 0.1,
        prediction_name: str = "prediction",
    ) -> None:
        model = YOLO(model_path)
        model.predict(
            project=self.project_folder,
            source=self.frames_dir / "0.png",
            name=prediction_name,
            augment=True,
            line_width=2,
            save=True,
            show_labels=False,
            save_txt=True,
            save_conf=True,
            iou=detections_iou,
            max_det=20000,
        )

    def fit(
        self,
        initial_dataset: pathlib.Path,
        initial_model: str | pathlib.Path = "yolo12x.pt",
        training_epochs: int = 100,
        training_patience: int = 100,
        batch_size: int = 16,
        workers: int = 8,
        device: int | str | list[int] | None = None,
    ) -> None:
        current_dataset = initial_dataset
        guess_model_name = "v0"
        """
        self.train(
            yaml_file=current_dataset,
            initial_model=initial_model,
            training_epochs=2,
            batch_size=batch_size,
            workers=workers,
            device=device,
            training_name=guess_model_name,
        )"""
        current_model = YOLO(
            self.project_folder
            / "models"
            / guess_model_name
            / "weights"
            / "best.pt"
        )
        prediction_number = 0
        detection_results = []
        for f in range(0, self.n_frames, 10):  # TEMPORARY
            frame_file = self.frames_dir / f"{f}.png"
            prediction = current_model.predict(
                source=frame_file,
                imgsz=self.video_size,
                augment=True,
                save=True,
                save_txt=True,
                save_conf=True,
                show_labels=False,
                name=f"attempt_{prediction_number}",
                iou=0.1,
                max_det=20000,
                project=self.project_folder / "predictions",
                line_width=2,
                exist_ok=True,
            )
            n_detection = len(prediction)
            xywh = prediction[0].boxes.xywh.cpu().numpy()
            conf = prediction[0].boxes.conf.cpu().numpy()
            cls = prediction[0].boxes.cls.cpu().numpy()

            for i in range(n_detection):
                x, y, w, h = xywh[i]
                detection_results.append(
                    {
                        "frame": f,
                        "class_id": int(cls[i]),
                        "center_x": float(x),
                        "center_y": float(y),
                        "width": float(w),
                        "height": float(h),
                        "confidence": float(conf[i]),
                    }
                )
        widths = np.array([d["width"] for d in detection_results], dtype=float)
        heights = np.array(
            [d["height"] for d in detection_results],
            dtype=float,
        )
        outliers_plt_folder = (
            self.project_folder
            / "predictions"
            / f"attempt_{prediction_number}"
            / "outliers"
        )
        outliers_plt_folder.mkdir(exist_ok=True)
        out_width = _find_outliers(
            distribution=widths,
            save_path=outliers_plt_folder,
            fig_name="width",
        )
        out_height = _find_outliers(
            distribution=heights,
            save_path=outliers_plt_folder,
            fig_name="height",
        )
        print(out_width)
        print(out_height)

    def _create_collage(
        self,
        images_folder: pathlib.Path,
        width: int,
        height: int,
        patience: int = 1000,
        max_repeats: int = 1,
    ) -> tuple[Image.Image, list[str]]:
        collage = Image.new("RGBA", (width, height), (255, 255, 255, 255))
        placed_rects = []
        label_lines = []
        cropped_images = []
        for file in images_folder.iterdir():
            if file.suffix.lower() in {".png", ".jpg", ".jpeg"}:
                try:
                    img = Image.open(file).convert("RGBA")
                    cropped_images.append(img)
                except Exception as e:
                    msg = f"Error with {file.name}: {e}"
                    print(msg)
        if not cropped_images:
            msg = "No images found in the specified folder"
            raise ValueError(msg)

        total_placement = len(cropped_images) * max_repeats
        placed_count = 0

        while placed_count < total_placement:
            cropped = random.choice(cropped_images)
            w, h = cropped.size
            max_x = width - w
            max_y = height - h
            placed = False

            for _ in range(patience):
                x = random.randint(0, max_x)
                y = random.randint(0, max_y)
                new_rect = (x, y, x + w, y + h)
                overlap = any(
                    not (
                        new_rect[2] <= rect[0]
                        or new_rect[0] >= rect[2]
                        or new_rect[3] <= rect[1]
                        or new_rect[1] >= rect[3]
                    )
                    for rect in placed_rects
                )
                if not overlap:
                    collage.paste(cropped, (x, y), cropped)
                    placed_rects.append(new_rect)

                    center_x = (x + h / 2) / width
                    center_y = (y + h / 2) / height
                    width_norm = w / width
                    height_norm = h / height
                    label_line = (
                        f"0 {center_x:.6f} {center_y:.6f} "
                        f"{width_norm:.6f} {height_norm:.6f}"
                    )
                    label_lines.append(label_line)
                    placed_count += 1
                    placed = True
                    break
            if not placed:
                break
        return collage, label_lines


def _find_outliers(
    distribution: np.ndarray,
    save_path: pathlib.Path,
    fig_name: str,
    thr: float = 1e-5,
) -> np.ndarray:
    def _gaussian(
        x: np.ndarray, mu: float, sigma: float, amplitude: float
    ) -> np.ndarray:
        return amplitude * norm.pdf(x, mu, sigma)

    # Compute histogram and bin centers
    hist, bin_edges = np.histogram(distribution, bins="auto", density=True)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Fit the Gaussian curve to the histogram data
    popt, _ = curve_fit(
        _gaussian,
        bin_centers,
        hist,
        p0=[np.mean(distribution), np.std(distribution), np.max(hist)],
    )
    mu, sigma, amplitude = popt

    # Generate fitted Gaussian curve for plotting
    x = np.linspace(bin_edges[0], bin_edges[-1], 1000)
    fitted_curve = _gaussian(x, mu, sigma, amplitude)

    # Calculate PDF threshold-based cutoffs
    base_pdf = amplitude / (sigma * np.sqrt(2 * np.pi))
    x_threshold_min = mu - np.sqrt(-2 * sigma**2 * np.log(thr / base_pdf))
    x_threshold_max = mu + np.sqrt(-2 * sigma**2 * np.log(thr / base_pdf))

    # Identify outliers using numpy boolean indexing
    outliers: np.ndarray = distribution[
        (distribution < x_threshold_min) | (distribution > x_threshold_max)
    ]

    # Plot histogram, fitted curve, and threshold lines
    plt.hist(
        distribution,
        bins="auto",
        density=True,
        alpha=0.6,
        color="g",
        label="Histogram",
    )
    plt.plot(
        x,
        fitted_curve,
        "k-",
        linewidth=2,
        label=f"Gaussian fit  μ={mu:.2f}, σ={sigma:.2f}",
    )
    plt.axvline(
        x_threshold_min,
        color="r",
        linestyle="--",
        label=f"Threshold Min = {x_threshold_min:.2f}",
    )
    plt.axvline(
        x_threshold_max,
        color="b",
        linestyle="--",
        label=f"Threshold Max = {x_threshold_max:.2f}",
    )
    plt.legend(loc="best")
    plt.title("Histogram with Gaussian Fit and Thresholds")
    plt.xlabel("Values")
    plt.ylabel("Density")
    plt.tight_layout()
    plt.savefig(save_path / fig_name)
    plt.close()

    return outliers
