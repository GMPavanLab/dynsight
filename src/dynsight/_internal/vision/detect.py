from __future__ import annotations

import pathlib
import random
import shutil
import tkinter as tk
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, TypedDict, cast

import matplotlib.pyplot as plt
import numpy as np
import yaml
from PIL import Image
from scipy.optimize import curve_fit
from scipy.stats import norm
from ultralytics import YOLO

from .vision_gui import VisionGUI

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from .video_to_frame import Video


class YAMLConfig(TypedDict):
    train: list[str]
    val: list[str]
    nc: int
    names: list[str]


class Detect:
    """A class to manage the full pipeline of object detection from videos."""

    def __init__(
        self,
        input_video: Video,
        project_folder: pathlib.Path = Path("output_folder"),
    ) -> None:
        """Initialize a detection project.

        Creates all necessary subdirectories and extracts video frames if not
        already present.

        Args:
            input_video: The input video object from which to extract frames.
            project_folder: Path to the main project directory.

        """
        # Define the project folder path
        self.project_folder = project_folder
        """Main directory for all the project outputs."""

        self.frames_dir = self.project_folder / "frames"
        """Directory where store extracted video frames."""

        self.training_items_path = self.project_folder / "training_items"
        """Directory path where save the training item crops selected."""

        self.syn_dataset_path = self.project_folder / "synthetic_dataset"
        """Directory path for the generated synthetic dataset."""

        self.models_path = self.project_folder / "models"
        """Directory path where save trained models."""

        self.predictions_path = self.project_folder / "predictions"
        """Directory path where save prediction outputs."""

        # Define the config file path for the training
        self.yaml_file = self.project_folder / "training_options.yaml"
        """Path to the YAML configuration file for dataset training."""

        # Extract information from the input video
        self.video_size = input_video.resolution()
        """Resolution of the input video (width, height) in pixels."""
        self.n_frames = input_video.count_frames()
        """Total number of frames extracted from the video."""

        # Check if the video's frame are already present
        # if not -> extract them
        if not (self.frames_dir.exists() and self.frames_dir.is_dir()):
            input_video.extract_frames(project_folder)

    def synthesize(
        self,
        dataset_dimension: int = 1000,
        reference_img_path: None | pathlib.Path = None,
        validation_set_fraction: float = 0.2,
        collage_size: tuple[int, int] = (1080, 1080),
        collage_max_repeats: int = 30,
        sample_from: Literal["gui"] = "gui",
    ) -> None:
        """Generate a synthetic dataset by creating collages of training items.

        Parameters:
            dataset_dimension: Total number of synthetic samples to generate.
            reference_img_path: Path to the reference image to initialize the
                GUI. If None, uses the first extracted frame.
            validation_set_fraction: Fraction of samples to allocate to the
                validation set.
            collage_size: Size (width, height) of the generated collage images.
            collage_max_repeats: Maximum number of training items that can be
                placed in a single collage.
            sample_from: Mode to collect training items. in the current version
                only the "gui" mode is available, other modes will come
                in the future.
        """
        # Dataset structure
        images_train_dir = self.syn_dataset_path / "images" / "train"
        images_val_dir = self.syn_dataset_path / "images" / "val"
        labels_train_dir = self.syn_dataset_path / "labels" / "train"
        labels_val_dir = self.syn_dataset_path / "labels" / "val"

        # GUI mode
        if sample_from == "gui":
            # If not specified by the user use the first img as example
            if reference_img_path is None:
                reference_img_path = self.frames_dir / "0.png"
            # Open the GUI
            root = tk.Tk()
            VisionGUI(
                master=root,
                image_path=reference_img_path,
                destination_folder=self.project_folder,
            )
            root.mainloop()
            # Check if the training items has been properly created
            if not (
                self.training_items_path.exists()
                and self.training_items_path.is_dir()
            ):
                msg = "'training_items' folder not created or not found"
                raise ValueError(msg)

        # Initialize a new dataset
        self.syn_dataset_path.mkdir(exist_ok=True)
        for d in [
            images_train_dir,
            images_val_dir,
            labels_train_dir,
            labels_val_dir,
        ]:
            d.mkdir(exist_ok=True, parents=True)

        # Split between training and validation set
        num_val = int(dataset_dimension * validation_set_fraction)
        num_train = dataset_dimension - num_val
        remaining = dataset_dimension - (num_train + num_val)
        num_train += remaining

        assignements = ["train"] * num_train + ["val"] * num_val
        random.shuffle(assignements)

        # Create synthetic images to fill the dataset
        # Create labels for each images generated
        for i in range(1, dataset_dimension + 1):
            collage, label_lines = self._create_collage(
                images_folder=self.training_items_path,
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

        # Generate the config file for the dataset created
        self._add_or_create_yaml(self.syn_dataset_path)

    # Just a bridge to the YOLO library
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
        """Train a YOLO model on the selected dataset.

        Parameters:
            yaml_file: Path to the dataset YAML configuration file.
            initial_model: Initial pretrained model to fine-tune.
            training_name: Name for the training run.
            training_epochs: Maximum number of training epochs for each
                training session.
            training_patience: Early stopping patience
                (number of epochs without improvement).
            batch_size: Batch size for training.
            workers: Number of dataloader worker threads.
            device: Device(s) on which run training.
        """
        model = YOLO(initial_model)
        model.train(
            data=yaml_file,
            epochs=training_epochs,
            patience=training_patience,
            batch=batch_size,
            imgsz=self.video_size,
            workers=workers,
            project=self.models_path,
            name=training_name,
            device=device,
            plots=False,
        )

    # Just a bridge to the YOLO library
    def predict(
        self,
        model_path: str | pathlib.Path,
        detections_iou: float = 0.1,
        prediction_name: str = "prediction",
    ) -> None:
        """Perform object detection predictions on the extracted frames.

        Parameters:
            model_path: Path to the trained model.
            detections_iou: IOU threshold for object detection filtering.
            prediction_name: Name under which save the prediction results.
        """
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
        max_sessions: int = 2,
        training_epochs: int = 2,
        training_patience: int = 2,
        batch_size: int = 16,
        workers: int = 8,
        device: int | str | list[int] | None = None,
    ) -> None:
        """Train an object detection model through iterative self-training.

        This method performs multiple rounds of training and prediction:
            1. Train the model on the initial dataset.
            2. Predict bounding boxes on video frames using the trained model.
            3. Identify and remove outlier detections based on box sizes.
            4. Build a new training dataset from filtered detections.
            5. Retrain the model on the refined dataset.
            6. Repeat steps 2 to 5 for a given number of sessions to
            progressively refine the model.

        Args:
            initial_dataset: Path to the initial dataset YAML file.
            initial_model: Path to the initial model weights (.pt file).
            max_sessions: Number of retraining cycles.
            training_epochs: Number of epochs per training session.
            training_patience: Early stopping patience during training.
            batch_size: Batch size for training.
            workers: Number of data loader workers.
            device: Device(s) to use (e.g., "cpu", "0", [0,1]).
            real_n_particles: Optional, real number of particles expected.
        """
        # Initilize the first training
        current_dataset = initial_dataset
        guess_model_name = "v0"
        # Initialize the first prediction
        prediction_number = 0
        detection_results = []

        # First training
        self.train(
            yaml_file=current_dataset,
            initial_model=initial_model,
            training_epochs=2,
            batch_size=batch_size,
            workers=workers,
            device=device,
            training_name=guess_model_name,
        )
        # Update the model with the new one
        current_model_path = (
            self.project_folder
            / "models"
            / guess_model_name
            / "weights"
            / "best.pt"
        )
        current_model = YOLO(current_model_path)
        # First prediction
        for f in range(0, self.n_frames, 50):  # TEMP
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
                project=self.predictions_path,
                line_width=2,
                exist_ok=True,
            )
            # Read and save the prediction results
            if prediction and prediction[0].boxes:
                xywh = prediction[0].boxes.xywh.cpu().numpy()
                conf = prediction[0].boxes.conf.cpu().numpy()
                cls = prediction[0].boxes.cls.cpu().numpy()
                n_detection = len(xywh)

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
        # Filter detections
        detection_results = self._filter_detections(
            detection_results,
            prediction_number,
        )
        # Build a new dataset based on the "filtered" detection results
        # New dataset path
        train_dataset_path = (
            self.project_folder
            / "train_datasets"
            / f"dataset_{prediction_number}"
        )

        # Build the dataset
        self._build_dataset(
            detection_results=detection_results,
            dataset_name=f"dataset_{prediction_number}",
        )
        # Add the new dataset to the training config file
        self._add_or_create_yaml(train_dataset_path)

        # Iterative part to improve the model performace
        for s in range(max_sessions):
            new_model_name = f"v{s + 1}"
            self.train(
                yaml_file=self.yaml_file,
                initial_model=current_model_path,
                training_epochs=training_epochs,
                training_patience=training_patience,
                batch_size=batch_size,
                workers=workers,
                device=device,
                training_name=new_model_name,
            )
            current_model_path = (
                self.project_folder
                / "models"
                / new_model_name
                / "weights"
                / "best.pt"
            )
            current_model = YOLO(current_model_path)
            prediction_number += 1
            detection_results = []
            for f in range(0, self.n_frames, 50):  # TEMP
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
                # Read prediction
                if prediction and prediction[0].boxes:
                    xywh = prediction[0].boxes.xywh.cpu().numpy()
                    conf = prediction[0].boxes.conf.cpu().numpy()
                    cls = prediction[0].boxes.cls.cpu().numpy()
                    n_detection = len(xywh)

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

            # Filter detections
            detection_results = self._filter_detections(
                detection_results,
                prediction_number,
            )
            # Build the new dataset
            self._build_dataset(
                detection_results=detection_results,
                dataset_name=f"dataset_{prediction_number}",
            )
            # Remove the oldest dataset in config
            # It has been made to avoid training bias on worst results
            self._remove_old_dataset()

            train_dataset_path = (
                self.project_folder
                / "train_datasets"
                / f"dataset_{prediction_number}"
            )
            # Update the dataset config file
            self._add_or_create_yaml(train_dataset_path)

    def _filter_detections(
        self,
        input_results: list[dict[str, int | float]],
        prediction_number: int,
    ) -> list[dict[str, int | float]]:
        # Initialize outliers folder in the prediction folder
        outliers_plt_folder = (
            self.project_folder
            / "predictions"
            / f"attempt_{prediction_number}"
            / "outliers"
        )
        outliers_plt_folder.mkdir(exist_ok=True)

        # Look for outliers in the boxes width and height
        widths = np.array([d["width"] for d in input_results], dtype=float)
        heights = np.array(
            [d["height"] for d in input_results],
            dtype=float,
        )
        out_width = set(
            _find_outliers(
                distribution=widths,
                save_path=outliers_plt_folder,
                fig_name="width",
            )
        )
        out_height = set(
            _find_outliers(
                distribution=heights,
                save_path=outliers_plt_folder,
                fig_name="height",
            )
        )
        # Exclude the outliers from the detection results
        return [
            det
            for det in input_results
            if (det["width"] not in out_width)
            and (det["height"] not in out_height)
        ]

    def _remove_old_dataset(self) -> None:
        """Removes the oldest dataset from the YAML configuration."""
        yaml_path = self.yaml_file

        if not yaml_path.exists():
            return

        with yaml_path.open("r") as f:
            cfg = yaml.safe_load(f) or {}

        for key in ("train", "val"):
            if key not in cfg:
                return
            if isinstance(cfg[key], str):
                cfg[key] = [cfg[key]]
            elif not isinstance(cfg[key], list):
                return

        for key in ("train", "val"):
            if cfg[key]:
                cfg[key].pop(0)

        with yaml_path.open("w") as f:
            yaml.safe_dump(cfg, f, sort_keys=False)

    def _add_or_create_yaml(self, new_dataset_path: Path) -> None:
        yaml_path = Path(self.yaml_file)

        train_p = str((new_dataset_path / "images/train").resolve())
        val_p = str((new_dataset_path / "images/val").resolve())

        if not yaml_path.exists():
            cfg: YAMLConfig = {
                "train": [train_p],
                "val": [val_p],
                "nc": 1,
                "names": ["obj"],
            }
        else:
            raw = yaml.safe_load(yaml_path.open("r")) or {}
            cfg = cast("YAMLConfig", raw)

            if not isinstance(cfg.get("train"), list):
                cfg["train"] = [str(cfg.get("train") or train_p)]
            if not isinstance(cfg.get("val"), list):
                cfg["val"] = [str(cfg.get("val") or val_p)]

            cfg["train"].append(train_p)
            cfg["val"].append(val_p)
            cfg["train"] = list(dict.fromkeys(cfg["train"]))
            cfg["val"] = list(dict.fromkeys(cfg["val"]))
            # cfg.pop("path", None) # noqa: ERA001

            if isinstance(cfg.get("nc"), str):
                cfg["nc"] = int(cfg["nc"])

        with yaml_path.open("w") as f:
            yaml.safe_dump(cfg, f, sort_keys=False)

    def _build_dataset(
        self,
        detection_results: list[dict[str, Any]],
        dataset_name: str,
        split_ratio: float = 0.8,
    ) -> None:
        """Builds a dataset by splitting frames/labels into train and val."""
        output_dir = self.project_folder / "train_datasets" / dataset_name
        imgs_train_dir = output_dir / "images" / "train"
        imgs_val_dir = output_dir / "images" / "val"
        labs_train_dir = output_dir / "labels" / "train"
        labs_val_dir = output_dir / "labels" / "val"
        for d in (imgs_train_dir, imgs_val_dir, labs_train_dir, labs_val_dir):
            d.mkdir(parents=True, exist_ok=True)

        detections_by_frame: dict[int, list[dict[str, Any]]] = {}
        for det in detection_results:
            frame_idx = det["frame"]
            detections_by_frame.setdefault(frame_idx, []).append(det)

        all_frames = sorted(detections_by_frame.keys())

        split_point = int(len(all_frames) * split_ratio)
        train_frames = set(all_frames[:split_point])

        for frame_idx, dets in detections_by_frame.items():
            if frame_idx in train_frames:
                img_dest = imgs_train_dir
                lab_dest = labs_train_dir
            else:
                img_dest = imgs_val_dir
                lab_dest = labs_val_dir

            img_src = self.frames_dir / f"{frame_idx}.png"
            img_dst = img_dest / f"{frame_idx}.png"
            shutil.copy(img_src, img_dst)

            lab_file = lab_dest / f"{frame_idx}.txt"
            with lab_file.open("w") as f:
                img_w, img_h = Image.open(img_src).size

                for det in dets:
                    x_ctr = det["center_x"]
                    y_ctr = det["center_y"]
                    w = det["width"]
                    h = det["height"]
                    cls = det["class_id"]
                    x_ctr_n = x_ctr / img_w
                    y_ctr_n = y_ctr / img_h
                    w_n = w / img_w
                    h_n = h / img_h
                    f.write(
                        f"{cls} {x_ctr_n:.6f} {y_ctr_n:.6f} "
                        f"{w_n:.6f} {h_n:.6f}\n"
                    )

    def _create_collage(
        self,
        images_folder: pathlib.Path,
        width: int,
        height: int,
        patience: int = 1000,
        max_repeats: int = 1,
    ) -> tuple[Image.Image, list[str]]:
        """Creates a collage of images by placing them randomly on a canvas."""
        collage = Image.new("RGBA", (width, height), (255, 255, 255, 255))
        placed_rects: list[tuple[int, int, int, int]] = []
        label_lines = []
        cropped_images = []

        for file in images_folder.iterdir():
            if file.suffix.lower() in {".png", ".jpg", ".jpeg"}:
                img = Image.open(file).convert("RGBA")
                cropped_images.append(img)

        if not cropped_images:
            msg = "No images found in the specified folder"
            raise ValueError(msg)

        total_placement = len(cropped_images) * max_repeats
        placed_count = 0
        cropped_images_array = np.array(cropped_images, dtype=object)
        rng = np.random.default_rng()
        while placed_count < total_placement:
            cropped = rng.choice(cropped_images_array)
            w, h = cropped.size
            max_x = width - w
            max_y = height - h
            placed = False

            for _ in range(patience):
                x = rng.integers(0, max_x + 1)
                y = rng.integers(0, max_y + 1)
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

                    center_x = (x + w / 2) / width
                    center_y = (y + h / 2) / height
                    width_norm = w / width
                    height_norm = h / height
                    label_line = (
                        f"0 {center_x:.6f} "
                        f"{center_y:.6f} "
                        f"{width_norm:.6f} "
                        f"{height_norm:.6f}"
                    )
                    label_lines.append(label_line)
                    placed_count += 1
                    placed = True
                    break

            if not placed:
                break

        return collage, label_lines


def _find_outliers(
    distribution: NDArray[np.float64],
    save_path: pathlib.Path,
    fig_name: str,
    thr: float = 1e-5,
) -> NDArray[np.float64]:
    """Detects outliers in a distribution by fitting a normal distribution."""

    def _gaussian(
        x: NDArray[np.float64], mu: float, sigma: float, amplitude: float
    ) -> NDArray[np.float64]:
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
    outliers: NDArray[np.float64] = distribution[
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
        label=rf"Gaussian fit  $\mu={mu:.2f},\ \sigma={sigma:.2f}$",
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
