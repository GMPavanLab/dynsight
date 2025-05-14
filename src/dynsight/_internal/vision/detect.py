from __future__ import annotations

import logging
import pathlib
import shutil
import tkinter as tk
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, TypedDict, cast

import numpy as np
import yaml
from PIL import Image

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None


from .vision_gui import VisionGUI
from .vision_utilities import find_outliers

if TYPE_CHECKING:
    from .video_to_frame import Video

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)


class YAMLConfig(TypedDict):
    """Class for YAML file configurations."""

    train: list[str]
    val: list[str]
    nc: int
    names: list[str]


class Detect:
    """A class to manage the full pipeline of object detection from videos.

    * Author: Simone Martino

    .. caution::
        This part of the code is still under development and may
        contain errors.

    """

    def __init__(
        self,
        input_video: Video,
        project_folder: pathlib.Path = Path("output_folder"),
    ) -> None:
        """Initialize a detection project.

        Creates all necessary subdirectories and extracts video frames if not
        already present.

        Parameters:
            input_video:
                The input video object from which to extract frames.

            project_folder:
                Path to the main project directory.
        """
        # Main directory for all the project outputs.
        self._project_folder = project_folder
        self._project_folder.mkdir(parents=True, exist_ok=True)

        # Directory where store extracted video frames.
        self._frame_path = self._project_folder / "frames"

        # Directory path where save the training item crops selected.
        self._training_items_path = self._project_folder / "training_items"

        # Directory path for the generated synthetic dataset.
        self._syn_dataset_path = self._project_folder / "synthetic_dataset"

        # Directory path where save trained models.
        self._models_path = self._project_folder / "models"

        # Directory path where save prediction outputs.
        self._predictions_path = self._project_folder / "predictions"

        # Path to the YAML configuration file for dataset training.
        self._yaml_file_path = self._project_folder / "training_options.yaml"

        # Resolution of the input video (width, height) in pixels.
        self._video_size = input_video.resolution()
        # Total number of frames extracted from the video.
        self._n_frames = input_video.count_frames()

        # Check if the video's frame are already present
        # if not -> extract them
        if not self._frame_path.exists():
            input_video.extract_frames(project_folder)

    def get_project_path(self) -> pathlib.Path:
        """It returns the path of the detection project."""
        return self._project_folder

    def synthesize(
        self,
        dataset_dimension: int = 1000,
        reference_img_path: None | pathlib.Path = None,
        validation_set_fraction: float = 0.2,
        collage_size: None | tuple[int, int] = None,
        collage_max_repeats: int = 30,
        sample_from: Literal["gui"] = "gui",
        random_seed: int | None = None,
    ) -> None:
        """Generate a synthetic dataset by creating collages of training items.

        Parameters:
            dataset_dimension:
                Total number of synthetic samples to generate.

            reference_img_path:
                Path to the reference image to initialize the GUI.
                If None, uses the first extracted frame.

            validation_set_fraction:
                Fraction of samples to allocate to the validation set.

            collage_size:
                Size (width, height) of the generated collage images.
                If None the input_video size will be taken.

            collage_max_repeats:
                Maximum number of training items that can be placed in
                a single collage.

            sample_from:
                Mode to collect training items. in the current version
                only the "gui" mode is available, other modes will come
                in the future.

            random_seed:
                Seed for shuffling the dataset.
        """
        if collage_size is None:
            collage_size = self._video_size
        # Dataset structure
        images_train_dir = self._syn_dataset_path / "images" / "train"
        images_val_dir = self._syn_dataset_path / "images" / "val"
        labels_train_dir = self._syn_dataset_path / "labels" / "train"
        labels_val_dir = self._syn_dataset_path / "labels" / "val"

        # Sampling method
        self._sample(
            sample_from=sample_from,
            reference_img_path=reference_img_path,
        )

        logger.info("Initializing the synthetic dataset")
        # Initialize a new dataset
        self._syn_dataset_path.mkdir(exist_ok=True)
        images_train_dir.mkdir(exist_ok=True, parents=True)
        images_val_dir.mkdir(exist_ok=True, parents=True)
        labels_train_dir.mkdir(exist_ok=True, parents=True)
        labels_val_dir.mkdir(exist_ok=True, parents=True)

        # Split between training and validation set
        num_val = int(dataset_dimension * validation_set_fraction)
        num_train = dataset_dimension - num_val
        remaining = dataset_dimension - (num_train + num_val)
        num_train += remaining

        assignments = ["train"] * num_train + ["val"] * num_val

        rng = np.random.default_rng(seed=random_seed)
        rng.shuffle(assignments)

        # Create synthetic images to fill the dataset
        # Create labels for each images generated
        logger.info("Generating synthetic images")
        for i in range(1, dataset_dimension + 1):
            collage, label_lines = self._create_collage(
                images_folder=self._training_items_path,
                width=collage_size[0],
                height=collage_size[1],
                max_repeats=collage_max_repeats,
                random_seed=random_seed,
            )
            subset = assignments[i - 1]
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

        logger.info("Generating yaml configuration file")
        # Generate the config file for the dataset created
        self._add_or_create_yaml(self._syn_dataset_path)

    # Just a bridge to the YOLO library
    def train(
        self,
        yaml_file: pathlib.Path,
        batch_size: int,
        workers: int,
        initial_model: str | pathlib.Path = "yolo12x.pt",
        training_name: str | None = None,
        training_epochs: int = 100,
        training_patience: int = 100,
        device: int | str | list[int] | None = None,
    ) -> None:
        """Train a YOLO model on the selected dataset.

        This function uses the
        `ultralytics YOLO library <https://github.com/ultralytics/ultralytics>`_
        for the model training.

        Parameters:
            yaml_file:
                Path to the dataset YAML configuration file.

            initial_model:
                Initial pretrained model to fine-tune.

            training_name:
                Name for the training run.

            training_epochs:
                Maximum number of training epochs for each training session.

            training_patience:
                Early stopping patience (number of epochs without improvement).

            batch_size:
                Batch size for training.

            workers:
                Number of dataloader worker threads.

            device:
                Device(s) on which run training.
        """
        model = YOLO(initial_model)
        model.train(
            data=yaml_file,
            epochs=training_epochs,
            patience=training_patience,
            batch=batch_size,
            imgsz=self._video_size,
            workers=workers,
            project=self._models_path,
            name=training_name,
            device=device,
            plots=False,
        )

    # Just a bridge to the YOLO library
    def predict_frames(
        self,
        model_path: str | pathlib.Path,
        detections_iou: float = 0.1,
        prediction_name: str = "prediction",
    ) -> None:
        """Perform object detection predictions on the extracted frames.

        This function uses the
        `ultralytics YOLO library <https://github.com/ultralytics/ultralytics>`_
        to detect objects in videos.

        Parameters:
            model_path:
                Path to the trained model.

            detections_iou:
                IOU threshold for object detection filtering.

            prediction_name:
                Name under which save the prediction results.
        """
        model = YOLO(model_path)
        for frame in range(self._n_frames):
            model.predict(
                project=self._project_folder,
                source=self._frame_path / f"{frame}.png",
                name=prediction_name,
                augment=True,
                line_width=2,
                save=True,
                show_labels=False,
                save_txt=True,
                save_conf=True,
                iou=detections_iou,
                max_det=20000,
                exist_ok=True,
                imgsz=self._video_size,
            )

    def fit(
        self,
        initial_dataset: pathlib.Path,
        max_sessions: int,
        training_epochs: int,
        training_patience: int,
        batch_size: int,
        workers: int,
        initial_model: str | pathlib.Path = "yolo12x.pt",
        device: int | str | list[int] | None = None,
        frame_reading_step: int = 1,
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

        This method uses the
        `ultralytics YOLO library <https://github.com/ultralytics/ultralytics>`_.

        Parameters:
            initial_dataset:
                Path to the initial dataset YAML file.

            initial_model:
                Path to the initial model weights (.pt file).

            max_sessions:
                Number of retraining cycles.

            training_epochs:
                Number of epochs per training session.

            training_patience:
                Early stopping patience during training.

            batch_size:
                Batch size for training.

            workers:
                Number of data loader workers.

            device:
                Device(s) to use (e.g., "cpu", "0", [0,1]).

            frame_reading_step:
                Specifies the interval at which frames are sampled from
                the video during processing.
        """
        # Initilize the first training
        current_dataset = initial_dataset
        guess_model_name = "v0"
        prediction_number = 0
        detection_results = []

        logger.info("First training begins.")
        self.train(
            yaml_file=current_dataset,
            initial_model=initial_model,
            training_epochs=2,
            batch_size=batch_size,
            workers=workers,
            device=device,
            training_name=guess_model_name,
        )

        current_model_path = (
            self._project_folder
            / "models"
            / guess_model_name
            / "weights"
            / "best.pt"
        )
        current_model = YOLO(current_model_path)
        logger.info(f"Starting prediction number {prediction_number}")
        for f in range(0, self._n_frames, frame_reading_step):
            frame_file = self._frame_path / f"{f}.png"
            prediction = current_model.predict(
                source=frame_file,
                imgsz=self._video_size,
                augment=True,
                save=True,
                save_txt=True,
                save_conf=True,
                show_labels=False,
                name=f"attempt_{prediction_number}",
                iou=0.1,
                max_det=20000,
                project=self._predictions_path,
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
        logger.info("Looking for outliers")
        # Filter detections
        detection_results = self._filter_detections(
            detection_results,
            prediction_number,
        )
        # Build a new dataset based on the "filtered" detection results
        # New dataset path
        train_dataset_path = (
            self._project_folder
            / "train_datasets"
            / f"dataset_{prediction_number}"
        )
        logger.info(f"Building the dataset (version {prediction_number})")
        # Build the dataset
        self._build_dataset(
            detection_results=detection_results,
            dataset_name=f"dataset_{prediction_number}",
        )
        # Add the new dataset to the training config file
        self._add_or_create_yaml(train_dataset_path)

        # Iterative part to improve the model performace
        for s in range(max_sessions):
            logger.info(f"Starting a new training session (number {s + 1}")
            new_model_name = f"v{s + 1}"
            self.train(
                yaml_file=self._yaml_file_path,
                initial_model=current_model_path,
                training_epochs=training_epochs,
                training_patience=training_patience,
                batch_size=batch_size,
                workers=workers,
                device=device,
                training_name=new_model_name,
            )
            current_model_path = (
                self._project_folder
                / "models"
                / new_model_name
                / "weights"
                / "best.pt"
            )
            current_model = YOLO(current_model_path)
            prediction_number += 1
            detection_results = []
            logger.info(f"Starting prediction number {prediction_number}")
            for f in range(0, self._n_frames, frame_reading_step):
                frame_file = self._frame_path / f"{f}.png"
                prediction = current_model.predict(
                    source=frame_file,
                    imgsz=self._video_size,
                    augment=True,
                    save=True,
                    save_txt=True,
                    save_conf=True,
                    show_labels=False,
                    name=f"attempt_{prediction_number}",
                    iou=0.1,
                    max_det=20000,
                    project=self._project_folder / "predictions",
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
            logger.info("Looking for outliers")
            # Filter detections
            detection_results = self._filter_detections(
                detection_results,
                prediction_number,
            )
            logger.info(f"Building the dataset (version {prediction_number}")
            # Build the new dataset
            self._build_dataset(
                detection_results=detection_results,
                dataset_name=f"dataset_{prediction_number}",
            )
            # Remove the oldest dataset in config
            # It has been made to avoid training bias on worst results
            self._remove_old_dataset()

            train_dataset_path = (
                self._project_folder
                / "train_datasets"
                / f"dataset_{prediction_number}"
            )
            # Update the dataset config file
            self._add_or_create_yaml(train_dataset_path)

    def compute_xyz(
        self,
        prediction_folder_path: pathlib.Path,
        output_path: pathlib.Path,
    ) -> None:
        """Computes and saves the trajectory of detections to xyz file.

        Parameters:
            prediction_folder_path:
                The path to the folder containing detection data files.
            output_path:
                The path where the resulting trajectory data should be saved.
        """
        lab_folder = prediction_folder_path / "labels"
        frame_positions = []
        for frame in range(self._n_frames):
            label_file = lab_folder / f"{frame}.txt"

            with label_file.open("r") as file:
                frame_detections = []
                for line in file:
                    values = line.strip().split()
                    _, x, y, width, height, confidence = map(float, values)
                    x *= self._video_size[0]
                    y *= self._video_size[1]
                    frame_detections.append((x, y))
                frame_positions.append(frame_detections)

        with output_path.open("w") as file:
            for frame_index, detections in enumerate(frame_positions):
                file.write(f"{len(detections)}\n")
                file.write(f"Frame {frame_index}\n")
                for x, y in detections:
                    z = 0
                    file.write(f"{x:.6f} {y:.6f} {z:.6f}\n")

    def _filter_detections(
        self,
        input_results: list[dict[str, int | float]],
        prediction_number: int,
    ) -> list[dict[str, int | float]]:
        # Initialize outliers folder in the prediction folder
        outliers_plt_folder = (
            self._project_folder
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
        try:
            out_width = set(
                find_outliers(
                    distribution=widths,
                    save_path=outliers_plt_folder,
                    fig_name="width",
                )
            )
        except (RuntimeError, ValueError) as e:
            logger.warning(
                "Outlier detection for width failed: %s. No width outliers",
                e,
            )
            out_width = set()

        try:
            out_height = set(
                find_outliers(
                    distribution=heights,
                    save_path=outliers_plt_folder,
                    fig_name="height",
                )
            )
        except (RuntimeError, ValueError) as e:
            logger.warning(
                "Outlier detection for height failed: %s. No height outliers.",
                e,
            )
            out_height = set()

        filtered_results = [
            det
            for det in input_results
            if (det["width"] not in out_width)
            and (det["height"] not in out_height)
        ]

        if not filtered_results:
            logger.warning("No outliers detected.")
            filtered_results = input_results

        return filtered_results

    def _remove_old_dataset(self) -> None:
        """Removes the oldest dataset from the YAML configuration."""
        yaml_path = self._yaml_file_path

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
        yaml_path = Path(self._yaml_file_path)

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
        output_dir = self._project_folder / "train_datasets" / dataset_name
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

            img_src = self._frame_path / f"{frame_idx}.png"
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
        random_seed: int | None = None,
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
        rng = np.random.default_rng(seed=random_seed)
        for _ in range(total_placement):
            cropped = rng.choice(cropped_images_array)
            w, h = cropped.size[:2]
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

    def _sample(
        self, sample_from: str, reference_img_path: pathlib.Path | None
    ) -> None:
        # GUI mode
        if sample_from == "gui":
            logger.info("Loading Graphic User Interface")
            # If not specified by the user use the first img as example
            if reference_img_path is None:
                logger.info("Using first frame as reference img.")
                reference_img_path = self._frame_path / "0.png"
            # Open the GUI
            root = tk.Tk()
            VisionGUI(
                master=root,
                image_path=reference_img_path,
                destination_folder=self._project_folder,
            )
            root.mainloop()
            # Check if the training items has been properly created
            if not (
                self._training_items_path.exists()
                and self._training_items_path.is_dir()
            ):
                msg = "'training_items' folder not created or not found"
                raise ValueError(msg)
        else:
            raise NotImplementedError
