"""dynsight.vision module for particle detection from media files."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Callable

import numpy as np
import torch
import yaml
from PIL import Image
from ultralytics import YOLO

if TYPE_CHECKING:
    from ultralytics.engine.results import Results
    from ultralytics.utils.metrics import DetMetrics

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)

logger = logging.getLogger(__name__)

# Defaults hyperparameters dictionary.
default_hyperparams = {
    "lr0": 0.01,
    "lrf": 0.01,
    "momentum": 0.937,
    "weight_decay": 0.0005,
    "warmup_epochs": 3.0,
    "warmup_momentum": 0.8,
    "box": 7.5,
    "cls": 0.5,
    "dfl": 1.5,
    "hsv_h": 0.015,
    "hsv_s": 0.7,
    "hsv_v": 0.4,
    "degrees": 0.0,
    "translate": 0.1,
    "scale": 0.5,
    "shear": 0.0,
    "perspective": 0.0,
    "flipud": 0.0,
    "fliplr": 0.5,
    "bgr": 0.0,
    "mosaic": 1,
    "mixup": 0.0,
    "cutmix": 0.0,
    "copy_paste": 0.0,
}


class VisionInstance:
    def __init__(
        self,
        source: str | Path,
        output_path: Path,
        model: str | Path = "yolo12n.pt",
        device: str | None = None,
        workers: int = 8,
    ) -> None:
        """Class for performing computer vision tasks using YOLO models.

        This class supports object detection, Convolutional Neural Network
        (CNN) training and fine-tuning, as well as the creation and management
        of training datasets.

        .. caution::
            This class is still under development and may not function as
            intended.

        Parameters:
            source:
                The source of the images or videos to be processed. For the
                list of the possible sources, we refer the user to the
                following `sources table <https://docs.ultralytics.com/modes/predict/#inference-sources>`_.
                For the list of the supported formats see this `formats table <https://docs.ultralytics.com/modes/predict/#images>`_.

            output_path:
                The path to save the output folder.

            model:
                The path to the YOLO model file. Defaults to "yolo12n.pt". See
                `here <https://docs.ultralytics.com/models/yolo12/>`_ for more
                information.

            device:
                Allows users to select between cpu, a specific gpu ID or
                "mps" for MacOS users to perform the calculation
                ("cuda:0" or "0" for GPUs, "cpu" or "mps" for MacOS).

            workers:
                Number of worker threads for data loading. Influences the speed
                of data preprocessing and feeding into the model, especially
                useful in multi-GPU setups. (only for training sessions).

        """
        self.output_path = Path(output_path)
        self.training_data_yaml: Path | None = None

        self.model = YOLO(model)
        self.source = source
        self.device = self._normalize_device_string(device)
        self.workers = workers

        self.prediction_results: list[Results] | None = None
        self.training_results: DetMetrics | None = None

        self._check_device()

    def set_training_dataset(self, training_data_yaml: Path) -> None:
        """Set the training dataset for the model training.

        Training dataset are setted through a ``yaml`` file that should have
        the following structure:

        .. code-block:: yaml

            path: path/to/dataset/folder
            train: path/to/train/images
            val: path/to/val/images

            nc: number_of_classes
            names: [class1, class2, ...]

        With a dataset folder structure like this:

        .. code-block:: none

            dataset/
            ├── images/
            │   ├── train/
            │   │   ├── 1.jpg
            │   │   ├── 2.jpg
            │   │   └── ...
            │   └── val/
            │       ├── 5.jpg
            │       ├── 6.jpg
            │       └── ...
            └── labels/
                ├── train/
                │   ├── 1.txt
                │   ├── 2.txt
                │   └── ...
                └── val/
                    ├── 5.txt
                    ├── 6.txt
                    └── ...


        Parameters:
            training_data_yaml:
                Path to the training data YAML file.
        """
        self.training_data_yaml = training_data_yaml

    def predict(
        self,
        prediction_title: str,
        augment: bool = False,
        agnostic_nms: bool = False,
        show_labels: bool = False,
        class_filter: list[int] | None = None,
        confidence: float = 0.25,
        iou: float = 0.7,
        imgsz: int | tuple[int, int] = 640,
        max_det: int = 500,
    ) -> None:
        """Detect objects within the source.

        Parameters:
            prediction_title:
                The name of the prediction session.

            augment:
                Enables test-time augmentation (TTA) for predictions,
                potentially improving detection robustness at the cost of
                inference speed.

            agnostic_nms:
                Enables class-agnostic Non-Maximum Suppression (NMS), which
                merges overlapping boxes of different classes. Useful in
                multi-class detection scenarios where class overlap is common.

            show_labels:
                Show labels names in the detected source version.

            class_filter:
                Filters predictions to a set of class IDs. Only detections
                belonging to the specified classes will be returned.

            confidence:
                Sets the minimum confidence threshold for detections.
                Objects detected with confidence below this threshold will
                be disregarded.

            iou:
                Lower values result in fewer detections by eliminating
                overlapping boxes, useful for reducing duplicates.

            imgsz:
                Defines the image size for inference. Can be a single integer
                for square resizing or a tuple. Proper sizing can improve
                detection accuracy and processingspeed.

            max_det:
                The maximum number of detections for a single frame / image.

        """
        self.prediction_results = self.model.predict(
            source=self.source,
            save=True,
            save_txt=False,
            save_conf=True,
            show_labels=show_labels,
            name=prediction_title,
            project=self.output_path,
            device=self.device,
            augment=augment,
            agnostic_nms=agnostic_nms,
            classes=class_filter,
            conf=confidence,
            iou=iou,
            imgsz=imgsz,
            max_det=max_det,
        )

    def create_dataset_from_predictions(
        self,
        dataset_name: str,
        train_split: float = 0.8,
        load_dataset: bool = True,
    ) -> None:
        """Create a YOLO training dataset from ``predict`` results.

        Parameters:
            dataset_name:
                Name of the dataset that will be created.

            train_split:
                Fraction of images to be used as training set, the remaining
                fraction will be used for the validation set.

            load_dataset:
                Directly load the dataset for the next training sessions.
        """
        if self.prediction_results is None:
            msg = "No prediction results available."
            raise ValueError(msg)

        dataset_path = self.output_path / dataset_name
        images_train = dataset_path / "images" / "train"
        images_val = dataset_path / "images" / "val"
        labels_train = dataset_path / "labels" / "train"
        labels_val = dataset_path / "labels" / "val"

        images_train.mkdir(parents=True, exist_ok=True)
        images_val.mkdir(parents=True, exist_ok=True)
        labels_train.mkdir(parents=True, exist_ok=True)
        labels_val.mkdir(parents=True, exist_ok=True)

        names = self.prediction_results[0].names

        sorted_results = sorted(self.prediction_results, key=lambda r: r.path)

        num_train = int(len(sorted_results) * train_split)

        video_exts = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
        is_video = False
        if (
            isinstance(self.source, (str, Path))
            and Path(self.source).suffix.lower() in video_exts
        ):
            is_video = True

        for idx, result in enumerate(sorted_results):
            src = Path(result.path)
            subset = "train" if idx < num_train else "val"
            if is_video:
                frame_name = f"{src.stem}_{idx:06d}.jpg"
                img_dst = dataset_path / "images" / subset / frame_name
                lbl_dst = (
                    dataset_path
                    / "labels"
                    / subset
                    / (Path(frame_name).stem + ".txt")
                )

                img = Image.fromarray(result.orig_img[..., ::-1])
                img.save(img_dst)
            else:
                img_dst = dataset_path / "images" / subset / src.name
                lbl_dst = (
                    dataset_path / "labels" / subset / (src.stem + ".txt")
                )

                img_dst.write_bytes(src.read_bytes())

            boxes = result.boxes
            if boxes is None:
                lbl_dst.write_text("")
                continue

            xywhn = boxes.xywhn
            classes = boxes.cls
            with lbl_dst.open("w") as f:
                for xywh, cls in zip(xywhn, classes):
                    f.write(
                        f"{int(cls)} {xywh[0]:.6f} {xywh[1]:.6f} "
                        f"{xywh[2]:.6f} {xywh[3]:.6f}\n"
                    )

        dataset_yaml = dataset_path / "dataset.yaml"
        yaml_content = {
            "path": str(dataset_path.resolve()),
            "train": "images/train",
            "val": "images/val",
            "nc": len(names),
            "names": [names[i] for i in range(len(names))],
        }
        with dataset_yaml.open("w") as f:
            yaml.safe_dump(yaml_content, f)

        if load_dataset:
            self.training_data_yaml = dataset_yaml

    def tune_hyperparams(
        self,
        iterations: int = 15,
        epochs: int = 50,
        imgsz: int | tuple[int, int] = 640,
        batch_size: int = 16,
    ) -> dict[str, float]:
        """Tune hyperparameters for the model.

        Optimize the CNN hyperparameters by leveraging the Ultralytics YOLO
        `genetic algorithm <https://docs.ultralytics.com/guides/hyperparameter-tuning/>`_.
        It returns a dictionary of the best hyperparameters, which can be
        directly used as input to the hyperparameters parameter in the train
        method.

        Parameters:
            iterations:
                The number of exploring iterations. The higher the number, the
                more accurate the results will be, increasing the computational
                cost.

            epochs:
                The number of epochs to perform for each iteration. Each epoch
                represents a full pass over the entire dataset.

            imgsz:
                Defines the image size for inference. Can be a single integer
                for square resizing or a tuple. Proper sizing can improve
                detection accuracy and processing speed.

            batch_size:
                Three modes available: set as an integer (batch=16),
                auto mode for 60% GPU memory utilization (batch=-1), or auto
                mode with specified utilization fraction (batch=0.70).
        """
        if self.training_data_yaml is None:
            msg = "Training dataset has not been set."
            raise ValueError(msg)

        self.model.tune(
            data=self.training_data_yaml,
            epochs=epochs,
            iterations=iterations,
            project=self.output_path / "tuning",
            name="results",
            device=self.device,
            imgsz=imgsz,
            batch=batch_size,
        )
        yaml_path = (
            self.output_path
            / "tuning"
            / "results"
            / "best_hyperparameters.yaml"
        )
        with yaml_path.open("r") as f:
            return yaml.safe_load(f)

    def train(
        self,
        title: str,
        hyperparams: dict[str, float] | None = None,
        epochs: int = 100,
        batch_size: int = 16,
        patience: int = 20,
        imgsz: int | tuple[int, int] = 640,
    ) -> None:
        """Train a custom model using a training dataset.

        This function trains a custom model using a training dataset. The
        dataset should be set before calling this function with the
        ``set_training_data`` method.

        Parameters:
            title:
                The name of the resulting model.

            hyperparams:
                The dictionary that contains all the hyperparameters for the
                model training. The following default ``dict`` is used if not
                provided:

                .. code-block:: python

                    # Defaults hyperparameters dictionary.
                    default_hyperparams = {
                        "lr0": 0.01,
                        "lrf": 0.01,
                        "momentum": 0.937,
                        "weight_decay": 0.0005,
                        "warmup_epochs": 3.0,
                        "warmup_momentum": 0.8,
                        "box": 7.5,
                        "cls": 0.5,
                        "dfl": 1.5,
                        "hsv_h": 0.015,
                        "hsv_s": 0.7,
                        "hsv_v": 0.4,
                        "degrees": 0.0,
                        "translate": 0.1,
                        "scale": 0.5,
                        "shear": 0.0,
                        "perspective": 0.0,
                        "flipud": 0.0,
                        "fliplr": 0.5,
                        "bgr": 0.0,
                        "mosaic": 1,
                        "mixup": 0.0,
                        "cutmix": 0.0,
                        "copy_paste": 0.0
                    }

                Manually customize this ``dict`` to change the training
                performance or use the ``tune_hyperparams`` method to
                automatically optimize hyperparameters.

            epochs:
                Total number of training epochs. Each epoch represents a full
                pass over the entire dataset.

            batch_size:
                Three modes available: set as an integer (batch=16),
                auto mode for 60% GPU memory utilization (batch=-1), or auto
                mode with specified utilization fraction (batch=0.70).

            patience:
                Number of epochs to wait without improvement in validation
                metrics before early stopping the training. Helps to prevent
                overfitting.

            imgsz:
                Defines the image size for inference. Can be a single integer
                for square resizing or a tuple. Proper sizing can improve
                detection accuracy and processing speed.

        """
        if self.training_data_yaml is None:
            msg = "Training dataset has not been set."
            raise ValueError(msg)

        full_params = default_hyperparams.copy()
        if hyperparams is not None:
            unknown_keys = set(hyperparams.keys()) - set(full_params.keys())
            if unknown_keys:
                msg = f"Unknown hyperparameters: {unknown_keys}"
                raise ValueError(msg)
            for key in hyperparams:
                full_params[key] = hyperparams[key]

        self.training_results = self.model.train(
            data=self.training_data_yaml,
            epochs=epochs,
            imgsz=imgsz,
            batch=batch_size,
            workers=self.workers,
            name=title,
            project=self.output_path,
            patience=patience,
            device=self.device,
            **full_params,
        )
        self.model = YOLO(self.output_path / title / "weights" / "best.pt")

    def export_prediction_to_xyz(
        self, file_name: Path, class_filter: list[int] | None = None
    ) -> Path:
        """Export prediction results into a single ``.xyz`` file.

        Each frame of the resulting ``.xyz`` corresponds to one of the
        images/frames present in the source and used in the ``predict`` method.

        Parameters:
            file_name:
                File name for the ``.xyz`` file.

            class_filter:
                Limit exported detections to the specified class IDs. If
                ``None`` all detected objects will be exported.

        Returns:
            Path to the exported ``.xyz`` file.
        """
        if self.prediction_results is None:
            msg = "No prediction results available."
            raise ValueError(msg)

        sorted_results = sorted(self.prediction_results, key=lambda r: r.path)
        file_path = self.output_path / file_name

        with file_path.open("w") as f:
            for result in sorted_results:
                boxes = result.boxes

                coords: list[str] = []
                if boxes is not None:
                    xyxy_raw = boxes.xyxy
                    if isinstance(xyxy_raw, torch.Tensor):
                        xyxy = xyxy_raw.cpu().numpy()
                    else:
                        xyxy = np.asarray(xyxy_raw)

                    cls_raw = boxes.cls
                    if isinstance(cls_raw, torch.Tensor):
                        classes = cls_raw.cpu().numpy().astype(int)
                    else:
                        classes = np.asarray(cls_raw).astype(int)
                    for (x1, y1, x2, y2), cls_id in zip(xyxy, classes):
                        if (
                            class_filter is not None
                            and cls_id not in class_filter
                        ):
                            continue
                        cx = (x1 + x2) / 2
                        cy = (y1 + y2) / 2
                        coords.append(f"{cls_id} {cx:.6f} {cy:.6f} 0.0")

                f.write(f"{len(coords)}\n")
                f.write("class x y z\n")
                for line in coords:
                    f.write(f"{line}\n")
        return file_path

    def _normalize_device_string(self, device: str | None) -> str:
        """Normalize device string to match Ultralytics expectations."""
        if device is None:
            return "0" if torch.cuda.is_available() else "cpu"

        device = str(device).lower()

        if device in {"cpu", "mps", "cuda"}:
            return device

        # Allow "cuda:0" -> "0", "cuda:0,1" -> "0,1"
        if device.startswith("cuda:"):
            return device.replace("cuda:", "")

        # Allow "0", "0,1", etc.
        if all(part.strip().isdigit() for part in device.split(",")):
            return device
        msg = f"Unsupported device string: '{device}'"
        raise ValueError(msg)

    def _check_device(self) -> None:
        """Verify and validate the selected device for compatibility."""
        self.device = self._normalize_device_string(self.device)

        def _device_error(msg: str) -> None:
            raise RuntimeError(msg)

        try:
            if self.device == "cpu":
                self._check_cpu_device()
            elif self.device == "mps":
                self._check_mps_device(_device_error)
            elif self.device == "cuda":
                self._check_single_cuda_device(_device_error)
            elif all(
                part.strip().isdigit() for part in self.device.split(",")
            ):
                self._check_multi_cuda_devices(_device_error)
            else:
                _device_error(f"Unsupported device string: '{self.device}'")
        except (ValueError, RuntimeError, IndexError, OSError) as e:
            _device_error(str(e))

    def _check_cpu_device(self) -> None:
        logger.info("Using CPU.")

    def _check_mps_device(self, _device_error: Callable[[str], None]) -> None:
        if not (
            hasattr(torch.backends, "mps")
            and torch.backends.mps.is_available()
        ):
            _device_error("MPS device requested but not available.")
        logger.info("Using Apple MPS backend.")

    def _check_single_cuda_device(
        self, _device_error: Callable[[str], None]
    ) -> None:
        if not torch.cuda.is_available():
            _device_error("CUDA requested but not available.")
        name = torch.cuda.get_device_name(0)
        backend = "ROCm" if torch.version.hip else "CUDA"
        mem_free, mem_total = torch.cuda.mem_get_info(0)
        logger.info(f"Using GPU 0: {name} [{backend}]")
        logger.info(
            "Memory: %.1f MB free / %.1f MB total",
            mem_free / 1024**2,
            mem_total / 1024**2,
        )
        _ = torch.tensor([0.0]).to("cuda:0")

    def _check_multi_cuda_devices(
        self, _device_error: Callable[[str], None]
    ) -> None:
        gpus = [int(d) for d in self.device.split(",")]
        for idx in gpus:
            if idx >= torch.cuda.device_count():
                _device_error(
                    f"Requested GPU index {idx}, but only "
                    f"{torch.cuda.device_count()} available."
                )
            name = torch.cuda.get_device_name(idx)
            mem_free, mem_total = torch.cuda.mem_get_info(idx)
            logger.info(f"Using GPU {idx}: {name}")
            logger.info(
                "Memory: %.1f MB free / %.1f MB total",
                mem_free / 1024**2,
                mem_total / 1024**2,
            )
        _ = torch.tensor([0.0]).to(f"cuda:{gpus[0]}")
