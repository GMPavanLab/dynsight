from __future__ import annotations

from typing import TYPE_CHECKING

from ultralytics import YOLO

if TYPE_CHECKING:
    from pathlib import Path

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


class VisionInstance:
    """Class for performing vision tasks using YOLO models.

    It allows to perform object detection, Convolutional Neural Network (CNN)
    training and tuning, training dataset creation and handling.

    .. caution::
        This class is still in development and may not work as expected.

    """
    def __init__(
        self, source: str | Path,
        output_path: Path,
        model: str | Path = "yolo12n.pt",
        device: str | None = None,
        workers: int = 8,
    ) -> None:
        """Initialize a Vision instance.

        Defines the starting model, the output folder and the devices that will
        be used for the computation.

        Parameters:
            source:
                The source of the images or videos to be processed. For the
                list of the possible sources, we refer the user to the
                following `table <https://docs.ultralytics.com/modes/predict/#inference-sources>`_.
                For the list of the supported formats see this second `table <https://docs.ultralytics.com/modes/predict/#images>`_.

            output_path:
                The path where save the output folder.

            model:
                The path to the YOLO model file. Defaults to "yolo12n.pt". see
                `here <https://docs.ultralytics.com/models/yolo12/>`_ for more
                information.

            device:
                Allows users to select between CPU, a specific GPU, or other
                compute devices for model execution (`cpu`, `cuda:0` or `0`).

            workers:
                Number of worker threads for data loading. Influences the speed
                of data preprocessing and feeding into the model, especially
                useful in multi-GPU setups. (only for training sessions).

        """
        self.output_path: Path = output_path
        self.training_data_yaml: Path | None = None

        self.model = YOLO(model)
        self.source = source
        self.device = device
        self.workers = workers

        self.opt_results: dict[str, float] | None = None
        self.prediction_results = None
        self.training_results = None

    def set_training_dataset(self, training_data_yaml: Path) -> None:
        """Set the training dataset for the model training.

        Training dataset are setted through a ``yaml`` file that should have
        thefollowing structure:

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
        """Detect objects in the source.

        Parameters:
            prediction_title:
                The name of the prediction.

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

    def tune_hyperparams(self,
        title:str,
        epochs:int=50,
        iterations:int=15,
    )-> dict[str, float]:
        """Tune hyperparameters for the model.

        temporary.
        """
        if self.training_data_yaml is None:
            msg = "Training dataset has not been set."
            raise ValueError(msg)

        self.opt_results = self.model.tune(
            data=self.training_data_yaml,
            epochs=epochs,
            iterations=iterations,
            project=self.output_path,
            name=title,
            device=self.device)

        return self.opt_results

    def train(self,
        title: str,
        hyperparams: dict[str, float] | None = None,
        epochs: int = 100,
        batch_size: int = 16,
        patience: int = 20,
        imgsz: int | tuple[int, int] = 640,
        ) -> None:
        """Train a custom model using a training dataset.

        This function trains a custom model using a training dataset. Dataset
        should be set before calling this function with the
        ``set_training_data`` method.

        Parameters:
            title:
                The name of the model.

            hyperparams:
                The dictionary that contains all the hyperparameters for the
                model. The following default ``dict`` is used if not provided:

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
                performance or use the ``tune`` method to automatically
                optimize hyperparameters.

            epochs:
                Total number of training epochs. Each epoch represents a full
                pass over the entire dataset.

            batch_size:
                Batch size, with three modes: set as an integer (batch=16),
                auto mode for 60% GPU memory utilization (batch=-1), or auto
                mode with specified utilization fraction (batch=0.70).

            patience:
                Number of epochs to wait without improvement in validation
                metrics before early stopping the training. Helps to prevent
                overfitting.

            imgsz:
                Defines the image size for inference. Can be a single integer
                for square resizing or a tuple. Proper sizing can improve
                detection accuracy and processingspeed.

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
            **full_params
        )
