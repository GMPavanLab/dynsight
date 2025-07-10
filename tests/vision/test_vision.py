from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import yaml
from PIL import Image

if TYPE_CHECKING:
    from pathlib import Path

from dynsight._internal.vision.vision import VisionInstance

DEFAULT_MODEL = "yolo12n.pt"


def create_dummy_yolo_dataset(
    root_path: Path,
    num_train: int = 5,
    num_val: int = 2,
    image_size: tuple[int, int] = (100, 100),
    num_classes: int = 1,
    class_names: list[str] | None = None,
    rng: np.random.Generator | None = None,
) -> None:
    if rng is None:
        rng = np.random.default_rng()
    if class_names is None:
        class_names = [f"class_{i}" for i in range(num_classes)]

    def generate_split(split_name: str, num_images: int) -> None:
        images_dir = root_path / "dataset" / "images" / split_name
        labels_dir = root_path / "dataset" / "labels" / split_name
        images_dir.mkdir(parents=True, exist_ok=True)
        labels_dir.mkdir(parents=True, exist_ok=True)

        for i in range(num_images):
            array = rng.integers(
                0, 256, size=(image_size[1], image_size[0], 3), dtype=np.uint8
            )
            img = Image.fromarray(array)
            img_path = images_dir / f"img_{i}.jpg"
            img.save(img_path)

            x_center = np.round(rng.uniform(0.3, 0.7), 6)
            y_center = np.round(rng.uniform(0.3, 0.7), 6)
            width = np.round(rng.uniform(0.1, 0.3), 6)
            height = np.round(rng.uniform(0.1, 0.3), 6)
            class_id = int(rng.integers(0, num_classes))

            label_path = labels_dir / f"img_{i}.txt"
            label_path.write_text(
                f"{class_id} {x_center} {y_center} {width} {height}\n"
            )

    generate_split("train", num_train)
    generate_split("val", num_val)

    data_yaml = {
        "path": str((root_path / "dataset").resolve()),
        "train": str((root_path / "dataset" / "images" / "train").resolve()),
        "val": str((root_path / "dataset" / "images" / "val").resolve()),
        "nc": num_classes,
        "names": class_names,
    }
    yaml_path = root_path / "data.yaml"
    yaml_path.write_text(yaml.dump(data_yaml))


def test_vision_instance_creation(tmp_path: Path) -> None:
    source_path = tmp_path / "source.jpg"
    img = Image.new("RGB", (100, 100))
    img.save(source_path)
    model_path = tmp_path / DEFAULT_MODEL
    out_path = tmp_path / "output"

    instance = VisionInstance(
        source=source_path,
        output_path=out_path,
        model=model_path,
        device="cpu",
        workers=1,
    )
    assert model_path.exists()
    assert instance.training_data_yaml is None
    assert instance.training_results is None
    assert instance.prediction_results is None
    assert instance.device == "cpu"


def test_vision_training(tmp_path: Path) -> None:
    source_path = tmp_path / "source.jpg"
    img = Image.new("RGB", (100, 100))
    img.save(source_path)
    model_path = tmp_path / DEFAULT_MODEL
    out_path = tmp_path / "output"

    instance = VisionInstance(
        source=source_path,
        output_path=out_path,
        model=model_path,
        device="cpu",
        workers=1,
    )
    old_model = instance.model
    create_dummy_yolo_dataset(tmp_path)
    instance.set_training_dataset(tmp_path / "data.yaml")
    assert (tmp_path / "data.yaml").exists()
    instance.train(
        title="test_train",
        epochs=1,
        batch_size=-1,
        imgsz=100,
    )
    new_model = instance.model
    new_model_path = out_path / "test_train" / "weights" / "best.pt"
    assert instance.training_results is not None

    assert str(instance.training_results.names[0]) == "class_0"
    assert str(instance.training_results.task) == "detect"

    assert new_model_path.exists()
    assert old_model != new_model


def test_vision_predict(tmp_path: Path) -> None:
    out_path = tmp_path / "output"

    source_path = tmp_path / "imgs"
    source_path.mkdir(parents=True, exist_ok=True)
    for i in range(10):
        img = Image.new("RGB", (100, 100))
        img.save(source_path / f"img_{i}.jpg")
    model_path = tmp_path / DEFAULT_MODEL

    instance = VisionInstance(
        source=source_path,
        output_path=out_path,
        model=model_path,
        device="cpu",
        workers=1,
    )
    instance.predict(prediction_title="test_predict")
    assert instance.prediction_results is not None
    for i in range(10):
        assert (source_path / f"img_{i}.jpg").exists()
    instance.create_dataset_from_predictions("test_dataset_from_pred")

    dataset_img_t = out_path / "test_dataset_from_pred" / "images" / "train"
    dataset_img_v = out_path / "test_dataset_from_pred" / "images" / "val"
    dataset_lab_t = out_path / "test_dataset_from_pred" / "labels" / "train"
    dataset_lab_v = out_path / "test_dataset_from_pred" / "labels" / "val"

    files_img_t = list(dataset_img_t.glob("*.jpg"))
    files_img_v = list(dataset_img_v.glob("*.jpg"))
    files_lab_t = list(dataset_lab_t.glob("*.txt"))
    files_lab_v = list(dataset_lab_v.glob("*.txt"))

    expected_train_set_len = 8
    expected_val_set_len = 2
    assert len(files_img_t) == expected_train_set_len
    assert len(files_img_v) == expected_val_set_len
    assert len(files_lab_t) == expected_train_set_len
    assert len(files_lab_v) == expected_val_set_len


def test_vision_tuning(tmp_path: Path) -> None:
    source_path = tmp_path / "source.jpg"
    img = Image.new("RGB", (100, 100))
    img.save(source_path)
    model_path = tmp_path / DEFAULT_MODEL
    out_path = tmp_path / "output"

    instance = VisionInstance(
        source=source_path,
        output_path=out_path,
        model=model_path,
        device="cpu",
        workers=1,
    )
    create_dummy_yolo_dataset(tmp_path)
    instance.set_training_dataset(tmp_path / "data.yaml")
    hyp = instance.tune_hyperparams(
        iterations=1,
        epochs=1,
        imgsz=100,
        batch_size=-1,
    )
    assert (
        out_path / "tuning" / "results" / "best_hyperparameters.yaml"
    ).exists()
    assert isinstance(hyp, dict)
