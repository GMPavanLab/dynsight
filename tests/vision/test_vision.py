from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterator

import numpy as np
import pytest
import yaml

from dynsight.vision import VisionInstance


class DummyTensor:
    def __init__(self, arr: np.ndarray[Any, Any]) -> None:
        self.arr = np.asarray(arr, dtype=float)

    def cpu(self) -> DummyTensor:
        return self

    def numpy(self) -> np.ndarray[Any, Any]:
        return np.asarray(self.arr)

    def __iter__(self) -> Iterator[float]:
        return iter(self.arr)


class DummyBoxes:
    def __init__(
        self,
        xyxy: np.ndarray[Any, Any],
        cls_ids: np.ndarray[Any, Any],
    ) -> None:
        self._xyxy = np.array(xyxy, dtype=float)
        self._cls = np.array(cls_ids, dtype=float)

    @property
    def xyxy(self) -> DummyTensor:
        return DummyTensor(self._xyxy)

    @property
    def cls(self) -> DummyTensor:
        return DummyTensor(self._cls)

    @property
    def xywhn(self) -> np.ndarray[Any, Any]:
        xywh = np.column_stack(
            [
                (self._xyxy[:, 0] + self._xyxy[:, 2]) / 2,
                (self._xyxy[:, 1] + self._xyxy[:, 3]) / 2,
                self._xyxy[:, 2] - self._xyxy[:, 0],
                self._xyxy[:, 3] - self._xyxy[:, 1],
            ]
        )
        return xywh / 100.0


class DummyResult:
    def __init__(self, path: Path, boxes: DummyBoxes) -> None:
        self.path = str(path)
        self.names = {0: "obj"}
        self.boxes = boxes


@pytest.fixture(autouse=True)
def patch_yolo(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "dynsight._internal.vision.vision.YOLO",
        lambda *_: SimpleNamespace(),
    )
    monkeypatch.setattr(VisionInstance, "_check_device", lambda _: None)

def test_set_training_dataset(tmp_path: Path) -> None:
    vi = VisionInstance(source="src", output_path=tmp_path)
    yaml_file = tmp_path / "data.yaml"
    vi.set_training_dataset(yaml_file)
    assert vi.training_data_yaml == yaml_file


def create_fake_results(tmp_path: Path) -> list[DummyResult]:
    files: list[Path] = []
    results: list[DummyResult] = []
    for i in range(3):
        img = tmp_path / f"img{i}.jpg"
        img.write_bytes(b"test")
        boxes = DummyBoxes(np.array([[0, 0, 10, 10]]), np.array([0]))
        results.append(DummyResult(img, boxes))
        files.append(img)
    return results


def test_create_dataset_from_predictions(tmp_path: Path) -> None:
    vi = VisionInstance(source="src", output_path=tmp_path)
    vi.prediction_results = create_fake_results(tmp_path)
    vi.create_dataset_from_predictions("dataset", train_split=0.5)
    dataset = tmp_path / "dataset"
    assert (dataset / "images/train/img0.jpg").exists()
    assert (dataset / "labels/train/img0.txt").exists()
    assert (dataset / "images/val/img1.jpg").exists()
    yaml_path = dataset / "dataset.yaml"
    data = yaml.safe_load(yaml_path.read_text())
    assert data["nc"] == 1
    assert data["names"] == ["obj"]
    assert vi.training_data_yaml == yaml_path


def test_export_prediction_to_xyz(tmp_path: Path) -> None:
    vi = VisionInstance(source="src", output_path=tmp_path)
    vi.prediction_results = create_fake_results(tmp_path)
    xyz_path = vi.export_prediction_to_xyz(Path("out.xyz"))
    lines = xyz_path.read_text().splitlines()
    lines_num = 9
    assert len(lines) == lines_num
    assert lines[0] == "1"
    assert lines[1] == "x y z"
    assert xyz_path.exists()
