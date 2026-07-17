from __future__ import annotations

import io
import json
import threading
import urllib.error
import urllib.request
from typing import TYPE_CHECKING, Any

import pytest
import yaml
from PIL import Image

from dynsight._internal.vision.label_tool import (
    _LabelToolServer,
    _safe_name,
    _split_count,
    _Workspace,
    export_dataset,
    load_session_file,
    save_session_file,
    synthesize_dataset,
)

if TYPE_CHECKING:
    from pathlib import Path


def make_image_bytes(
    width: int = 64, height: int = 48, color: str = "red"
) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (width, height), color).save(buffer, format="PNG")
    return buffer.getvalue()


@pytest.fixture
def workspace(tmp_path: Path) -> _Workspace:
    ws = _Workspace(tmp_path / "ws")
    for i in range(4):
        ws.add_image(f"img_{i}.png", make_image_bytes())
    return ws


def make_session() -> dict[str, Any]:
    return {
        "labels": [
            {"name": "particle", "color": "#ff0000"},
            {"name": "aggregate", "color": "#00ff00"},
        ],
        "annotations": {
            "img_0.png": [
                {"label": "aggregate", "x": 4, "y": 6, "w": 20, "h": 10},
                {"label": "particle", "x": 30, "y": 20, "w": 10, "h": 12},
            ],
            "img_1.png": [
                {"label": "particle", "x": 0, "y": 0, "w": 8, "h": 8},
            ],
        },
    }


def test_safe_name_blocks_traversal() -> None:
    assert _safe_name("../../etc/secret.png") == "secret.png"
    assert _safe_name("a b/c?.png") == "c_.png"
    with pytest.raises(ValueError, match="Invalid file name"):
        _safe_name("...")


def test_split_count_keeps_val_nonempty() -> None:
    assert _split_count(10, 0.8) == 8  # noqa: PLR2004
    assert _split_count(2, 0.99) == 1
    assert _split_count(2, 0.01) == 1
    assert _split_count(1, 0.8) == 1


def test_add_image_rejects_invalid_data(tmp_path: Path) -> None:
    ws = _Workspace(tmp_path / "ws")
    with pytest.raises(ValueError, match="not a readable image"):
        ws.add_image("bad.png", b"not an image")
    assert ws.list_images() == []
    with pytest.raises(ValueError, match="Unsupported image format"):
        ws.add_image("file.txt", b"hello")


def test_session_file_roundtrip(tmp_path: Path) -> None:
    session = make_session()
    path = save_session_file(session, tmp_path / "sub" / "session.json")
    assert path.is_file()
    assert load_session_file(path) == session
    # A directory gets a default file name, missing suffixes are added.
    assert save_session_file(session, tmp_path).name == "session.json"
    assert save_session_file(session, tmp_path / "named").name == (
        "named.json"
    )
    with pytest.raises(ValueError, match="file path is required"):
        save_session_file(session, "")
    with pytest.raises(ValueError, match="not found"):
        load_session_file(tmp_path / "missing.json")


def test_export_dataset_layout(workspace: _Workspace) -> None:
    result = export_dataset(
        workspace,
        make_session(),
        name="my_dataset",
        train_split=0.75,
        shuffle=True,
        seed=42,
    )
    dataset = workspace.root / "my_dataset"
    assert result["num_train"] == 3  # noqa: PLR2004
    assert result["num_val"] == 1

    with (dataset / "dataset.yaml").open() as f:
        content = yaml.safe_load(f)
    assert content["path"] == str(dataset.resolve())
    assert content["train"] == "images/train"
    assert content["val"] == "images/val"
    assert content["nc"] == 2  # noqa: PLR2004
    assert content["names"] == ["particle", "aggregate"]

    images = sorted(p.name for p in dataset.glob("images/*/*"))
    labels = sorted(p.name for p in dataset.glob("labels/*/*"))
    assert images == [f"img_{i}.png" for i in range(4)]
    assert labels == [f"img_{i}.txt" for i in range(4)]

    # Every image has a label file in the matching split folder.
    for img_path in dataset.glob("images/*/*"):
        split = img_path.parent.name
        lbl = dataset / "labels" / split / (img_path.stem + ".txt")
        assert lbl.is_file()


def test_export_dataset_stable_class_ids(workspace: _Workspace) -> None:
    # "aggregate" is the second label: its class ID must be 1 even if
    # it is the first annotation encountered.
    export_dataset(workspace, make_session(), name="ds", shuffle=False, seed=0)
    dataset = workspace.root / "ds"
    lines = (
        (dataset / "labels" / "train" / "img_0.txt")
        .read_text()
        .strip()
        .splitlines()
    )
    class_ids = [line.split()[0] for line in lines]
    assert class_ids == ["1", "0"]
    # YOLO boxes are normalized cx cy w h in [0, 1].
    for line in lines:
        values = [float(v) for v in line.split()[1:]]
        assert all(0.0 <= v <= 1.0 for v in values)


def test_export_dataset_errors(workspace: _Workspace) -> None:
    with pytest.raises(ValueError, match="No labels defined"):
        export_dataset(workspace, {"labels": []}, name="ds")
    with pytest.raises(ValueError, match="train_split"):
        export_dataset(workspace, make_session(), name="ds", train_split=1.5)


def test_export_dataset_custom_output(
    workspace: _Workspace, tmp_path: Path
) -> None:
    out = tmp_path / "elsewhere"
    result = export_dataset(
        workspace, make_session(), name="ds", output_dir=out
    )
    assert result["path"] == str((out / "ds").resolve())
    assert (out / "ds" / "dataset.yaml").is_file()


def test_synthesize_dataset(workspace: _Workspace) -> None:
    result = synthesize_dataset(
        workspace,
        make_session(),
        name="synt",
        num_images=5,
        width=128,
        height=96,
        per_image=3,
        train_split=0.8,
        scale_range=(0.5, 1.5),
        seed=7,
    )
    dataset = workspace.root / "synt"
    assert result["num_train"] == 4  # noqa: PLR2004
    assert result["num_val"] == 1
    train_images = list(dataset.glob("images/train/*.jpg"))
    assert len(train_images) == 4  # noqa: PLR2004
    with Image.open(train_images[0]) as img:
        assert img.size == (128, 96)
    for lbl in dataset.glob("labels/*/*.txt"):
        for line in lbl.read_text().splitlines():
            parts = line.split()
            assert parts[0] in {"0", "1"}
            assert all(0.0 <= float(v) <= 1.0 for v in parts[1:])


def test_synthesize_requires_annotations(tmp_path: Path) -> None:
    ws = _Workspace(tmp_path / "ws")
    ws.add_image("img.png", make_image_bytes())
    with pytest.raises(ValueError, match="No annotations"):
        synthesize_dataset(ws, {"labels": [], "annotations": {}}, name="s")


def test_http_api_roundtrip(tmp_path: Path) -> None:
    server = _LabelToolServer(0, _Workspace(tmp_path / "ws"))
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{port}"

    def request(
        path: str, method: str = "GET", data: bytes | None = None
    ) -> dict[str, Any]:
        req = urllib.request.Request(  # noqa: S310
            base + path, data=data, method=method
        )
        with urllib.request.urlopen(req) as response:  # noqa: S310
            return json.loads(response.read())

    try:
        info = request("/api/images?name=img.png", "POST", make_image_bytes())
        assert info == {"name": "img.png", "width": 64, "height": 48}

        # Edits are mirrored to the server memory, without disk writes.
        session = make_session()
        request("/api/sync", "POST", json.dumps(session).encode("utf-8"))

        state = request("/api/state")
        assert [img["name"] for img in state["images"]] == ["img.png"]
        assert state["labels"] == session["labels"]
        assert state["dirty"] is True
        assert state["session_path"] is None

        # No long operation running: the progress endpoint is idle.
        assert request("/api/progress") == {"active": False}

        # Saving requires an explicit path.
        with pytest.raises(urllib.error.HTTPError):
            request("/api/session", "POST", b"{}")
        session_file = tmp_path / "saved" / "session.json"
        saved = request(
            "/api/session",
            "POST",
            json.dumps({"path": str(session_file)}).encode("utf-8"),
        )
        assert saved["path"] == str(session_file)
        assert session_file.is_file()
        assert request("/api/state")["dirty"] is False

        loaded = request(
            "/api/session/load",
            "POST",
            json.dumps({"path": str(session_file)}).encode("utf-8"),
        )
        assert loaded == session

        export = request(
            "/api/export",
            "POST",
            json.dumps({"name": "ds", "seed": 1}).encode("utf-8"),
        )
        assert (tmp_path / "ws" / "ds" / "dataset.yaml").is_file()
        assert export["num_train"] == 1

        request("/api/images?name=img.png", "DELETE")
        assert request("/api/state")["images"] == []
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
