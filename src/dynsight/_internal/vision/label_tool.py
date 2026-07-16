"""Local web application for building YOLO training datasets.

The tool starts a small HTTP server (standard library only) that serves
a single-page labeling GUI and a JSON API. Images (or video frames) are
stored inside a *workspace* directory together with the labeling
session, so the work survives page reloads and restarts. Datasets are
written directly to disk in the exact layout expected by
:class:`dynsight.vision.VisionInstance`.
"""

from __future__ import annotations

import json
import logging
import random
import re
import shutil
import threading
import webbrowser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

import yaml
from PIL import Image

logger = logging.getLogger(__name__)

_STATIC_DIR = Path(__file__).parent / "label_tool"
_STATIC_FILES = {
    "/": ("index.html", "text/html; charset=utf-8"),
    "/index.html": ("index.html", "text/html; charset=utf-8"),
    "/styles.css": ("styles.css", "text/css; charset=utf-8"),
    "/app.js": ("app.js", "text/javascript; charset=utf-8"),
}

_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
_VIDEO_SUFFIXES = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
_UNSAFE_CHARS = re.compile(r"[^A-Za-z0-9._-]+")

_MIN_SPLIT_IMAGES = 2
_MAX_PLACEMENT_TRIES = 50


def _safe_name(raw: str) -> str:
    """Reduce a client-provided file name to a safe basename."""
    name = _UNSAFE_CHARS.sub("_", Path(raw).name)
    if not re.search(r"[A-Za-z0-9]", Path(name).stem):
        msg = f"Invalid file name: '{raw}'"
        raise ValueError(msg)
    return name


def _image_size(path: Path) -> tuple[int, int]:
    """Return (width, height) of an image without loading pixel data."""
    with Image.open(path) as img:
        return img.size


class _Workspace:
    """Filesystem-backed state of a labeling session."""

    def __init__(self, root: Path) -> None:
        self.root = root.resolve()
        self.images_dir = self.root / "images"
        self.session_file = self.root / "session.json"
        self.images_dir.mkdir(parents=True, exist_ok=True)

    def load_session(self) -> dict[str, Any]:
        """Read the stored session, returning an empty one if missing."""
        if self.session_file.is_file():
            with self.session_file.open("r", encoding="utf-8") as f:
                data: dict[str, Any] = json.load(f)
            return data
        return {"labels": [], "annotations": {}}

    def save_session(self, session: dict[str, Any]) -> None:
        """Persist the session to disk."""
        with self.session_file.open("w", encoding="utf-8") as f:
            json.dump(session, f, indent=1)

    def list_images(self) -> list[dict[str, Any]]:
        """Return metadata for every image stored in the workspace."""
        infos = []
        for path in sorted(self.images_dir.iterdir()):
            if path.suffix.lower() not in _IMAGE_SUFFIXES:
                continue
            width, height = _image_size(path)
            infos.append({"name": path.name, "width": width, "height": height})
        return infos

    def add_image(self, name: str, data: bytes) -> dict[str, Any]:
        """Store an uploaded image after validating it."""
        safe = _safe_name(name)
        if Path(safe).suffix.lower() not in _IMAGE_SUFFIXES:
            msg = f"Unsupported image format: '{safe}'"
            raise ValueError(msg)
        dst = self.images_dir / safe
        dst.write_bytes(data)
        try:
            width, height = _image_size(dst)
        # Broad catch: PIL.Image.open may be monkey-patched by other
        # libraries (e.g. ultralytics) and raise unexpected errors.
        except Exception:  # noqa: BLE001
            dst.unlink(missing_ok=True)
            msg = f"'{safe}' is not a readable image."
            raise ValueError(msg) from None
        return {"name": safe, "width": width, "height": height}

    def add_video(
        self, name: str, data: bytes, stride: int
    ) -> list[dict[str, Any]]:
        """Extract frames from an uploaded video into the workspace."""
        import cv2  # noqa: PLC0415 (heavy import, only needed here)

        safe = _safe_name(name)
        if Path(safe).suffix.lower() not in _VIDEO_SUFFIXES:
            msg = f"Unsupported video format: '{safe}'"
            raise ValueError(msg)
        stride = max(1, stride)
        tmp = self.root / f"_upload_{safe}"
        tmp.write_bytes(data)
        stem = Path(safe).stem
        frames: list[dict[str, Any]] = []
        try:
            capture = cv2.VideoCapture(str(tmp))
            if not capture.isOpened():
                msg = f"Could not open video '{safe}'."
                raise ValueError(msg)
            index = 0
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                if index % stride == 0:
                    frame_name = f"{stem}_{index:06d}.jpg"
                    cv2.imwrite(str(self.images_dir / frame_name), frame)
                    height, width = frame.shape[:2]
                    frames.append(
                        {
                            "name": frame_name,
                            "width": int(width),
                            "height": int(height),
                        }
                    )
                index += 1
            capture.release()
        finally:
            tmp.unlink(missing_ok=True)
        if not frames:
            msg = f"No frames could be extracted from '{safe}'."
            raise ValueError(msg)
        return frames

    def delete_image(self, name: str) -> None:
        """Remove an image from the workspace."""
        (self.images_dir / _safe_name(name)).unlink(missing_ok=True)


def _yolo_lines(
    boxes: list[dict[str, Any]],
    class_ids: dict[str, int],
    width: int,
    height: int,
) -> str:
    """Convert pixel-space boxes to YOLO txt content."""
    lines = []
    for box in boxes:
        label = box["label"]
        if label not in class_ids:
            continue
        cx = (box["x"] + box["w"] / 2) / width
        cy = (box["y"] + box["h"] / 2) / height
        w = box["w"] / width
        h = box["h"] / height
        cx, cy = min(max(cx, 0.0), 1.0), min(max(cy, 0.0), 1.0)
        w, h = min(max(w, 0.0), 1.0), min(max(h, 0.0), 1.0)
        lines.append(f"{class_ids[label]} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
    return "".join(f"{line}\n" for line in lines)


def _dataset_dirs(dataset_path: Path) -> dict[str, Path]:
    """Create and return the YOLO dataset directory layout."""
    dirs = {
        "images/train": dataset_path / "images" / "train",
        "images/val": dataset_path / "images" / "val",
        "labels/train": dataset_path / "labels" / "train",
        "labels/val": dataset_path / "labels" / "val",
    }
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return dirs


def _write_dataset_yaml(dataset_path: Path, names: list[str]) -> Path:
    """Write the dataset.yaml file consumed by ``set_training_dataset``."""
    yaml_path = dataset_path / "dataset.yaml"
    content = {
        "path": str(dataset_path.resolve()),
        "train": "images/train",
        "val": "images/val",
        "nc": len(names),
        "names": names,
    }
    with yaml_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(content, f, sort_keys=False)
    return yaml_path


def _split_count(total: int, train_split: float) -> int:
    """Number of training items for a given split fraction."""
    num_train = round(total * train_split)
    if total >= _MIN_SPLIT_IMAGES:
        num_train = min(max(num_train, 1), total - 1)
    return num_train


def export_dataset(
    workspace: _Workspace,
    session: dict[str, Any],
    name: str,
    train_split: float = 0.8,
    shuffle: bool = True,
    seed: int | None = None,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    """Write a YOLO dataset from the current session to disk.

    Class IDs follow the order of the session label list, so they are
    stable across exports. Every image receives a label file (empty if
    it has no annotations) and every known class appears in
    ``dataset.yaml`` even when unused.
    """
    if not 0.0 < train_split < 1.0:
        msg = "train_split must be between 0 and 1."
        raise ValueError(msg)
    images = workspace.list_images()
    if not images:
        msg = "No images in the workspace."
        raise ValueError(msg)
    names = [label["name"] for label in session.get("labels", [])]
    if not names:
        msg = "No labels defined."
        raise ValueError(msg)
    class_ids = {label: idx for idx, label in enumerate(names)}
    annotations: dict[str, Any] = session.get("annotations", {})

    base = output_dir if output_dir is not None else workspace.root
    dataset_path = (base / _safe_name(name)).resolve()
    dirs = _dataset_dirs(dataset_path)

    if shuffle:
        random.Random(seed).shuffle(images)  # noqa: S311
    num_train = _split_count(len(images), train_split)

    for idx, info in enumerate(images):
        subset = "train" if idx < num_train else "val"
        src = workspace.images_dir / info["name"]
        shutil.copy2(src, dirs[f"images/{subset}"] / info["name"])
        txt = _yolo_lines(
            annotations.get(info["name"], []),
            class_ids,
            info["width"],
            info["height"],
        )
        lbl = dirs[f"labels/{subset}"] / (Path(info["name"]).stem + ".txt")
        lbl.write_text(txt, encoding="utf-8")

    yaml_path = _write_dataset_yaml(dataset_path, names)
    return {
        "path": str(dataset_path),
        "yaml": str(yaml_path),
        "num_train": num_train,
        "num_val": len(images) - num_train,
    }


def _place_crop(
    rng: random.Random,
    crop_size: tuple[int, int],
    canvas_size: tuple[int, int],
    placed: list[tuple[float, float, float, float]],
    scale_range: tuple[float, float],
) -> tuple[int, int, int, int] | None:
    """Find a non-overlapping position for a crop, or ``None``."""
    for _ in range(_MAX_PLACEMENT_TRIES):
        scale = rng.uniform(*scale_range)
        w = max(1, int(crop_size[0] * scale))
        h = max(1, int(crop_size[1] * scale))
        if w >= canvas_size[0] or h >= canvas_size[1]:
            continue
        x = rng.randint(0, canvas_size[0] - w)
        y = rng.randint(0, canvas_size[1] - h)
        overlap = any(
            x < px + pw and x + w > px and y < py + ph and y + h > py
            for px, py, pw, ph in placed
        )
        if not overlap:
            return x, y, w, h
    return None


def synthesize_dataset(
    workspace: _Workspace,
    session: dict[str, Any],
    name: str,
    num_images: int = 10,
    width: int = 640,
    height: int = 640,
    per_image: int = 10,
    train_split: float = 0.8,
    scale_range: tuple[float, float] = (1.0, 1.0),
    background: str = "#ffffff",
    seed: int | None = None,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    """Generate a synthetic YOLO dataset from the annotated crops.

    Annotated regions are cut out of the source images and pasted at
    random non-overlapping positions onto uniform background canvases.
    """
    names = [label["name"] for label in session.get("labels", [])]
    class_ids = {label: idx for idx, label in enumerate(names)}
    annotations: dict[str, Any] = session.get("annotations", {})
    crops = [
        {"image": image_name, **box}
        for image_name, boxes in annotations.items()
        for box in boxes
        if box["label"] in class_ids
        and (workspace.images_dir / image_name).is_file()
    ]
    if not crops:
        msg = "No annotations available to synthesize from."
        raise ValueError(msg)

    base = output_dir if output_dir is not None else workspace.root
    dataset_path = (base / _safe_name(name)).resolve()
    dirs = _dataset_dirs(dataset_path)
    rng = random.Random(seed)  # noqa: S311
    num_train = _split_count(num_images, train_split)

    sources: dict[str, Image.Image] = {}
    for idx in range(num_images):
        canvas = Image.new("RGB", (width, height), background)
        placed: list[tuple[float, float, float, float]] = []
        boxes: list[dict[str, Any]] = []
        for _ in range(per_image):
            crop = rng.choice(crops)
            if crop["image"] not in sources:
                src_path = workspace.images_dir / crop["image"]
                sources[crop["image"]] = Image.open(src_path).convert("RGB")
            source = sources[crop["image"]]
            left, top = int(crop["x"]), int(crop["y"])
            cw = max(1, int(crop["w"]))
            ch = max(1, int(crop["h"]))
            patch = source.crop((left, top, left + cw, top + ch))
            spot = _place_crop(
                rng, (cw, ch), (width, height), placed, scale_range
            )
            if spot is None:
                continue
            x, y, w, h = spot
            canvas.paste(patch.resize((w, h)), (x, y))
            placed.append((x, y, w, h))
            boxes.append(
                {"label": crop["label"], "x": x, "y": y, "w": w, "h": h}
            )
        subset = "train" if idx < num_train else "val"
        canvas.save(dirs[f"images/{subset}"] / f"synt_{idx:05d}.jpg")
        txt = _yolo_lines(boxes, class_ids, width, height)
        lbl = dirs[f"labels/{subset}"] / f"synt_{idx:05d}.txt"
        lbl.write_text(txt, encoding="utf-8")

    for source in sources.values():
        source.close()
    yaml_path = _write_dataset_yaml(dataset_path, names)
    return {
        "path": str(dataset_path),
        "yaml": str(yaml_path),
        "num_train": num_train,
        "num_val": num_images - num_train,
    }


class _LabelToolServer(ThreadingHTTPServer):
    """HTTP server carrying the workspace shared by all requests."""

    allow_reuse_address = True
    daemon_threads = True

    def __init__(self, port: int, workspace: _Workspace) -> None:
        self.workspace = workspace
        super().__init__(("127.0.0.1", port), _RequestHandler)


class _RequestHandler(BaseHTTPRequestHandler):
    """Routes static files and the JSON API."""

    server: _LabelToolServer

    def log_message(self, fmt: str, *args: object) -> None:
        """Silence default request logging."""

    @property
    def _workspace(self) -> _Workspace:
        return self.server.workspace

    def _send_json(
        self, payload: dict[str, Any], status: int = HTTPStatus.OK
    ) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_file(
        self, path: Path, content_type: str, cache_control: str = "no-store"
    ) -> None:
        if not path.is_file():
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        body = path.read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", cache_control)
        self.end_headers()
        self.wfile.write(body)

    def _read_body(self) -> bytes:
        length = int(self.headers.get("Content-Length", "0"))
        return self.rfile.read(length)

    def _query(self) -> dict[str, str]:
        parsed = parse_qs(urlparse(self.path).query)
        return {key: values[0] for key, values in parsed.items()}

    def do_GET(self) -> None:
        """Serve the GUI, workspace images and the state endpoint."""
        route = urlparse(self.path).path
        if route in _STATIC_FILES:
            file_name, content_type = _STATIC_FILES[route]
            self._send_file(_STATIC_DIR / file_name, content_type)
        elif route.startswith("/images/"):
            name = _safe_name(route[len("/images/") :])
            suffix = Path(name).suffix.lower().lstrip(".")
            content_type = f"image/{'jpeg' if suffix == 'jpg' else suffix}"
            self._send_file(
                self._workspace.images_dir / name,
                content_type,
                cache_control="max-age=300",
            )
        elif route == "/api/state":
            self._api(self._handle_state)
        else:
            self.send_error(HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:
        """Dispatch API mutations."""
        route = urlparse(self.path).path
        handlers = {
            "/api/session": self._handle_save_session,
            "/api/images": self._handle_upload_image,
            "/api/video": self._handle_upload_video,
            "/api/export": self._handle_export,
            "/api/synthesize": self._handle_synthesize,
            "/api/shutdown": self._handle_shutdown,
        }
        handler = handlers.get(route)
        if handler is None:
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        self._api(handler)

    def do_DELETE(self) -> None:
        """Delete a workspace image."""
        route = urlparse(self.path).path
        if route == "/api/images":
            self._api(self._handle_delete_image)
        else:
            self.send_error(HTTPStatus.NOT_FOUND)

    def _api(self, handler: Any) -> None:
        try:
            payload = handler()
        except (ValueError, KeyError, TypeError, json.JSONDecodeError) as e:
            logger.warning(f"Request failed: {e}")
            self._send_json({"error": str(e)}, status=HTTPStatus.BAD_REQUEST)
        else:
            self._send_json(payload)

    def _json_body(self) -> dict[str, Any]:
        data: dict[str, Any] = json.loads(self._read_body() or b"{}")
        return data

    def _handle_state(self) -> dict[str, Any]:
        session = self._workspace.load_session()
        return {
            "workspace": str(self._workspace.root),
            "images": self._workspace.list_images(),
            "labels": session.get("labels", []),
            "annotations": session.get("annotations", {}),
        }

    def _handle_save_session(self) -> dict[str, Any]:
        body = self._json_body()
        session = {
            "labels": body.get("labels", []),
            "annotations": body.get("annotations", {}),
        }
        self._workspace.save_session(session)
        return {"saved": True}

    def _handle_upload_image(self) -> dict[str, Any]:
        query = self._query()
        return self._workspace.add_image(query["name"], self._read_body())

    def _handle_upload_video(self) -> dict[str, Any]:
        query = self._query()
        frames = self._workspace.add_video(
            query["name"],
            self._read_body(),
            stride=int(query.get("stride", "1")),
        )
        return {"frames": frames}

    def _handle_delete_image(self) -> dict[str, Any]:
        query = self._query()
        self._workspace.delete_image(query["name"])
        return {"deleted": True}

    def _handle_export(self) -> dict[str, Any]:
        body = self._json_body()
        output = body.get("output_dir")
        return export_dataset(
            self._workspace,
            self._workspace.load_session(),
            name=body.get("name", "yolo_dataset"),
            train_split=float(body.get("train_split", 0.8)),
            shuffle=bool(body.get("shuffle", True)),
            seed=body.get("seed"),
            output_dir=Path(output) if output else None,
        )

    def _handle_synthesize(self) -> dict[str, Any]:
        body = self._json_body()
        output = body.get("output_dir")
        return synthesize_dataset(
            self._workspace,
            self._workspace.load_session(),
            name=body.get("name", "synt_dataset"),
            num_images=int(body.get("num_images", 10)),
            width=int(body.get("width", 640)),
            height=int(body.get("height", 640)),
            per_image=int(body.get("per_image", 10)),
            train_split=float(body.get("train_split", 0.8)),
            scale_range=(
                float(body.get("scale_min", 1.0)),
                float(body.get("scale_max", 1.0)),
            ),
            background=str(body.get("background", "#ffffff")),
            seed=body.get("seed"),
            output_dir=Path(output) if output else None,
        )

    def _handle_shutdown(self) -> dict[str, Any]:
        logger.info("Shutdown requested from the GUI.")
        threading.Thread(target=self.server.shutdown).start()
        return {"shutdown": True}


def label_tool(
    port: int = 8888,
    workspace: str | Path | None = None,
    open_browser: bool = True,
) -> None:
    """Start the dynsight labeling tool.

    The tool opens in the default web browser. All uploaded images and
    the labeling progress are stored in ``workspace`` so a session can
    be resumed at any time. The server stops with the *Quit* button in
    the GUI or with ``Ctrl+C`` in the terminal.

    Parameters:
        port:
            Port for the local HTTP server.

        workspace:
            Directory where images, the labeling session and exported
            datasets are stored. Defaults to ``./label_tool_workspace``.

        open_browser:
            Automatically open the GUI in the default browser.
    """
    root = (
        Path(workspace) if workspace else Path.cwd() / ("label_tool_workspace")
    )
    server = _LabelToolServer(port, _Workspace(root))
    url = f"http://127.0.0.1:{port}/"
    logger.info(f"Labeling tool running at {url}")
    logger.info(f"Workspace: {root.resolve()}")
    if open_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info("Interrupted.")
    finally:
        server.server_close()
        logger.info("Server closed.")
