from __future__ import annotations

import http.client
import socket
import threading
import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pytest

from dynsight.vision import label_tool


def free_port() -> int:
    sock = socket.socket()
    sock.bind(("", 0))
    port = sock.getsockname()[1]
    sock.close()
    return port


def test_label_tool(monkeypatch: pytest.MonkeyPatch) -> None:
    port = free_port()
    opened: list[str] = []
    monkeypatch.setattr("webbrowser.open", lambda url: opened.append(url))

    thread = threading.Thread(target=label_tool, kwargs={"port": port})
    thread.start()
    time.sleep(0.5)

    conn = http.client.HTTPConnection("localhost", port)
    conn.request("GET", "/index.html")
    resp = conn.getresponse()
    status_num = 200
    assert resp.status == status_num
    conn.close()

    conn = http.client.HTTPConnection("localhost", port)
    conn.request("POST", "/shutdown")
    conn.getresponse()
    conn.close()

    thread.join(timeout=5)
    assert not thread.is_alive()
    assert opened == [f"http://localhost:{port}/index.html"]
