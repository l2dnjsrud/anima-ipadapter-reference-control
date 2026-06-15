from __future__ import annotations

import urllib.request
from pathlib import Path

from tools.comfy_api_client import (
    ComfyUIExecutionError,
    upload_image,
    view_image_bytes,
    wait_history,
)


class FakeResponse:
    def __init__(self, payload: bytes) -> None:
        self._payload = payload

    def __enter__(self) -> FakeResponse:
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        return None

    def read(self) -> bytes:
        return self._payload


def test_upload_image_posts_comfyui_multipart(
    tmp_path: Path,
    monkeypatch,
) -> None:
    image_path = tmp_path / "source.jpg"
    image_path.write_bytes(b"fake-jpeg")
    requests: list[urllib.request.Request] = []

    def fake_urlopen(
        request: urllib.request.Request,
        timeout: int,
    ) -> FakeResponse:
        requests.append(request)
        return FakeResponse(b'{"name":"reference.jpg","subfolder":"","type":"input"}')

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)

    uploaded = upload_image("http://127.0.0.1:8102", image_path, "reference.jpg")

    assert uploaded == "reference.jpg"
    assert requests[0].full_url == "http://127.0.0.1:8102/upload/image"
    assert requests[0].headers["Content-type"].startswith("multipart/form-data; boundary=")
    body = requests[0].data or b""
    assert b'name="image"; filename="reference.jpg"' in body
    assert b'name="type"' in body
    assert b"input" in body
    assert b"fake-jpeg" in body


def test_view_image_bytes_downloads_output_image(monkeypatch) -> None:
    requested_urls: list[str] = []

    def fake_urlopen(url: str, timeout: int) -> FakeResponse:
        requested_urls.append(url)
        return FakeResponse(b"png-bytes")

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)

    payload = view_image_bytes(
        "http://127.0.0.1:8102",
        filename="result image.png",
        subfolder="nested/out",
        image_type="output",
    )

    assert payload == b"png-bytes"
    assert requested_urls == [
        "http://127.0.0.1:8102/view?filename=result+image.png&subfolder=nested%2Fout&type=output"
    ]


def test_wait_history_raises_on_comfyui_execution_error(monkeypatch) -> None:
    payload = (
        b'{"prompt-1":{"outputs":{},"status":{"status_str":"error",'
        b'"messages":[["execution_error",{"node_id":"2","node_type":"Loader",'
        b'"exception_message":"shape mismatch"}]]}}}'
    )

    def fake_urlopen(url: str, timeout: int) -> FakeResponse:
        return FakeResponse(payload)

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)

    try:
        wait_history("http://127.0.0.1:8116", "prompt-1")
    except ComfyUIExecutionError as exc:
        assert exc.prompt_id == "prompt-1"
        assert exc.node_id == "2"
        assert "shape mismatch" in str(exc)
    else:
        raise AssertionError("ComfyUI execution errors should stop history waits")
