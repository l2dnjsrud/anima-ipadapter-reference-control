from __future__ import annotations

import json
import time
import urllib.parse
import urllib.request
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from tools.siglip_auto_caption_types import JsonObject, JsonValue


@dataclass(frozen=True, slots=True)
class ComfyUIExecutionError(Exception):
    prompt_id: str
    node_id: str
    node_type: str
    message: str

    def __str__(self) -> str:
        node = f"{self.node_type}({self.node_id})" if self.node_type else self.node_id
        return f"ComfyUI execution failed for {self.prompt_id} at {node}: {self.message}"


def post_json(base_url: str, path: str, payload: JsonObject) -> JsonObject:
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        f"{base_url}{path}",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        loaded = json.loads(response.read().decode("utf-8"))
    return loaded


def post_multipart_file(
    base_url: str,
    path: str,
    *,
    field_name: str,
    file_path: Path,
    upload_name: str,
    fields: Mapping[str, str],
) -> JsonObject:
    boundary = f"----codex-comfyui-{uuid.uuid4().hex}"
    body = b"".join(
        (
            *(
                _multipart_text_part(boundary, name, value)
                for name, value in fields.items()
            ),
            _multipart_file_part(boundary, field_name, file_path, upload_name),
            f"--{boundary}--\r\n".encode(),
        )
    )
    request = urllib.request.Request(
        f"{base_url}{path}",
        data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        loaded = json.loads(response.read().decode("utf-8"))
    return loaded


def upload_image(base_url: str, image_path: Path, image_name: str) -> str:
    response = post_multipart_file(
        base_url,
        "/upload/image",
        field_name="image",
        file_path=image_path,
        upload_name=image_name,
        fields={"type": "input", "overwrite": "true"},
    )
    name = str(response.get("name", image_name))
    subfolder = str(response.get("subfolder", ""))
    if subfolder:
        return f"{subfolder}/{name}"
    return name


def view_image_bytes(
    base_url: str,
    *,
    filename: str,
    subfolder: str,
    image_type: str,
) -> bytes:
    query = urllib.parse.urlencode(
        {"filename": filename, "subfolder": subfolder, "type": image_type}
    )
    with urllib.request.urlopen(f"{base_url}/view?{query}", timeout=60) as response:
        return response.read()


def get_json(base_url: str, path: str) -> JsonObject:
    with urllib.request.urlopen(f"{base_url}{path}", timeout=30) as response:
        loaded = json.loads(response.read().decode("utf-8"))
    return loaded


def wait_history(base_url: str, prompt_id: str) -> JsonObject:
    deadline = time.monotonic() + 1800
    while time.monotonic() < deadline:
        history = get_json(base_url, f"/history/{prompt_id}")
        record = history.get(prompt_id)
        if isinstance(record, dict) and record.get("outputs"):
            return record
        if isinstance(record, dict):
            execution_error = _execution_error_from_record(prompt_id, record)
            if execution_error is not None:
                raise execution_error
        time.sleep(2)
    raise TimeoutError(prompt_id)


def first_image_info(history: JsonObject) -> JsonObject:
    outputs = history["outputs"]
    if not isinstance(outputs, dict):
        raise TypeError("history outputs missing")
    for output in outputs.values():
        if isinstance(output, dict) and isinstance(output.get("images"), list):
            first = output["images"][0]
            if isinstance(first, dict):
                return first
    raise RuntimeError("history did not contain an image output")


def _multipart_text_part(boundary: str, name: str, value: str) -> bytes:
    return (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="{name}"\r\n\r\n'
        f"{value}\r\n"
    ).encode()


def _multipart_file_part(
    boundary: str,
    field_name: str,
    file_path: Path,
    upload_name: str,
) -> bytes:
    with file_path.open("rb") as handle:
        payload = handle.read()
    header = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="{field_name}"; filename="{upload_name}"\r\n'
        "Content-Type: image/jpeg\r\n\r\n"
    ).encode()
    return header + payload + b"\r\n"


def _execution_error_from_record(
    prompt_id: str,
    record: dict[str, JsonValue],
) -> ComfyUIExecutionError | None:
    status = record.get("status")
    if not isinstance(status, dict) or status.get("status_str") != "error":
        return None
    messages = status.get("messages")
    if isinstance(messages, list):
        for item in messages:
            if not isinstance(item, list) or len(item) < 2:
                continue
            tag = item[0]
            details = item[1]
            if tag == "execution_error" and isinstance(details, dict):
                return ComfyUIExecutionError(
                    prompt_id=prompt_id,
                    node_id=_text_field(details, "node_id"),
                    node_type=_text_field(details, "node_type"),
                    message=_text_field(details, "exception_message"),
                )
    return ComfyUIExecutionError(
        prompt_id=prompt_id,
        node_id="",
        node_type="",
        message="ComfyUI execution failed without execution_error details",
    )


def _text_field(data: dict[str, JsonValue], key: str) -> str:
    value = data.get(key)
    if isinstance(value, str):
        return value
    return ""
