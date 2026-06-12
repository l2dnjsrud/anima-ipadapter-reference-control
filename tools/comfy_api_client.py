from __future__ import annotations

import json
import time
import urllib.request

from tools.siglip_auto_caption_types import JsonObject


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
