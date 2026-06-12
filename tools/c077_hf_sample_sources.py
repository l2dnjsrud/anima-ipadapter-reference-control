from __future__ import annotations

import json
import socket
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Final

from tools.siglip_auto_caption_types import JsonObject, JsonValue

HF_DATASET_API: Final = "https://huggingface.co/api/datasets"


@dataclass(frozen=True, slots=True)
class C077SampleSource:
    repo: str
    license_note: str
    source_note: str


@dataclass(frozen=True, slots=True)
class C077TreeProbe:
    repo: str
    official_url: str
    access_status: str
    license_note: str
    path_status: str
    inspected_path_count: int
    sample_paths: tuple[str, ...]
    source_note: str


C077_SAMPLE_SOURCES: Final = (
    C077SampleSource("CyberHarem/green_heart_azurlane", "mit; not-for-all-audiences", "green-named anime sample assets; likely hair/outfit guard source"),
    C077SampleSource("CyberHarem/poppy_leagueoflegends", "mit; not-for-all-audiences", "non-human/yordle sample assets; likely not direct-green skin"),
    C077SampleSource("CyberHarem/tristana_leagueoflegends", "mit; not-for-all-audiences", "non-human/yordle sample assets; likely not direct-green skin"),
    C077SampleSource("CyberHarem/lulu_leagueoflegends", "mit; not-for-all-audiences", "non-human/yordle sample assets; likely not direct-green skin"),
    C077SampleSource("CyberHarem/soraka_leagueoflegends", "mit; not-for-all-audiences", "non-human/horned sample assets; likely purple rather than green"),
    C077SampleSource("CyberHarem/nami_leagueoflegends", "mit; not-for-all-audiences", "non-human/mermaid sample assets; likely blue rather than green"),
    C077SampleSource("CyberHarem/vex_leagueoflegends", "mit; not-for-all-audiences", "non-human/yordle sample assets; likely dark/blue rather than green"),
)


def fetch_c077_tree_probe(source: C077SampleSource, *, timeout_seconds: float) -> C077TreeProbe:
    payload = _get_json(_tree_url(source.repo), timeout_seconds)
    access_status = "public"
    path_status = "paths_ok"
    if "error" in payload:
        access_status = f"api_error:{payload['error']}"
        path_status = f"paths_error:{payload['error']}"
    paths = _sample_png_paths(payload)
    return C077TreeProbe(
        repo=source.repo,
        official_url=f"https://huggingface.co/datasets/{source.repo}",
        access_status=access_status,
        license_note=source.license_note,
        path_status=path_status,
        inspected_path_count=len(paths),
        sample_paths=paths,
        source_note=source.source_note,
    )


def _tree_url(repo: str) -> str:
    encoded = urllib.parse.quote(repo, safe="/")
    return f"{HF_DATASET_API}/{encoded}/tree/main/samples?recursive=1"


def _sample_png_paths(payload: JsonObject) -> tuple[str, ...]:
    rows = payload if isinstance(payload, list) else payload.get("siblings", ())
    paths: list[str] = []
    if isinstance(rows, list):
        for row in rows:
            match row:
                case {"path": str(path)} if path.endswith(".png"):
                    paths.append(path)
                case _:
                    continue
    return tuple(sorted(paths))


def _get_json(url: str, timeout_seconds: float) -> JsonObject:
    request = urllib.request.Request(url, headers={"User-Agent": "codex-c077-sample-source"})
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            value: JsonValue = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        return {"error": f"http_{error.code}"}
    except urllib.error.URLError as error:
        return {"error": f"url_error:{error.reason}"}
    except TimeoutError:
        return {"error": "timeout"}
    except socket.timeout:
        return {"error": "socket_timeout"}
    except json.JSONDecodeError as error:
        return {"error": f"json_decode:{error.msg}"}
    if isinstance(value, dict):
        return value
    if isinstance(value, list):
        return {"siblings": value}
    return {"error": "non_json_object_or_list"}
