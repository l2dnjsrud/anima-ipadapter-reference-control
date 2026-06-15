from __future__ import annotations

import json
import socket
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Final

from tools.siglip_auto_caption_types import JsonObject, JsonValue

HF_DATASETS_API: Final = "https://huggingface.co/api/datasets"
HF_VIEWER_API: Final = "https://datasets-server.huggingface.co"


@dataclass(frozen=True, slots=True)
class SourceSpec:
    repo: str
    official_url: str
    fallback_license: str
    source_note: str


@dataclass(frozen=True, slots=True)
class DatasetProbe:
    repo: str
    official_url: str
    access_status: str
    license_note: str
    metadata_probe_status: str
    features: tuple[str, ...]
    inspected_row_count: int
    rows: tuple[JsonObject, ...]
    probe_note: str


SOURCES: Final = (
    SourceSpec(
        "Wenaka/anima-ip-adapter-dataset",
        "https://huggingface.co/datasets/Wenaka/anima-ip-adapter-dataset",
        "unknown",
        "image-only tar shards; no model card/license in API",
    ),
    SourceSpec(
        "mrzjy/AniGamePersonaCaps",
        "https://huggingface.co/datasets/mrzjy/AniGamePersonaCaps",
        "cc-by-sa-4.0",
        "anime/manga/game character metadata and captions",
    ),
    SourceSpec(
        "mrzjy/AnimeMangaCharacters-247K",
        "https://huggingface.co/datasets/mrzjy/AnimeMangaCharacters-247K",
        "cc-by-4.0",
        "Fandom character metadata with source image URLs",
    ),
    SourceSpec(
        "alfredplpl/anime-with-caption-cc0",
        "https://huggingface.co/datasets/alfredplpl/anime-with-caption-cc0",
        "cc0-1.0",
        "anime image-caption rows; useful for synthetic-style positives",
    ),
    SourceSpec(
        "CaptionEmporium/furry-e621-safe-llama3.2-11b",
        "https://huggingface.co/datasets/CaptionEmporium/furry-e621-safe-llama3.2-11b",
        "cc-by-sa-4.0",
        "caption metadata only; high non-human density but no image URL in viewer row",
    ),
)


def fetch_probe(source: SourceSpec, *, row_limit: int, timeout_seconds: float) -> DatasetProbe:
    meta = _get_json(f"{HF_DATASETS_API}/{urllib.parse.quote(source.repo, safe='/')}", timeout_seconds)
    rows_payload = _get_json(
        f"{HF_VIEWER_API}/rows?{urllib.parse.urlencode({'dataset': source.repo, 'config': 'default', 'split': 'train', 'offset': 0, 'length': row_limit})}",
        timeout_seconds,
    )
    access_status = "public"
    metadata_status = "rows_ok"
    rows = tuple(row["row"] for row in _list_value(rows_payload.get("rows")) if isinstance(row.get("row"), dict))
    if "error" in rows_payload:
        metadata_status = f"rows_error:{rows_payload['error']}"
    if "error" in meta:
        access_status = f"api_error:{meta['error']}"
    features = tuple(str(feature.get("name", "")) for feature in _list_value(rows_payload.get("features")) if isinstance(feature, dict))
    return DatasetProbe(
        repo=source.repo,
        official_url=source.official_url,
        access_status=access_status,
        license_note=_license_note(meta, source.fallback_license),
        metadata_probe_status=metadata_status,
        features=features,
        inspected_row_count=len(rows),
        rows=rows,
        probe_note=source.source_note,
    )


def _get_json(url: str, timeout_seconds: float) -> JsonObject:
    request = urllib.request.Request(url, headers={"User-Agent": "codex-c072-source-discovery"})
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
    return {"error": "non_object_response"}


def _license_note(meta: JsonObject, fallback: str) -> str:
    card = meta.get("cardData")
    if isinstance(card, dict):
        license_value = card.get("license")
        if isinstance(license_value, str) and license_value:
            return license_value
    tags = tuple(str(tag) for tag in _list_value(meta.get("tags")))
    for tag in tags:
        if tag.startswith("license:"):
            return tag.removeprefix("license:")
    return fallback


def _list_value(value: JsonValue | None) -> tuple[JsonValue, ...]:
    return tuple(value) if isinstance(value, list) else ()
