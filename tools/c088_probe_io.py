from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from tools.siglip_auto_caption_types import JsonObject, JsonValue


DEFAULT_OUT_DIR: Final = Path("eval/c088_shape_silhouette_feature_probe_20260613")
DEFAULT_CROP_SUMMARY: Final = Path(
    "eval/qwenvl_c087_expanded_crop_positive_gate_20260613/crop_pair_summary.json"
)
DEFAULT_FULL_SUMMARY: Final = Path(
    "eval/qwenvl_c087_expanded_crop_positive_gate_20260613/summary.json"
)
VARIANTS: Final = (
    "no_ip",
    "blend_species_face",
    "c085_anchored_full_adapter_w14",
    "c086_hard_negative_w14",
    "c087_expanded_crop_positive_w14",
)


@dataclass(frozen=True, slots=True)
class C088ProbeError(Exception):
    detail: str

    def __str__(self) -> str:
        return self.detail


def read_json(path: Path) -> JsonObject:
    raw: JsonValue = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise C088ProbeError(f"json root must be object: {path}")
    return raw


def write_json(path: Path, payload: JsonObject) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def json_object(row: JsonObject, field: str) -> JsonObject:
    value = row.get(field)
    if not isinstance(value, dict):
        raise C088ProbeError(f"field must be object: {field}")
    return value


def json_list(row: JsonObject, field: str) -> tuple[JsonObject, ...]:
    value = row.get(field)
    if not isinstance(value, list):
        raise C088ProbeError(f"field must be list: {field}")
    return tuple(item for item in value if isinstance(item, dict))


def count_field(rows: tuple[JsonObject, ...], field: str) -> JsonObject:
    counts: dict[str, int] = {}
    for row in rows:
        key = str(row[field])
        counts[key] = counts.get(key, 0) + 1
    return counts


def ensure_image(path: Path) -> None:
    if not path.is_file() or path.stat().st_size <= 0:
        raise C088ProbeError(f"missing or empty image: {path}")


def validate_manifest_rows(rows: tuple[JsonObject, ...]) -> None:
    if not rows:
        raise C088ProbeError("c088 manifest would be empty")
    for row in rows:
        ensure_image(Path(str(row["reference_path"])))
        candidates = json_object(row, "candidates")
        for variant in VARIANTS:
            ensure_image(Path(str(candidates[variant])))


def read_manifest_rows(path: Path) -> tuple[JsonObject, ...]:
    rows: list[JsonObject] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        raw: JsonValue = json.loads(line)
        if not isinstance(raw, dict):
            raise C088ProbeError(f"{path}:{line_number} row must be object")
        rows.append(raw)
    parsed = tuple(rows)
    validate_manifest_rows(parsed)
    return parsed
