from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from tools.c069_review_sheet import write_c069_review_sheet
from tools.siglip_auto_caption_types import JsonObject, JsonValue

LABEL_SCHEMA: Final = (
    "target_positive",
    "useful_proxy_non_human",
    "guard_false_positive_human",
    "guard_false_positive_background_object",
    "reject_unclear",
)


@dataclass(frozen=True, slots=True)
class C071PackageConfig:
    c068_reviewed_path: Path
    c069_reviewed_path: Path
    c070_reviewed_path: Path
    heldout_manifest_path: Path
    out_dir: Path


C071SeedPackageConfig = C071PackageConfig

SOURCE_PATHS: Final = ("c068", "c069", "c070")
REVIEW_BUCKETS: Final = tuple((label, label.replace("_", " ")) for label in LABEL_SCHEMA)


def build_c071_seed_package(config: C071PackageConfig) -> JsonObject:
    heldout_ids = _read_ids(config.heldout_manifest_path)
    rows = _collect_rows(config, heldout_ids=heldout_ids)
    candidates = _dedupe(rows)
    config.out_dir.mkdir(parents=True, exist_ok=True)
    _write_jsonl(config.out_dir / "annotation_candidates.jsonl", candidates)
    _write_csv(config.out_dir / "annotation_template.csv", candidates)
    write_c069_review_sheet(candidates, config.out_dir / "annotated_review_sheet.jpg", REVIEW_BUCKETS)
    summary = _summary(candidates, rows, heldout_ids=heldout_ids)
    (config.out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary


def _collect_rows(config: C071PackageConfig, *, heldout_ids: set[str]) -> tuple[JsonObject, ...]:
    rows: list[JsonObject] = []
    for source_name, path in _source_items(config):
        for index, raw in enumerate(_read_jsonl(path), start=1):
            image_id = str(raw.get("image_id", ""))
            if not image_id or image_id in heldout_ids:
                continue
            normalized = _normalize_row(raw, source_name=source_name, index=index)
            rows.append(normalized)
    return tuple(rows)


def _source_items(config: C071PackageConfig) -> tuple[tuple[str, Path], ...]:
    return (
        ("c068", config.c068_reviewed_path),
        ("c069", config.c069_reviewed_path),
        ("c070", config.c070_reviewed_path),
    )


def _normalize_row(raw: JsonObject, *, source_name: str, index: int) -> JsonObject:
    image_id = str(raw["image_id"])
    original_label = str(raw.get("review_label", "reject_unclear"))
    suggested_label = _suggested_label(original_label)
    score = _float_value(raw, "bucket_score", _float_value(raw, "score", 0.0))
    green = _float_value(raw, "green_ratio", 0.0)
    return {
        "candidate_id": f"c071_{source_name}_{index:03d}",
        "image_id": image_id,
        "image_path": str(raw["image_path"]),
        "source_bucket": suggested_label,
        "suggested_label": suggested_label,
        "source_experiments": [source_name],
        "source_labels": [original_label],
        "source_buckets": [str(raw.get("source_bucket", raw.get("query_key", "")))],
        "review_notes": [str(raw.get("review_note", ""))],
        "rank": int(raw.get("rank", index)),
        "bucket_score": score,
        "green_ratio": green,
        "central_green_ratio": _float_value(raw, "central_green_ratio", green),
        "red_ratio": _float_value(raw, "red_ratio", 0.0),
        "heldout_excluded": False,
        "path_exists": Path(str(raw["image_path"])).is_file(),
    }


def _suggested_label(original_label: str) -> str:
    match original_label:
        case "target_positive" | "useful_proxy_positive" | "useful_proxy_non_human":
            return "useful_proxy_non_human"
        case "false_positive_background_object" | "negative_anchor":
            return "guard_false_positive_background_object"
        case "false_positive_human" | "false_positive_human_face" | "false_positive_human_old_face" | "false_positive_red_eye_human":
            return "guard_false_positive_human"
        case _:
            return "reject_unclear"


def _dedupe(rows: tuple[JsonObject, ...]) -> tuple[JsonObject, ...]:
    by_id: dict[str, JsonObject] = {}
    for row in rows:
        image_id = str(row["image_id"])
        existing = by_id.get(image_id)
        if existing is None:
            by_id[image_id] = _candidate_copy(row)
            continue
        _append_unique(existing, "source_experiments", row)
        _append_unique(existing, "source_labels", row)
        _append_unique(existing, "source_buckets", row)
        _append_unique(existing, "review_notes", row)
        if _priority(str(row["suggested_label"])) < _priority(str(existing["suggested_label"])):
            existing["suggested_label"] = row["suggested_label"]
            existing["source_bucket"] = row["suggested_label"]
    return tuple(by_id.values())


def _candidate_copy(row: JsonObject) -> JsonObject:
    copied = dict(row)
    for key in ("source_experiments", "source_labels", "source_buckets", "review_notes"):
        values = row.get(key)
        if isinstance(values, list):
            copied[key] = list(values)
    return copied


def _append_unique(target: JsonObject, key: str, row: JsonObject) -> None:
    values = target.get(key)
    incoming = row.get(key)
    if not isinstance(values, list) or not isinstance(incoming, list):
        return
    for value in incoming:
        if value not in values:
            values.append(value)


def _priority(label: str) -> int:
    match label:
        case "useful_proxy_non_human":
            return 0
        case "guard_false_positive_human":
            return 1
        case "guard_false_positive_background_object":
            return 2
        case "reject_unclear":
            return 3
        case "target_positive":
            return 4
        case _:
            return 5


def _summary(candidates: tuple[JsonObject, ...], raw_rows: tuple[JsonObject, ...], *, heldout_ids: set[str]) -> JsonObject:
    return {
        "source": "c071_direct_green_manual_external_seed_package",
        "source_row_counts": _source_counts(raw_rows),
        "raw_candidate_rows": len(raw_rows),
        "unique_candidate_count": len(candidates),
        "heldout_ids_count": len(heldout_ids),
        "heldout_rows_used": sum(1 for row in candidates if bool(row["heldout_excluded"])),
        "missing_paths": sum(1 for row in candidates if not bool(row["path_exists"])),
        "label_schema": list(LABEL_SCHEMA),
        "suggested_label_counts": _count(candidates, "suggested_label"),
        "minimum_target_positive_required": 4,
        "decision": "manual_or_external_labels_required",
    }


def _source_counts(rows: tuple[JsonObject, ...]) -> dict[str, int]:
    counts: dict[str, int] = {source: 0 for source in SOURCE_PATHS}
    for row in rows:
        values = row.get("source_experiments")
        if not isinstance(values, list):
            continue
        for value in values:
            key = str(value)
            counts[key] = counts.get(key, 0) + 1
    return counts


def _count(rows: tuple[JsonObject, ...], key: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        value = str(row[key])
        counts[value] = counts.get(value, 0) + 1
    return counts


def _write_csv(path: Path, candidates: tuple[JsonObject, ...]) -> None:
    fieldnames = ("candidate_id", "image_id", "image_path", "suggested_label", "manual_label", "source_experiments", "source_labels")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in candidates:
            writer.writerow(
                {
                    "candidate_id": row["candidate_id"],
                    "image_id": row["image_id"],
                    "image_path": row["image_path"],
                    "suggested_label": row["suggested_label"],
                    "manual_label": "",
                    "source_experiments": "|".join(str(value) for value in row["source_experiments"] if isinstance(row["source_experiments"], list)),
                    "source_labels": "|".join(str(value) for value in row["source_labels"] if isinstance(row["source_labels"], list)),
                }
            )


def _float_value(row: JsonObject, key: str, default: float) -> float:
    value = row.get(key, default)
    if isinstance(value, int | float):
        return float(value)
    return default


def _read_ids(path: Path) -> set[str]:
    return {str(row["ref_id"]) for row in _read_jsonl(path) if isinstance(row.get("ref_id"), str)}


def _read_jsonl(path: Path) -> tuple[JsonObject, ...]:
    rows: list[JsonObject] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        raw: JsonValue = json.loads(line)
        if isinstance(raw, dict):
            rows.append(raw)
    return tuple(rows)


def _write_jsonl(path: Path, rows: tuple[JsonObject, ...]) -> None:
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
