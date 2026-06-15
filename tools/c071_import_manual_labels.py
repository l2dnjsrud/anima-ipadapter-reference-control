from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path

from tools.c071_seed_package import LABEL_SCHEMA
from tools.siglip_auto_caption_types import JsonObject, JsonValue


@dataclass(frozen=True, slots=True)
class C071ImportConfig:
    annotation_candidates_path: Path
    manual_labels_path: Path
    heldout_manifest_path: Path
    out_dir: Path


class ManualLabelImportError(ValueError):
    pass


def import_c071_manual_labels(config: C071ImportConfig) -> JsonObject:
    candidates = _candidate_map(config.annotation_candidates_path)
    heldout_ids = _read_ids(config.heldout_manifest_path)
    labels = _read_manual_labels(config.manual_labels_path)
    imported = _validated_import_rows(labels, candidates=candidates, heldout_ids=heldout_ids)
    target_rows = tuple(row for row in imported if row["manual_label"] == "target_positive")
    config.out_dir.mkdir(parents=True, exist_ok=True)
    _write_jsonl(config.out_dir / "imported_manual_labels.jsonl", imported)
    _write_jsonl(config.out_dir / "imported_confirmed_positives.jsonl", target_rows)
    summary = _summary(imported, target_rows)
    (config.out_dir / "import_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary


def _validated_import_rows(
    labels: tuple[JsonObject, ...],
    *,
    candidates: dict[str, JsonObject],
    heldout_ids: set[str],
) -> tuple[JsonObject, ...]:
    rows: list[JsonObject] = []
    positive_ids: set[str] = set()
    for label_row in labels:
        image_id = str(label_row.get("image_id", ""))
        manual_label = str(label_row.get("manual_label", ""))
        if image_id in heldout_ids:
            raise ManualLabelImportError(f"heldout image cannot be imported: {image_id}")
        if manual_label not in LABEL_SCHEMA:
            raise ManualLabelImportError(f"unknown label: {manual_label}")
        candidate = candidates.get(image_id)
        if candidate is None:
            raise ManualLabelImportError(f"unknown candidate image_id: {image_id}")
        if manual_label == "target_positive":
            if image_id in positive_ids:
                raise ManualLabelImportError(f"duplicate target_positive image_id: {image_id}")
            positive_ids.add(image_id)
        imported = dict(candidate)
        imported["manual_label"] = manual_label
        imported["manual_note"] = str(label_row.get("manual_note", ""))
        rows.append(imported)
    return tuple(rows)


def _summary(imported: tuple[JsonObject, ...], target_rows: tuple[JsonObject, ...]) -> JsonObject:
    target_count = len({str(row["image_id"]) for row in target_rows})
    return {
        "source": "c071_manual_label_import",
        "imported_rows": len(imported),
        "unique_target_positive_count": target_count,
        "label_counts": _count(imported, "manual_label"),
        "minimum_target_positive_required": 4,
        "decision": "ready_for_encoder_training" if target_count >= 4 else "external_manual_data_required",
    }


def _candidate_map(path: Path) -> dict[str, JsonObject]:
    return {str(row["image_id"]): row for row in _read_jsonl(path)}


def _read_manual_labels(path: Path) -> tuple[JsonObject, ...]:
    if path.suffix == ".jsonl":
        return _read_jsonl(path)
    with path.open(encoding="utf-8", newline="") as handle:
        return tuple(dict(row) for row in csv.DictReader(handle))


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


def _count(rows: tuple[JsonObject, ...], key: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        value = str(row[key])
        counts[value] = counts.get(value, 0) + 1
    return counts
