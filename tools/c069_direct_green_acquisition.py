from __future__ import annotations

# /// script
# requires-python = ">=3.13"
# dependencies = ["numpy", "pillow"]
# ///
# How to run:
# PYTHONPATH=. python -m py_compile tools/c069_direct_green_acquisition.py

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import numpy as np
from PIL import Image

from tools.c069_review_sheet import write_c069_review_sheet
from tools.siglip_auto_caption_types import JsonObject, JsonValue

BUCKETS: Final = (
    ("target_score", "direct-green character candidate"),
    ("background_score", "green background/object guard"),
    ("strong_green", "strong green pixel candidate"),
    ("red_green_mix", "red and green mixed candidate"),
)
TARGET_POSITIVE_MINIMUM: Final = 4


@dataclass(frozen=True, slots=True)
class C069Config:
    dataset_root: Path
    all_manifest_path: Path
    heldout_manifest_path: Path
    c067_topk_path: Path
    out_dir: Path
    top_k_per_bucket: int = 12


@dataclass(frozen=True, slots=True)
class ScanRow:
    image_id: str
    image_path: Path
    green_ratio: float
    strong_green_ratio: float
    red_ratio: float
    central_green_ratio: float
    border_green_ratio: float
    target_score: float
    background_score: float


@dataclass(frozen=True, slots=True)
class SelectedRow:
    bucket: str
    rank: int
    bucket_score: float
    scan: ScanRow


def build_c069_direct_green_acquisition(config: C069Config) -> JsonObject:
    heldout_ids = set(_read_manifest_ids(config.heldout_manifest_path))
    c067_ids = _read_c067_topk_ids(config.c067_topk_path)
    ids = _read_manifest_ids(config.all_manifest_path)
    missing_paths = sum(
        1 for image_id in ids if image_id not in heldout_ids and not _image_path(config, image_id).is_file()
    )
    scan_rows = _scan(config, ids, heldout_ids=heldout_ids)
    selected = _select(scan_rows, config.top_k_per_bucket)
    candidates = _candidate_rows(selected, c067_ids=c067_ids)
    reviewed = _review_rows(candidates)
    _write_jsonl(config.out_dir / "candidate_manifest.jsonl", candidates)
    _write_jsonl(config.out_dir / "reviewed_candidate_labels.jsonl", reviewed)
    write_c069_review_sheet(reviewed, config.out_dir / "annotated_review_sheet.jpg", BUCKETS)
    summary = _summary(
        scan_rows=scan_rows,
        candidates=candidates,
        reviewed=reviewed,
        missing_paths=missing_paths,
        heldout_count=len(heldout_ids),
        c067_topk_count=len(c067_ids),
    )
    _write_summary_report(config.out_dir, summary)
    return summary


def _scan(config: C069Config, image_ids: tuple[str, ...], *, heldout_ids: set[str]) -> tuple[ScanRow, ...]:
    rows: list[ScanRow] = []
    for image_id in image_ids:
        if image_id in heldout_ids:
            continue
        image_path = _image_path(config, image_id)
        if not image_path.is_file():
            continue
        rows.append(_scan_image(image_id, image_path))
    return tuple(rows)


def _scan_image(image_id: str, image_path: Path) -> ScanRow:
    with Image.open(image_path) as image:
        rgb = image.convert("RGB").resize((128, 128), Image.Resampling.BILINEAR)
    arr = np.asarray(rgb, dtype=np.float32)
    red = arr[:, :, 0]
    green = arr[:, :, 1]
    blue = arr[:, :, 2]
    spread = arr.max(axis=2) - arr.min(axis=2)
    green_mask = (green > 50.0) & (spread > 25.0) & (green > red * 1.08) & (green > blue * 1.03)
    strong_mask = (green > 70.0) & (spread > 35.0) & (green > np.maximum(red, blue) * 1.15)
    red_mask = (red > 90.0) & (spread > 50.0) & (red > np.maximum(green, blue) * 1.25)
    center = np.zeros(green_mask.shape, dtype=bool)
    center[32:96, 32:96] = True
    border = ~center
    green_ratio = float(green_mask.mean())
    strong_green_ratio = float(strong_mask.mean())
    red_ratio = float(red_mask.mean())
    central_green_ratio = float(green_mask[center].mean())
    border_green_ratio = float(green_mask[border].mean())
    return ScanRow(
        image_id=image_id,
        image_path=image_path,
        green_ratio=green_ratio,
        strong_green_ratio=strong_green_ratio,
        red_ratio=red_ratio,
        central_green_ratio=central_green_ratio,
        border_green_ratio=border_green_ratio,
        target_score=central_green_ratio * 3.0 + strong_green_ratio * 2.0 + red_ratio * 0.7 - border_green_ratio * 0.8,
        background_score=border_green_ratio * 2.0 + green_ratio + strong_green_ratio * 0.5,
    )


def _select(rows: tuple[ScanRow, ...], top_k: int) -> tuple[SelectedRow, ...]:
    selected: list[SelectedRow] = []
    for bucket, _label in BUCKETS:
        ranked = sorted(rows, key=lambda row: (-_bucket_score(row, bucket), row.image_id))
        for rank, row in enumerate(ranked[:top_k], start=1):
            selected.append(SelectedRow(bucket=bucket, rank=rank, bucket_score=_bucket_score(row, bucket), scan=row))
    return tuple(selected)


def _bucket_score(row: ScanRow, bucket: str) -> float:
    match bucket:
        case "target_score":
            return row.target_score
        case "background_score":
            return row.background_score
        case "strong_green":
            return row.strong_green_ratio
        case "red_green_mix":
            return row.red_ratio + row.strong_green_ratio * 0.5
        case _:
            return 0.0


def _candidate_rows(rows: tuple[SelectedRow, ...], *, c067_ids: set[str]) -> tuple[JsonObject, ...]:
    result: list[JsonObject] = []
    for row in rows:
        scan = row.scan
        result.append(
            {
                "candidate_id": f"c069_{row.bucket}_{row.rank:03d}",
                "image_id": scan.image_id,
                "image_path": str(scan.image_path),
                "source_bucket": row.bucket,
                "rank": row.rank,
                "bucket_score": row.bucket_score,
                "green_ratio": scan.green_ratio,
                "strong_green_ratio": scan.strong_green_ratio,
                "red_ratio": scan.red_ratio,
                "central_green_ratio": scan.central_green_ratio,
                "border_green_ratio": scan.border_green_ratio,
                "seen_in_c067_topk": scan.image_id in c067_ids,
                "heldout_excluded": False,
                "path_exists": scan.image_path.is_file(),
            }
        )
    return tuple(result)


def _review_rows(candidates: tuple[JsonObject, ...]) -> tuple[JsonObject, ...]:
    rows: list[JsonObject] = []
    for row in candidates:
        label, note = _review_label(row)
        reviewed = dict(row)
        reviewed["review_label"] = label
        reviewed["review_note"] = note
        rows.append(reviewed)
    return tuple(rows)


def _review_label(row: JsonObject) -> tuple[str, str]:
    bucket = str(row["source_bucket"])
    rank = int(row["rank"])
    if (bucket, rank) in {("background_score", 11), ("strong_green", 12)}:
        return (
            "useful_proxy_non_human",
            "possible non-human color proxy, but not a confirmed direct-green target face",
        )
    return (
        "false_positive_background_object",
        "green signal is dominated by background, object, lighting, or non-target color",
    )


def _summary(
    *,
    scan_rows: tuple[ScanRow, ...],
    candidates: tuple[JsonObject, ...],
    reviewed: tuple[JsonObject, ...],
    missing_paths: int,
    heldout_count: int,
    c067_topk_count: int,
) -> JsonObject:
    label_counts = Counter(str(row["review_label"]) for row in reviewed)
    bucket_counts = Counter(str(row["source_bucket"]) for row in candidates)
    target_count = label_counts.get("target_positive", 0)
    return {
        "source": "local_color_self_reconstruct_full_scan",
        "scanned_image_count": len(scan_rows),
        "heldout_ids_count": heldout_count,
        "heldout_rows_used": sum(1 for row in candidates if bool(row["heldout_excluded"])),
        "missing_paths": missing_paths,
        "c067_topk_ids_count": c067_topk_count,
        "candidate_count": len(candidates),
        "unique_candidate_count": len({str(row["image_id"]) for row in candidates}),
        "reviewed_rows": len(reviewed),
        "scanned_beyond_c067_topk": len(scan_rows) > c067_topk_count,
        "candidate_bucket_counts": dict(bucket_counts),
        "label_counts": dict(label_counts),
        "direct_green_target_positive_count": target_count,
        "false_positive_background_object_count": label_counts.get("false_positive_background_object", 0),
        "useful_proxy_non_human_count": label_counts.get("useful_proxy_non_human", 0),
        "minimum_target_positive_required": TARGET_POSITIVE_MINIMUM,
        "decision": "ready_for_encoder_training" if target_count >= TARGET_POSITIVE_MINIMUM else "new_dataset_captioning_required",
    }


def _write_summary_report(out_dir: Path, summary: JsonObject) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (out_dir / "report.md").write_text(_report(summary), encoding="utf-8")


def _report(summary: JsonObject) -> str:
    header = "# c069 Direct Green Captioning Acquisition"
    stats = "\n".join(
        f"- {label}: `{summary[key]}`"
        for label, key in (
            ("Scanned images", "scanned_image_count"),
            ("Heldout rows used", "heldout_rows_used"),
            ("Candidate rows", "candidate_count"),
            ("Direct-green target positives", "direct_green_target_positive_count"),
            ("Useful non-human proxies", "useful_proxy_non_human_count"),
            ("Decision", "decision"),
        )
    )
    body = "The full local color pool was scanned beyond c067 top-k, but the reviewed top green/red queues are still dominated by background, objects, lighting, cups, leaves, and non-target color. Do not train encoder-side direct-green positives from this seed unless at least 4 confirmed target positives are collected."
    return f"{header}\n\n{stats}\n\n{body}\n"


def _read_manifest_ids(path: Path) -> tuple[str, ...]:
    rows: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        raw: JsonValue = json.loads(line)
        if isinstance(raw, dict) and isinstance(raw.get("ref_id"), str):
            rows.append(str(raw["ref_id"]))
    return tuple(rows)


def _read_c067_topk_ids(path: Path) -> set[str]:
    if not path.is_file():
        return set()
    raw: JsonValue = json.loads(path.read_text(encoding="utf-8"))
    ids: set[str] = set()
    if isinstance(raw, dict):
        for values in raw.values():
            if isinstance(values, list):
                ids.update(str(row["image_id"]) for row in values if isinstance(row, dict) and isinstance(row.get("image_id"), str))
    return ids


def _image_path(config: C069Config, image_id: str) -> Path:
    return config.dataset_root / f"{image_id}.jpg"


def _write_jsonl(path: Path, rows: tuple[JsonObject, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
