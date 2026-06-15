from __future__ import annotations

# /// script
# requires-python = ">=3.13"
# dependencies = ["numpy", "pillow"]
# ///
# How to run:
# PYTHONPATH=. python -m py_compile tools/c070_qwenvl_caption_search.py

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from tools.c070_color_metrics import c070_color_metrics
from tools.c069_review_sheet import write_c069_review_sheet
from tools.siglip_auto_caption_types import JsonObject, JsonValue

DIRECT_TERMS: Final = (
    "green-skinned",
    "green skin",
    "green monster",
    "green non-human",
    "green demon",
    "monster face",
    "creature face",
    "colored skin",
)
PROXY_TERMS: Final = ("red glowing", "red eye", "demonic eye", "non-human", "monster")
BUCKETS: Final = (
    ("caption_keyword_hit", "caption semantic hit"),
    ("semantic_target_fallback", "non-c069 target fallback"),
    ("red_green_proxy", "red/green proxy fallback"),
    ("background_green_guard", "green background guard"),
)
TARGET_POSITIVE_MINIMUM: Final = 4


@dataclass(frozen=True, slots=True)
class C070Config:
    dataset_root: Path
    all_manifest_path: Path
    heldout_manifest_path: Path
    c069_reviewed_path: Path
    out_dir: Path
    top_k_per_bucket: int = 12


@dataclass(frozen=True, slots=True)
class ScanRow:
    image_id: str
    image_path: Path
    caption: str
    caption_hits: tuple[str, ...]
    direct_caption_hits: tuple[str, ...]
    green_ratio: float
    strong_green_ratio: float
    red_ratio: float
    central_green_ratio: float
    border_green_ratio: float
    target_score: float
    background_score: float
    seen_in_c069: bool


@dataclass(frozen=True, slots=True)
class SelectedRow:
    bucket: str
    rank: int
    bucket_score: float
    scan: ScanRow


def build_c070_qwenvl_caption_search(config: C070Config) -> JsonObject:
    ids = _read_manifest_ids(config.all_manifest_path)
    heldout_ids = set(_read_manifest_ids(config.heldout_manifest_path))
    c069_ids = _read_image_ids(config.c069_reviewed_path)
    scan_rows = _scan_rows(config, ids, heldout_ids=heldout_ids, c069_ids=c069_ids)
    selected = _select(scan_rows, config.top_k_per_bucket)
    candidates = _candidate_rows(selected)
    reviewed = _review_rows(candidates)
    _write_jsonl(config.out_dir / "candidate_manifest.jsonl", candidates)
    _write_jsonl(config.out_dir / "reviewed_candidate_labels.jsonl", reviewed)
    write_c069_review_sheet(reviewed, config.out_dir / "annotated_review_sheet.jpg", BUCKETS)
    summary = _summary(scan_rows, candidates, reviewed, len(heldout_ids), len(c069_ids))
    _write_summary_report(config.out_dir, summary)
    return summary


def _scan_rows(
    config: C070Config,
    image_ids: tuple[str, ...],
    *,
    heldout_ids: set[str],
    c069_ids: set[str],
) -> tuple[ScanRow, ...]:
    rows: list[ScanRow] = []
    for image_id in image_ids:
        if image_id in heldout_ids:
            continue
        image_path = config.dataset_root / f"{image_id}.jpg"
        if not image_path.is_file():
            continue
        rows.append(_scan_image(image_id, image_path, c069_ids=c069_ids))
    return tuple(rows)


def _scan_image(image_id: str, image_path: Path, *, c069_ids: set[str]) -> ScanRow:
    caption = _caption(image_path.with_suffix(".txt"))
    hits = _hits(caption, DIRECT_TERMS + PROXY_TERMS)
    direct_hits = _hits(caption, DIRECT_TERMS)
    green_ratio, strong_green_ratio, red_ratio, central_green_ratio, border_green_ratio = c070_color_metrics(image_path)
    target_score = central_green_ratio * 3.0 + strong_green_ratio * 2.0 + red_ratio * 0.6 - border_green_ratio
    background_score = border_green_ratio * 2.0 + green_ratio + strong_green_ratio * 0.4
    return ScanRow(
        image_id=image_id,
        image_path=image_path,
        caption=caption,
        caption_hits=hits,
        direct_caption_hits=direct_hits,
        green_ratio=green_ratio,
        strong_green_ratio=strong_green_ratio,
        red_ratio=red_ratio,
        central_green_ratio=central_green_ratio,
        border_green_ratio=border_green_ratio,
        target_score=target_score,
        background_score=background_score,
        seen_in_c069=image_id in c069_ids,
    )


def _select(rows: tuple[ScanRow, ...], top_k: int) -> tuple[SelectedRow, ...]:
    selected: list[SelectedRow] = []
    for bucket, _label in BUCKETS:
        ranked = sorted(_bucket_pool(rows, bucket), key=lambda row: (-_score(row, bucket), row.image_id))
        for rank, row in enumerate(ranked[:top_k], start=1):
            selected.append(SelectedRow(bucket, rank, _score(row, bucket), row))
    return tuple(selected)


def _bucket_pool(rows: tuple[ScanRow, ...], bucket: str) -> tuple[ScanRow, ...]:
    if bucket == "caption_keyword_hit":
        return tuple(row for row in rows if row.caption_hits)
    return tuple(row for row in rows if not row.seen_in_c069)


def _score(row: ScanRow, bucket: str) -> float:
    if bucket == "caption_keyword_hit":
        return float(len(row.caption_hits)) + row.target_score
    if bucket == "semantic_target_fallback":
        return row.target_score
    if bucket == "red_green_proxy":
        return row.red_ratio + row.central_green_ratio * 0.5
    return row.background_score


def _candidate_rows(rows: tuple[SelectedRow, ...]) -> tuple[JsonObject, ...]:
    result: list[JsonObject] = []
    for row in rows:
        scan = row.scan
        result.append(
            {
                "candidate_id": f"c070_{row.bucket}_{row.rank:03d}",
                "image_id": scan.image_id,
                "image_path": str(scan.image_path),
                "caption": scan.caption,
                "caption_hits": list(scan.caption_hits),
                "direct_caption_hits": list(scan.direct_caption_hits),
                "source_bucket": row.bucket,
                "rank": row.rank,
                "bucket_score": row.bucket_score,
                "green_ratio": scan.green_ratio,
                "strong_green_ratio": scan.strong_green_ratio,
                "red_ratio": scan.red_ratio,
                "central_green_ratio": scan.central_green_ratio,
                "border_green_ratio": scan.border_green_ratio,
                "seen_in_c069_reviewed": scan.seen_in_c069,
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
    direct_hits = row.get("direct_caption_hits")
    if isinstance(direct_hits, list) and direct_hits:
        return "target_positive", "caption explicitly names green/non-human character attributes"
    if str(row["source_bucket"]) == "red_green_proxy":
        return "useful_proxy_non_human", "red/green visual proxy only; not confirmed direct-green target"
    if str(row["source_bucket"]) == "semantic_target_fallback":
        return "false_positive_human", "visual fallback lacks caption evidence and needs manual rejection"
    return "false_positive_background_object", "caption/search signal points to background/object or generic template"


def _summary(
    scan_rows: tuple[ScanRow, ...],
    candidates: tuple[JsonObject, ...],
    reviewed: tuple[JsonObject, ...],
    heldout_count: int,
    c069_count: int,
) -> JsonObject:
    labels = Counter(str(row["review_label"]) for row in reviewed)
    buckets = Counter(str(row["source_bucket"]) for row in candidates)
    target_count = len({str(row["image_id"]) for row in reviewed if row["review_label"] == "target_positive"})
    caption_hits = sum(1 for row in scan_rows if row.caption_hits)
    return {
        "source": "local_color_caption_search_plus_visual_fallback",
        "caption_signal_source": "sidecar_keyword_hits" if caption_hits else "sidecar_template_no_hits_fallback_visual_heuristics",
        "scanned_image_count": len(scan_rows),
        "heldout_ids_count": heldout_count,
        "heldout_rows_used": sum(1 for row in candidates if bool(row["heldout_excluded"])),
        "missing_paths": sum(1 for row in candidates if not bool(row["path_exists"])),
        "c069_reviewed_ids_count": c069_count,
        "c069_seed_excluded_or_annotated": "excluded_from_visual_fallback_buckets",
        "caption_keyword_hit_images": caption_hits,
        "candidate_count": len(candidates),
        "reviewed_rows": len(reviewed),
        "search_bucket_counts": dict(buckets),
        "label_counts": dict(labels),
        "direct_green_target_positive_count": target_count,
        "useful_proxy_non_human_count": labels.get("useful_proxy_non_human", 0),
        "false_positive_background_object_count": labels.get("false_positive_background_object", 0),
        "false_positive_human_count": labels.get("false_positive_human", 0),
        "minimum_target_positive_required": TARGET_POSITIVE_MINIMUM,
        "decision": "ready_for_encoder_training" if target_count >= TARGET_POSITIVE_MINIMUM else "external_manual_data_required",
    }


def _caption(path: Path) -> str:
    return path.read_text(encoding="utf-8").strip() if path.is_file() else ""


def _hits(text: str, terms: tuple[str, ...]) -> tuple[str, ...]:
    lowered = text.lower()
    return tuple(term for term in terms if term in lowered)


def _read_manifest_ids(path: Path) -> tuple[str, ...]:
    return tuple(str(row["ref_id"]) for row in _read_jsonl(path) if isinstance(row.get("ref_id"), str))


def _read_image_ids(path: Path) -> set[str]:
    if not path.is_file():
        return set()
    return {str(row["image_id"]) for row in _read_jsonl(path) if isinstance(row.get("image_id"), str)}


def _read_jsonl(path: Path) -> tuple[JsonObject, ...]:
    rows: list[JsonObject] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        raw: JsonValue = json.loads(line)
        if isinstance(raw, dict):
            rows.append(raw)
    return tuple(rows)


def _write_jsonl(path: Path, rows: tuple[JsonObject, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def _write_summary_report(out_dir: Path, summary: JsonObject) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (out_dir / "report.md").write_text(_report(summary), encoding="utf-8")


def _report(summary: JsonObject) -> str:
    return "\n".join(
        (
            "# c070 QwenVL Direct-Green Caption Search Acquisition",
            "",
            f"- Caption signal source: `{summary['caption_signal_source']}`",
            f"- Scanned images: `{summary['scanned_image_count']}`",
            f"- Heldout rows used: `{summary['heldout_rows_used']}`",
            f"- Candidate rows: `{summary['candidate_count']}`",
            f"- Direct-green target positives: `{summary['direct_green_target_positive_count']}`",
            f"- Useful non-human proxies: `{summary['useful_proxy_non_human_count']}`",
            f"- Decision: `{summary['decision']}`",
            "",
            "The local sidecar captions are audited before any training. If they remain template-only, c070 treats visual fallback queues as review evidence, not clean positives.",
            "",
        )
    )
