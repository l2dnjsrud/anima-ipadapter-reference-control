from __future__ import annotations

# /// script
# requires-python = ">=3.13"
# dependencies = ["pillow", "typer"]
# ///
# ─── How to run ───
# PYTHONPATH=. python tools/c100_local_positive_gate.py

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Final

import typer

from tools.c100_local_positive_text import build_plan, build_report
from tools.c100_review_sheet import write_c100_review_sheet
from tools.siglip_auto_caption_types import JsonObject, JsonValue

ROOT: Final = Path(__file__).resolve().parents[1]
OUT_DIR: Final = ROOT / "eval/c100_local_real_color_positive_acquisition_20260613"
CANDIDATE_BUCKETS: Final = (
    "direct_green_pixel_candidate",
    "pale_non_human_proxy",
    "fang_profile_proxy",
    "red_eye_proxy",
    "sidecar_attribute_candidate",
)


@dataclass(frozen=True, slots=True)
class C100Config:
    c066_manifest: Path = ROOT / "training/manifests/c066_direct_green_non_human_candidates_20260612.jsonl"
    c066_summary: Path = ROOT / "training/manifests/c066_direct_green_non_human_candidates_20260612.summary.json"
    c099_summary: Path = ROOT / "eval/c099_real_color_reference_data_gate_20260613/c099_candidate_summary.json"
    c099_inventory: Path = ROOT / "eval/c099_real_color_reference_data_gate_20260613/inventory.json"
    heldout_manifest: Path = ROOT / "training/manifests/local_color_single_character_clean32_heldout8_20260611.jsonl"
    review_labels: Path = OUT_DIR / "reviewed_local_labels.jsonl"
    out_dir: Path = OUT_DIR
    plan_path: Path = ROOT / "docs/c100_local_real_color_positive_acquisition_plan_ko.md"
    min_reviewed_positive: int = 8
    max_candidates: int = 64


def build_c100_local_positive_package(config: C100Config = C100Config()) -> JsonObject:
    heldout_ids = _read_ids(config.heldout_manifest)
    labels = _read_review_labels(config.review_labels)
    rows = _candidate_rows(config, heldout_ids, labels)
    inventory = _inventory(config, heldout_ids)
    summary = _summary(config, rows)
    config.out_dir.mkdir(parents=True, exist_ok=True)
    _write_jsonl(config.out_dir / "c100_candidate_manifest.jsonl", rows)
    _write_json(config.out_dir / "source_inventory.json", inventory)
    _write_json(config.out_dir / "c100_candidate_summary.json", summary)
    config.plan_path.parent.mkdir(parents=True, exist_ok=True)
    config.plan_path.write_text(build_plan(inventory), encoding="utf-8")
    (config.out_dir / "c100_decision_report.md").write_text(
        build_report(summary, inventory),
        encoding="utf-8",
    )
    write_c100_review_sheet(rows, config.out_dir / "c100_candidate_review_sheet.jpg")
    return summary


def _candidate_rows(
    config: C100Config,
    heldout_ids: set[str],
    labels: dict[str, str],
) -> tuple[JsonObject, ...]:
    rows: list[JsonObject] = []
    for raw in _read_jsonl(config.c066_manifest):
        image_id = _str(raw, "image_id")
        bucket = _str(raw, "source_bucket")
        if image_id in heldout_ids or bucket not in CANDIDATE_BUCKETS:
            continue
        row = _candidate_row(raw, labels.get(image_id, "needs_review"))
        rows.append(row)
    rows.sort(key=_row_sort_key)
    return tuple(rows[: config.max_candidates])


def _candidate_row(raw: JsonObject, review_label: str) -> JsonObject:
    image_path = Path(_str(raw, "image_path"))
    return {
        "image_id": _str(raw, "image_id"),
        "source_type": "real_local_color",
        "source_bucket": _str(raw, "source_bucket"),
        "candidate_source": _str(raw, "candidate_source"),
        "label": _str(raw, "label"),
        "review_label": review_label,
        "review_status": "reviewed" if review_label != "needs_review" else "needs_review",
        "caption": _str(raw, "caption"),
        "image_path": str(image_path),
        "green_ratio": _float(raw, "green_ratio"),
        "strong_green_ratio": _float(raw, "strong_green_ratio"),
        "red_ratio": _float(raw, "red_ratio"),
        "paths_ok": image_path.is_file(),
    }


def _summary(config: C100Config, rows: tuple[JsonObject, ...]) -> JsonObject:
    missing = sum(1 for row in rows if not bool(row.get("paths_ok")))
    reviewed_positive = sum(1 for row in rows if row.get("review_label") == "local_positive")
    decision = "c101_training_greenlit" if reviewed_positive >= config.min_reviewed_positive and missing == 0 else "c101_blocked_needs_manual_annotation_or_teacher"
    return {
        "candidate_rows": len(rows),
        "local_real_candidate_rows": len(rows),
        "heldout_leakage_count": 0,
        "missing_path_count": missing,
        "reviewed_local_positive_count": reviewed_positive,
        "review_required_count": sum(1 for row in rows if row.get("review_label") == "needs_review"),
        "min_reviewed_positive": config.min_reviewed_positive,
        "source_bucket_counts": _counts(rows, "source_bucket"),
        "decision": decision,
        "blocker_reason": "" if decision == "c101_training_greenlit" else "candidate sheet is ready, but local real-color positives still need manual review or a stronger attribute teacher before C101 training.",
        "next_c101_command_surface": _next_command(decision),
    }


def _inventory(config: C100Config, heldout_ids: set[str]) -> JsonObject:
    c066 = _read_json(config.c066_summary)
    c099 = _read_json(config.c099_summary)
    return {
        "source_paths": {
            "c066_manifest": str(config.c066_manifest),
            "c066_summary": str(config.c066_summary),
            "c099_summary": str(config.c099_summary),
            "c099_inventory": str(config.c099_inventory),
            "heldout_manifest": str(config.heldout_manifest),
            "review_labels": str(config.review_labels),
        },
        "heldout_ids": sorted(heldout_ids),
        "candidate_buckets": list(CANDIDATE_BUCKETS),
        "greenlight_criteria": {
            "min_reviewed_positive": config.min_reviewed_positive,
            "heldout_leakage_count": 0,
            "missing_path_count": 0,
        },
        "key_metrics": {
            "c099_decision": str(c099.get("decision", "")),
            "c066_direct_green_positive_count": int(c066.get("direct_green_positive_count", 0)),
            "c066_total_candidates": int(c066.get("total_candidates", 0)),
            "c066_source_buckets": c066.get("source_buckets", {}),
            "heldout_count": len(heldout_ids),
            "min_reviewed_positive": config.min_reviewed_positive,
        },
    }


def _row_sort_key(row: JsonObject) -> tuple[int, float, float, str]:
    bucket = _str(row, "source_bucket")
    bucket_rank = CANDIDATE_BUCKETS.index(bucket) if bucket in CANDIDATE_BUCKETS else len(CANDIDATE_BUCKETS)
    return (bucket_rank, -_float(row, "strong_green_ratio"), -_float(row, "green_ratio"), _str(row, "image_id"))


def _next_command(decision: str) -> str:
    if decision == "c101_training_greenlit":
        return "PYTHONPATH=. python training/siglip_encoder_lora_contrastive.py --manifest-path eval/c100_local_real_color_positive_acquisition_20260613/c100_candidate_manifest.jsonl --steps <pilot>"
    return ""


def _read_review_labels(path: Path) -> dict[str, str]:
    return {_str(row, "image_id"): _str(row, "manual_label") for row in _read_jsonl(path)}


def _read_ids(path: Path) -> set[str]:
    return {_str(row, "ref_id") for row in _read_jsonl(path)}


def _read_jsonl(path: Path) -> tuple[JsonObject, ...]:
    if not path.is_file():
        return ()
    rows: list[JsonObject] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        raw: JsonValue = json.loads(line)
        if isinstance(raw, dict):
            rows.append(raw)
    return tuple(rows)


def _read_json(path: Path) -> JsonObject:
    if not path.is_file():
        return {}
    raw: JsonValue = json.loads(path.read_text(encoding="utf-8"))
    return raw if isinstance(raw, dict) else {}


def _write_jsonl(path: Path, rows: tuple[JsonObject, ...]) -> None:
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def _write_json(path: Path, payload: JsonObject) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _str(row: JsonObject, key: str) -> str:
    value = row.get(key)
    return value if isinstance(value, str) else ""


def _float(row: JsonObject, key: str) -> float:
    value = row.get(key)
    return float(value) if isinstance(value, int | float) else 0.0


def _counts(rows: tuple[JsonObject, ...], field: str) -> dict[str, int]:
    return dict(Counter(str(row.get(field, "unknown")) for row in rows))


app = typer.Typer(add_completion=False)


def config_for_cli(out_dir: Path) -> C100Config:
    return C100Config(
        out_dir=out_dir,
        review_labels=out_dir / "reviewed_local_labels.jsonl",
    )


@app.command()
def main(out_dir: Annotated[Path, typer.Option()] = OUT_DIR) -> None:
    summary = build_c100_local_positive_package(config_for_cli(out_dir))
    typer.echo(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    app()
