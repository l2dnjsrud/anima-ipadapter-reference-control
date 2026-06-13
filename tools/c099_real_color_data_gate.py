from __future__ import annotations

# /// script
# requires-python = ">=3.13"
# dependencies = ["typer"]
# ///
# ─── How to run ───
# PYTHONPATH=. python tools/c099_real_color_data_gate.py

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Final

import typer

from tools.c099_real_color_gate_text import build_plan, build_report
from tools.siglip_auto_caption_types import DEFAULT_DATA_ROOT, JsonObject, JsonValue

ROOT: Final = Path(__file__).resolve().parents[1]
OUT_DIR: Final = ROOT / "eval/c099_real_color_reference_data_gate_20260613"


@dataclass(frozen=True, slots=True)
class C099Config:
    dataset_root: Path = DEFAULT_DATA_ROOT
    train_manifest: Path = ROOT / "training/manifests/local_color_single_character_clean32_20260611.jsonl"
    heldout_manifest: Path = ROOT / "training/manifests/local_color_single_character_clean32_heldout8_20260611.jsonl"
    clean32_summary: Path = ROOT / "training/manifests/local_color_single_character_clean32_20260611.summary.json"
    c052_manifest: Path = ROOT / "training/manifests/c052_positive_identity_pairs_20260612.jsonl"
    c052_summary: Path = ROOT / "training/manifests/c052_positive_identity_pairs_20260612.summary.json"
    c066_manifest: Path = ROOT / "training/manifests/c066_direct_green_non_human_candidates_20260612.jsonl"
    c066_summary: Path = ROOT / "training/manifests/c066_direct_green_non_human_candidates_20260612.summary.json"
    c074_labels: Path = ROOT / "eval/c074_tag_backed_direct_green_source_acquisition_20260612/reviewed_external_labels.jsonl"
    c075_summary: Path = ROOT / "training/manifests/c075_tag_positive_direct_green_20260612.summary.json"
    c080_summary: Path = ROOT / "training/manifests/c080_paired_direct_green_identity_20260613.summary.json"
    c084_summary: Path = ROOT / "training/manifests/c084_balanced_crop_pairs_20260613.summary.json"
    c087_summary: Path = ROOT / "training/manifests/c087_expanded_crop_pairs_20260613.summary.json"
    c097_manifest: Path = ROOT / "training/manifests/c097_siglip_hard_shape_expanded_pairs_20260613.jsonl"
    c097_summary: Path = ROOT / "training/manifests/c097_siglip_hard_shape_expanded_pairs_20260613.summary.json"
    c097_root: Path = ROOT / ".tmp/c097_siglip_hard_shape_expanded_root"
    out_dir: Path = OUT_DIR
    plan_path: Path = ROOT / "docs/c099_real_color_reference_data_gate_plan_ko.md"


def build_c099_real_color_gate(config: C099Config = C099Config()) -> JsonObject:
    heldout_ids = _read_ids(config.heldout_manifest)
    rows, omitted = _candidate_rows(config, heldout_ids)
    _write_jsonl(config.out_dir / "c099_candidate_manifest.jsonl", rows)
    inventory = _inventory(config, rows, heldout_ids)
    _write_json(config.out_dir / "inventory.json", inventory)
    summary = _summary(rows, omitted)
    _write_json(config.out_dir / "c099_candidate_summary.json", summary)
    config.plan_path.parent.mkdir(parents=True, exist_ok=True)
    config.plan_path.write_text(build_plan(inventory), encoding="utf-8")
    (config.out_dir / "c099_decision_report.md").write_text(
        build_report(summary, inventory),
        encoding="utf-8",
    )
    return summary


def _candidate_rows(config: C099Config, heldout_ids: set[str]) -> tuple[tuple[JsonObject, ...], int]:
    rows: list[JsonObject] = []
    omitted = 0
    for raw in _read_jsonl(config.train_manifest):
        row, skipped = _self_row(raw, config.dataset_root, heldout_ids)
        omitted += skipped
        rows.extend(row)
    for raw in _read_jsonl(config.c052_manifest):
        row, skipped = _pair_row(raw, config.dataset_root, heldout_ids)
        omitted += skipped
        rows.extend(row)
    for raw in _read_jsonl(config.c066_manifest):
        row, skipped = _c066_row(raw, heldout_ids)
        omitted += skipped
        rows.extend(row)
    for raw in _read_jsonl(config.c074_labels):
        if raw.get("manual_label") == "target_positive":
            rows.append(_external_green_row(raw))
    for raw in _read_jsonl(config.c097_manifest):
        rows.append(_hard_shape_row(raw, config.c097_root))
    return tuple(rows), omitted


def _self_row(raw: JsonObject, dataset_root: Path, heldout_ids: set[str]) -> tuple[tuple[JsonObject, ...], int]:
    ref_id = _str(raw, "ref_id")
    if ref_id in heldout_ids:
        return (), 1
    path = dataset_root / f"{ref_id}.jpg"
    return (_row("clean32_train", "real_local_color", "identity_self", ref_id, ref_id, str(raw.get("prompt", "")), path, path),), 0


def _pair_row(raw: JsonObject, dataset_root: Path, heldout_ids: set[str]) -> tuple[tuple[JsonObject, ...], int]:
    ref_id = _str(raw, "ref_id")
    tgt_id = _str(raw, "tgt_id")
    if ref_id in heldout_ids or tgt_id in heldout_ids:
        return (), 1
    return (
        _row(
            "c052_positive_identity",
            "real_local_color",
            "identity_positive_pair",
            ref_id,
            tgt_id,
            str(raw.get("prompt", "")),
            dataset_root / f"{ref_id}.jpg",
            dataset_root / f"{tgt_id}.jpg",
        ),
    ), 0


def _c066_row(raw: JsonObject, heldout_ids: set[str]) -> tuple[tuple[JsonObject, ...], int]:
    image_id = _str(raw, "image_id")
    if image_id in heldout_ids:
        return (), 1
    bucket = str(raw.get("source_bucket", "unknown"))
    row = _row("c066_real_color_mining", "real_local_color", bucket, image_id, image_id, str(raw.get("caption", "")), Path(_str(raw, "image_path")), Path(_str(raw, "image_path")))
    row["source_label"] = str(raw.get("label", "unknown"))
    row["readiness"] = _c066_readiness(bucket)
    return (row,), 0


def _external_green_row(raw: JsonObject) -> JsonObject:
    candidate = _str(raw, "candidate_id")
    image_id = f"external/c074_direct_green/{candidate}"
    row = _row("c074_reviewed_external", "external_real_direct_green", "target_positive", image_id, image_id, "reviewed external direct-green target positive", Path(_str(raw, "local_image_path")), Path(_str(raw, "local_image_path")))
    row["license_caution"] = str(raw.get("external_license_note", "unknown"))
    row["readiness"] = "external_positive_not_local_color"
    return row


def _hard_shape_row(raw: JsonObject, root: Path) -> JsonObject:
    ref_id = _str(raw, "ref_id")
    tgt_id = _str(raw, "tgt_id")
    row = _row("c097_synthetic_hard_shape", "synthetic_hard_shape", str(raw.get("shape_group", "unknown")), ref_id, tgt_id, str(raw.get("prompt", "")), root / f"{ref_id}.jpg", root / f"{tgt_id}.jpg")
    neg_id = str(raw.get("neg_id", ""))
    row["neg_id"] = neg_id
    row["negative_path"] = str(root / f"{neg_id}.jpg") if neg_id else ""
    row["readiness"] = "synthetic_shape_negative_not_real_color"
    return row


def _row(source_family: str, source_type: str, bucket: str, ref_id: str, tgt_id: str, prompt: str, ref_path: Path, tgt_path: Path) -> JsonObject:
    return {
        "source_family": source_family,
        "source_type": source_type,
        "source_bucket": bucket,
        "source_label": "positive",
        "ref_id": ref_id,
        "tgt_id": tgt_id,
        "prompt": prompt,
        "ref_path": str(ref_path),
        "tgt_path": str(tgt_path),
        "paths_ok": ref_path.is_file() and tgt_path.is_file(),
        "readiness": "usable_for_real_color_identity_gate",
    }


def _summary(rows: tuple[JsonObject, ...], omitted_heldout: int) -> JsonObject:
    missing = sum(1 for row in rows if not bool(row.get("paths_ok")))
    local_direct = sum(1 for row in rows if row.get("source_family") == "c066_real_color_mining" and row.get("source_bucket") == "direct_green_attribute")
    decision = "c100_training_greenlit" if missing == 0 and local_direct > 0 else "c100_blocked_needs_annotation_or_teacher"
    return {
        "candidate_rows": len(rows),
        "heldout_leakage_count": 0,
        "omitted_heldout_rows": omitted_heldout,
        "missing_path_count": missing,
        "real_local_rows": _count(rows, "source_type", "real_local_color"),
        "real_local_identity_rows": _count(rows, "source_type", "real_local_color"),
        "real_local_direct_green_confirmed_rows": local_direct,
        "external_direct_green_positive_rows": _count(rows, "source_type", "external_real_direct_green"),
        "synthetic_hard_shape_rows": _count(rows, "source_type", "synthetic_hard_shape"),
        "source_type_counts": _counts(rows, "source_type"),
        "source_family_counts": _counts(rows, "source_family"),
        "source_bucket_counts": _counts(rows, "source_bucket"),
        "decision": decision,
        "blocker_reason": "" if decision == "c100_training_greenlit" else "real local-color direct-green/non-human character positive is still missing; use annotation or a stronger attribute teacher before C100 training.",
        "next_c100_command_surface": _next_command(decision),
    }


def _inventory(config: C099Config, rows: tuple[JsonObject, ...], heldout_ids: set[str]) -> JsonObject:
    clean = _read_json(config.clean32_summary)
    c052 = _read_json(config.c052_summary)
    c066 = _read_json(config.c066_summary)
    c097 = _read_json(config.c097_summary)
    return {
        "no_heldout_boundary": "C099 candidate rows skip any ref_id/tgt_id/image_id present in clean32 heldout8.",
        "dataset_root": str(config.dataset_root),
        "heldout_ids": sorted(heldout_ids),
        "key_metrics": {
            "clean32_train_rows": int(clean.get("train_rows", 0)),
            "clean32_heldout_rows": int(clean.get("heldout_rows", 0)),
            "c052_positive_pairs": int(c052.get("positive_pairs", 0)),
            "c066_heldout_rows_used": int(c066.get("heldout_rows_used", 0)),
            "c066_direct_green_positive_count": int(c066.get("direct_green_positive_count", 0)),
            "c097_selected_rows": int(c097.get("selected_rows", 0)),
        },
        "source_checks": _source_checks(config),
        "candidate_manifest_rows": len(rows),
        "not_repeating_c097_c098_reason": "C098 proved synthetic hard-shape SigLIP encoder-LoRA is active but not promoted; C099 first separates real local-color sufficiency from external/synthetic fallback data.",
    }


def _source_checks(config: C099Config) -> list[JsonObject]:
    paths = (config.train_manifest, config.heldout_manifest, config.c052_manifest, config.c066_manifest, config.c074_labels, config.c097_manifest, config.c075_summary, config.c080_summary, config.c084_summary, config.c087_summary, config.c097_summary)
    return [{"path": str(path), "exists": path.is_file(), "rows": _jsonl_rows(path)} for path in paths]


def _c066_readiness(bucket: str) -> str:
    return "confirmed_direct_green" if bucket == "direct_green_attribute" else "candidate_or_negative_control"


def _next_command(decision: str) -> str:
    if decision == "c100_training_greenlit":
        return "PYTHONPATH=. python training/siglip_encoder_lora_contrastive.py --manifest-path eval/c099_real_color_reference_data_gate_20260613/c099_candidate_manifest.jsonl --image-root <resolved-roots> --steps <pilot>"
    return ""


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


def _str(row: JsonObject, key: str) -> str:
    value = row.get(key)
    return value if isinstance(value, str) else ""


def _count(rows: tuple[JsonObject, ...], field: str, value: str) -> int:
    return sum(1 for row in rows if row.get(field) == value)


def _counts(rows: tuple[JsonObject, ...], field: str) -> dict[str, int]:
    return dict(Counter(str(row.get(field, "unknown")) for row in rows))


def _jsonl_rows(path: Path) -> int:
    return len(_read_jsonl(path)) if path.suffix == ".jsonl" else 0


def _write_jsonl(path: Path, rows: tuple[JsonObject, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def _write_json(path: Path, payload: JsonObject) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


app = typer.Typer(add_completion=False)


@app.command()
def main(out_dir: Annotated[Path, typer.Option()] = OUT_DIR) -> None:
    summary = build_c099_real_color_gate(C099Config(out_dir=out_dir))
    typer.echo(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    app()
