from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from tools.c079_manifest_io import write_jsonl
from tools.c088_probe_io import (
    DEFAULT_CROP_SUMMARY,
    DEFAULT_FULL_SUMMARY,
    DEFAULT_OUT_DIR,
    VARIANTS,
    C088ProbeError,
    count_field,
    ensure_image,
    json_list,
    json_object,
    read_json,
    validate_manifest_rows,
    write_json,
)
from tools.siglip_auto_caption_types import JsonObject


@dataclass(frozen=True, slots=True)
class C088BuildConfig:
    crop_summary_path: Path = DEFAULT_CROP_SUMMARY
    full_summary_path: Path = DEFAULT_FULL_SUMMARY
    output_dir: Path = DEFAULT_OUT_DIR
    crop_limit: int = 10
    heldout_labels: tuple[str, ...] = ("heldout07",)


def build_c088_probe_manifest(config: C088BuildConfig) -> JsonObject:
    config.output_dir.mkdir(parents=True, exist_ok=True)
    rows = _rows_from_summary(config.crop_summary_path, limit=config.crop_limit)
    heldout_rows = _rows_from_summary(
        config.full_summary_path,
        include_labels=config.heldout_labels,
    )
    all_rows = rows + heldout_rows
    validate_manifest_rows(all_rows)
    manifest_path = config.output_dir / "probe_manifest.jsonl"
    write_jsonl(manifest_path, all_rows)
    summary = {
        "experiment": "c088_shape_silhouette_feature_probe",
        "manifest_path": str(manifest_path),
        "rows": len(all_rows),
        "crop_focus_rows": len(rows),
        "heldout_eval_rows": len(heldout_rows),
        "heldout_training_rows_used": 0,
        "shape_group_counts": count_field(all_rows, "shape_group"),
        "candidate_variants": list(VARIANTS),
        "decision": "ready_for_c088_shape_silhouette_scoring",
    }
    write_json(config.output_dir / "summary.json", summary)
    return summary


def _rows_from_summary(
    summary_path: Path,
    *,
    limit: int | None = None,
    include_labels: tuple[str, ...] = (),
) -> tuple[JsonObject, ...]:
    summary = read_json(summary_path)
    samples = json_list(summary, "samples")
    selected = samples[:limit] if limit is not None else tuple(
        sample for sample in samples if str(sample.get("label")) in include_labels
    )
    data_root = Path(str(summary["data_root"]))
    results = json_object(summary, "results")
    return tuple(_manifest_row(sample, data_root, results, summary_path) for sample in selected)


def _manifest_row(
    sample: JsonObject,
    data_root: Path,
    results: JsonObject,
    summary_path: Path,
) -> JsonObject:
    label = str(sample["label"])
    ref_id = str(sample["ref_id"])
    return {
        "sample": label,
        "split": str(sample["split"]),
        "shape_group": _shape_group(label, ref_id),
        "failure_mode": _failure_mode(label, ref_id),
        "reference_path": str(_reference_path(data_root, ref_id)),
        "candidates": {
            variant: str(_result_path(results, label, variant, summary_path))
            for variant in VARIANTS
        },
        "source_summary_path": str(summary_path),
        "heldout_training_rows_used": 0,
    }


def _reference_path(data_root: Path, ref_id: str) -> Path:
    jpg = data_root / f"{ref_id}.jpg"
    if jpg.is_file():
        return jpg
    png = data_root / f"{ref_id}.png"
    if png.is_file():
        return png
    raise C088ProbeError(f"missing reference image for {ref_id} under {data_root}")


def _result_path(results: JsonObject, label: str, variant: str, summary_path: Path) -> Path:
    key = f"{label}_{variant}"
    raw = results.get(key)
    if not isinstance(raw, dict):
        raise C088ProbeError(f"{summary_path}: missing result {key}")
    image_path = Path(str(raw["image"]))
    ensure_image(image_path)
    return image_path


def _shape_group(label: str, ref_id: str) -> str:
    if "frog_yokai_guard" in ref_id or label.startswith("crop_pair"):
        return "frog_yokai_guard"
    if "green_oni" in ref_id:
        return "green_oni"
    if "jade_lizard" in ref_id:
        return "jade_lizard"
    if "heldout07" in label:
        return "heldout07_non_human_profile"
    return "unknown_shape_group"


def _failure_mode(label: str, ref_id: str) -> str:
    if label.startswith("crop_pair"):
        return "adult_green_humanoid_collapse_from_chibi_non_human_reference"
    if "heldout07" in label or "08754" in ref_id:
        return "human_villain_collapse_from_non_human_side_profile"
    return "shape_identity_collapse"
