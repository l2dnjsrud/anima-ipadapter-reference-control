from __future__ import annotations

# /// script
# requires-python = ">=3.13"
# dependencies = ["numpy", "pillow", "typer"]
# ///
# ─── How to run ───
# PYTHONPATH=. python tools/build_c066_direct_green_candidates.py --help

import json
from dataclasses import asdict
from pathlib import Path
from typing import Annotated

import numpy as np
import typer
from PIL import Image

from tools.c066_candidate_types import (
    DIRECT_GREEN_KEYWORDS,
    FANG_PROFILE_KEYWORDS,
    HUMAN_KEYWORDS,
    OLD_HEADWEAR_KEYWORDS,
    PALE_KEYWORDS,
    RED_EYE_KEYWORDS,
    SIDECAR_KEYWORDS,
    C066CandidateRow,
    C066Config,
    C066InputError,
    C066PairRow,
    C066Summary,
    GreenMetrics,
    Label,
)
from tools.siglip_auto_caption_types import JsonObject, JsonValue


def build_c066_direct_green_candidates(config: C066Config) -> C066Summary:
    _read_manifest_ids(config.train_manifest_path, allow_empty=False)
    heldout_ids = set(_read_manifest_ids(config.heldout_manifest_path, allow_empty=True))
    attrs_by_id = _read_gate_attrs(config.gate_summary_path)
    attrs_by_id.update(_read_c065_attrs(config.c065_pair_manifest_path))
    rows = _build_candidate_rows(config, heldout_ids=heldout_ids, attrs_by_id=attrs_by_id)
    pairs = _build_pair_rows(rows)
    _write_jsonl(config.output_manifest_path, rows)
    _write_jsonl(config.output_pair_manifest_path, pairs)
    summary = C066Summary(
        total_candidates=len(rows),
        positive_candidates=_label_count(rows, "positive"),
        negative_candidates=_label_count(rows, "negative"),
        direct_green_positive_count=sum(1 for row in rows if row.source_bucket == "direct_green_attribute"),
        direct_green_pixel_candidate_count=sum(1 for row in rows if row.source_bucket == "direct_green_pixel_candidate"),
        non_human_positive_count=sum(1 for row in rows if row.label == "positive" and row.source_bucket != "direct_green_pixel_candidate"),
        heldout_rows_used=sum(1 for row in rows if row.image_id in heldout_ids),
        missing_paths=sum(1 for row in rows if not row.path_exists),
        sidecar_caption_keyword_hits=sum(1 for row in rows if row.candidate_source == "sidecar_caption"),
        source_buckets=_source_bucket_counts(rows),
        pair_rows=len(pairs),
    )
    config.output_summary_path.parent.mkdir(parents=True, exist_ok=True)
    config.output_summary_path.write_text(
        json.dumps(asdict(summary), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return summary


def _build_candidate_rows(config: C066Config, *, heldout_ids: set[str], attrs_by_id: dict[str, tuple[str, ...]]) -> tuple[C066CandidateRow, ...]:
    by_bucket: dict[str, list[C066CandidateRow]] = {}
    for image_path in sorted(config.dataset_root.rglob("*.jpg")):
        image_id = image_path.relative_to(config.dataset_root).with_suffix("").as_posix()
        if image_id in heldout_ids:
            continue
        caption_path = image_path.with_suffix(".txt")
        caption = caption_path.read_text(encoding="utf-8").strip() if caption_path.is_file() else ""
        attrs = attrs_by_id.get(image_id, ())
        metrics = _green_metrics(image_path)
        for row in _rows_for_image(config, image_id, image_path, caption_path, caption, attrs, metrics):
            by_bucket.setdefault(row.source_bucket, []).append(row)
    rows: list[C066CandidateRow] = []
    for bucket, bucket_rows in sorted(by_bucket.items()):
        rows.extend(bucket_rows[: config.max_per_bucket])
    return tuple(rows)


def _rows_for_image(
    config: C066Config,
    image_id: str,
    image_path: Path,
    caption_path: Path,
    caption: str,
    attrs: tuple[str, ...],
    metrics: GreenMetrics,
) -> tuple[C066CandidateRow, ...]:
    rows: list[C066CandidateRow] = []
    rows.extend(_attribute_rows(config, image_id, image_path, caption_path, caption, attrs, metrics))
    if metrics.green_ratio >= config.green_ratio_min and metrics.strong_green_ratio >= config.strong_green_ratio_min:
        rows.append(_row(image_id, "positive", "direct_green_pixel_candidate", "image_green_pixel_scan", (), image_path, caption_path, caption, attrs, metrics))
    sidecar_hits = _matched((caption,), SIDECAR_KEYWORDS)
    if sidecar_hits:
        rows.append(_row(image_id, "positive", "sidecar_attribute_candidate", "sidecar_caption", sidecar_hits, image_path, caption_path, caption, attrs, metrics))
    return tuple(rows)


def _attribute_rows(config: C066Config, image_id: str, image_path: Path, caption_path: Path, caption: str, attrs: tuple[str, ...], metrics: GreenMetrics) -> tuple[C066CandidateRow, ...]:
    buckets: tuple[tuple[str, Label, str, tuple[str, ...]], ...] = (
        ("direct_green_attribute", "positive", "qwen_attribute", DIRECT_GREEN_KEYWORDS),
        ("red_eye_proxy", "positive", "qwen_attribute", RED_EYE_KEYWORDS),
        ("pale_non_human_proxy", "positive", "qwen_attribute", PALE_KEYWORDS),
        ("fang_profile_proxy", "positive", "qwen_attribute", FANG_PROFILE_KEYWORDS),
        ("human_negative", "negative", "qwen_attribute", HUMAN_KEYWORDS),
        ("old_headwear_negative", "negative", "qwen_attribute", OLD_HEADWEAR_KEYWORDS),
    )
    rows: list[C066CandidateRow] = []
    for bucket, label, source, keywords in buckets:
        hits = _matched(attrs, keywords)
        if hits:
            rows.append(_row(image_id, label, bucket, source, hits, image_path, caption_path, caption, attrs, metrics))
    return tuple(rows)


def _row(image_id: str, label: Label, bucket: str, source: str, hits: tuple[str, ...], image_path: Path, caption_path: Path, caption: str, attrs: tuple[str, ...], metrics: GreenMetrics) -> C066CandidateRow:
    return C066CandidateRow(image_id, label, bucket, source, hits, attrs, caption, str(image_path), str(caption_path), metrics.green_ratio, metrics.strong_green_ratio, metrics.red_ratio, "train_or_dataset_scan", False, image_path.is_file())


def _build_pair_rows(rows: tuple[C066CandidateRow, ...]) -> tuple[C066PairRow, ...]:
    positives_by_bucket: dict[str, list[C066CandidateRow]] = {}
    negatives = [row for row in rows if row.label == "negative"]
    for row in rows:
        if row.label == "positive":
            positives_by_bucket.setdefault(row.source_bucket, []).append(row)
    pair_rows: list[C066PairRow] = []
    for bucket, positives in sorted(positives_by_bucket.items()):
        if len(positives) < 2 or not negatives:
            continue
        for index, anchor in enumerate(positives):
            positive = positives[(index + 1) % len(positives)]
            negative = negatives[index % len(negatives)]
            pair_rows.append(C066PairRow(f"c066_{bucket}_p{len(pair_rows):04d}", "positive", anchor.image_id, positive.image_id, bucket, bucket))
            pair_rows.append(C066PairRow(f"c066_{bucket}_n{len(pair_rows):04d}", "negative", anchor.image_id, negative.image_id, bucket, negative.source_bucket))
    return tuple(pair_rows)


def _green_metrics(image_path: Path) -> GreenMetrics:
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
    denom = float(arr.shape[0] * arr.shape[1])
    return GreenMetrics(float(green_mask.sum() / denom), float(strong_mask.sum() / denom), float(red_mask.sum() / denom))


def _read_manifest_ids(path: Path, *, allow_empty: bool) -> tuple[str, ...]:
    if not path.is_file():
        raise C066InputError(f"manifest not found: {path}")
    rows: list[str] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            raw: JsonValue = json.loads(line)
            if not isinstance(raw, dict):
                raise C066InputError(f"{path}:{line_number} row must be an object")
            value = raw.get("ref_id")
            if not isinstance(value, str):
                raise C066InputError(f"{path}:{line_number} missing ref_id")
            rows.append(_normalize_id(value))
    if not rows and not allow_empty:
        raise C066InputError(f"manifest has no rows: {path}")
    return tuple(rows)


def _read_gate_attrs(path: Path) -> dict[str, tuple[str, ...]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise C066InputError(f"gate summary must be an object: {path}")
    samples = raw.get("samples")
    attrs_by_id: dict[str, tuple[str, ...]] = {}
    if isinstance(samples, list):
        for sample in samples:
            if isinstance(sample, dict) and sample.get("split") != "heldout":
                _add_attrs(attrs_by_id, sample)
    return attrs_by_id


def _read_c065_attrs(path: Path | None) -> dict[str, tuple[str, ...]]:
    if path is None or not path.is_file():
        return {}
    attrs_by_id: dict[str, tuple[str, ...]] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            raw: JsonValue = json.loads(line)
            if isinstance(raw, dict):
                _add_pair_attrs(attrs_by_id, raw, "anchor_id", "anchor_attributes")
                _add_pair_attrs(attrs_by_id, raw, "candidate_id", "candidate_attributes")
    return attrs_by_id


def _add_attrs(attrs_by_id: dict[str, tuple[str, ...]], sample: JsonObject) -> None:
    ref_id = sample.get("ref_id")
    attrs = sample.get("selected_attributes")
    if isinstance(ref_id, str) and isinstance(attrs, list):
        attrs_by_id[_normalize_id(ref_id)] = tuple(str(attr) for attr in attrs)


def _add_pair_attrs(attrs_by_id: dict[str, tuple[str, ...]], row: JsonObject, id_field: str, attr_field: str) -> None:
    image_id = row.get(id_field)
    attrs = row.get(attr_field)
    if isinstance(image_id, str) and isinstance(attrs, list):
        attrs_by_id[_normalize_id(image_id)] = tuple(str(attr) for attr in attrs)


def _matched(values: tuple[str, ...], keywords: tuple[str, ...]) -> tuple[str, ...]:
    lowered = tuple(value.lower() for value in values)
    return tuple(keyword for keyword in keywords if any(keyword in value for value in lowered))


def _normalize_id(value: str) -> str:
    normalized = value.replace("\\", "/")
    return normalized[:-4] if normalized.lower().endswith(".jpg") else normalized


def _source_bucket_counts(rows: tuple[C066CandidateRow, ...]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        counts[row.source_bucket] = counts.get(row.source_bucket, 0) + 1
    return counts


def _label_count(rows: tuple[C066CandidateRow, ...], label: Label) -> int:
    return sum(1 for row in rows if row.label == label)


def _write_jsonl(path: Path, rows: tuple[C066CandidateRow, ...] | tuple[C066PairRow, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(asdict(row), ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


app = typer.Typer(add_completion=False)


@app.command()
def main(dataset_root: Annotated[Path, typer.Option()], train_manifest_path: Annotated[Path, typer.Option()], heldout_manifest_path: Annotated[Path, typer.Option()], gate_summary_path: Annotated[Path, typer.Option()], output_manifest_path: Annotated[Path, typer.Option()], output_summary_path: Annotated[Path, typer.Option()], output_pair_manifest_path: Annotated[Path, typer.Option()], c065_pair_manifest_path: Annotated[Path | None, typer.Option()] = None) -> None:
    summary = build_c066_direct_green_candidates(C066Config(dataset_root, train_manifest_path, heldout_manifest_path, gate_summary_path, c065_pair_manifest_path, output_manifest_path, output_summary_path, output_pair_manifest_path))
    typer.echo(json.dumps(asdict(summary), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    app()
