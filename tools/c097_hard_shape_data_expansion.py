# /// script
# dependencies = ["pillow"]
# ///
# --- How to run -----------------------------------------------------
# PYTHONPATH=. /home/wktwin/anima-lora-training-bundle/anima_lora/.venv/bin/python \
#   tools/c097_hard_shape_data_expansion.py

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Final

from PIL import Image

from tools.c079_manifest_io import read_jsonl, write_jsonl
from tools.c097_hard_shape_report import c097_report, write_c097_summary
from tools.c097_hard_shape_review import write_c097_review_sheet
from tools.siglip_auto_caption_types import JsonObject


SOURCE_PREFIX: Final = "external/c084_sheet_crop_pairs/"
OUTPUT_PREFIX: Final = "c097_hard_shape"
POSES: Final = ("three_quarter", "front", "profile", "action")
DEFAULT_SOURCE_MANIFEST: Final = Path("training/manifests/c087_expanded_crop_pairs_20260613.jsonl")
DEFAULT_SOURCE_ROOT: Final = Path(".tmp/c087_expanded_crop_pairs_root")
DEFAULT_OUTPUT_ROOT: Final = Path(".tmp/c097_siglip_hard_shape_expanded_root")
DEFAULT_OUTPUT_MANIFEST: Final = Path(
    "training/manifests/c097_siglip_hard_shape_expanded_pairs_20260613.jsonl"
)
DEFAULT_OUTPUT_SUMMARY: Final = Path(
    "training/manifests/c097_siglip_hard_shape_expanded_pairs_20260613.summary.json"
)
DEFAULT_EVAL_DIR: Final = Path("eval/c097_hard_shape_data_expansion_20260613")


class C097ExpansionError(Exception):
    def __init__(self, detail: str) -> None:
        self.detail = detail
        super().__init__(detail)

    def __str__(self) -> str:
        return self.detail


@dataclass(frozen=True, slots=True)
class C097ExpansionConfig:
    source_manifest: Path = DEFAULT_SOURCE_MANIFEST
    source_root: Path = DEFAULT_SOURCE_ROOT
    output_root: Path = DEFAULT_OUTPUT_ROOT
    output_manifest: Path = DEFAULT_OUTPUT_MANIFEST
    output_summary: Path = DEFAULT_OUTPUT_SUMMARY
    review_sheet: Path = DEFAULT_EVAL_DIR / "pair_review_sheet.jpg"
    report_path: Path = DEFAULT_EVAL_DIR / "report.md"
    max_rows_per_group: int = 16
    max_rows_per_source_pose_pair: int = 8
    minimum_groups: int = 4
    minimum_rows: int = 48


@dataclass(frozen=True, slots=True)
class ImageMeta:
    image_id: str
    group: str
    pose: str


@dataclass(frozen=True, slots=True)
class SourcePair:
    ref: ImageMeta
    tgt: ImageMeta
    prompt: str
    source_pose_pair: str


@dataclass(frozen=True, slots=True)
class C097ExpansionSummary:
    source_manifest: str
    source_root: str
    output_root: str
    output_manifest: str
    output_summary: str
    review_sheet: str
    report_path: str
    source_rows: int
    usable_positive_rows: int
    selected_rows: int
    explicit_negative_rows: int
    materialized_image_count: int
    heldout_rows_rejected: int
    heldout_rows_used: int
    cross_group_rows_rejected: int
    selected_group_counts: dict[str, int]
    selected_source_pose_pair_counts: dict[str, int]
    max_rows_per_group: int
    max_rows_per_source_pose_pair: int
    minimum_groups: int
    minimum_rows: int
    raw_crop_images_committed: bool
    training_started: bool
    decision: str


def build_c097_hard_shape_expansion(config: C097ExpansionConfig = C097ExpansionConfig()) -> C097ExpansionSummary:
    _validate_config(config)
    pairs, source_rows, heldout_rejected, cross_group_rejected = _load_pairs(config)
    selected = _select_pairs(pairs, config)
    group_counts = _group_counts(selected)
    decision = _decision(selected, group_counts, config)
    if decision != "data_gate_pass_for_deeper_siglip_encoder_training":
        raise C097ExpansionError(f"c097 data gate failed: {decision}")
    manifest_rows = _materialize_rows(selected, config)
    write_jsonl(config.output_manifest, tuple(manifest_rows))
    write_c097_review_sheet(manifest_rows, config.output_root, config.review_sheet)
    summary = C097ExpansionSummary(
        source_manifest=str(config.source_manifest),
        source_root=str(config.source_root),
        output_root=str(config.output_root),
        output_manifest=str(config.output_manifest),
        output_summary=str(config.output_summary),
        review_sheet=str(config.review_sheet),
        report_path=str(config.report_path),
        source_rows=source_rows,
        usable_positive_rows=len(pairs),
        selected_rows=len(selected),
        explicit_negative_rows=len(manifest_rows),
        materialized_image_count=len(manifest_rows) * 3,
        heldout_rows_rejected=heldout_rejected,
        heldout_rows_used=0,
        cross_group_rows_rejected=cross_group_rejected,
        selected_group_counts=group_counts,
        selected_source_pose_pair_counts=_pose_pair_counts(selected),
        max_rows_per_group=config.max_rows_per_group,
        max_rows_per_source_pose_pair=config.max_rows_per_source_pose_pair,
        minimum_groups=config.minimum_groups,
        minimum_rows=config.minimum_rows,
        raw_crop_images_committed=False,
        training_started=False,
        decision=decision,
    )
    summary_json = asdict(summary)
    write_c097_summary(config.output_summary, summary_json)
    config.report_path.parent.mkdir(parents=True, exist_ok=True)
    config.report_path.write_text(c097_report(summary_json), encoding="utf-8")
    return summary


def _load_pairs(config: C097ExpansionConfig) -> tuple[tuple[SourcePair, ...], int, int, int]:
    pairs: list[SourcePair] = []
    heldout_rejected = 0
    cross_group_rejected = 0
    source_rows = 0
    for line_number, row in read_jsonl(config.source_manifest):
        source_rows += 1
        ref_id = _string(row, "ref_id", config.source_manifest, line_number)
        tgt_id = _string(row, "tgt_id", config.source_manifest, line_number)
        if "heldout" in ref_id or "heldout" in tgt_id:
            heldout_rejected += 1
            continue
        ref = _parse_meta(ref_id)
        tgt = _parse_meta(tgt_id)
        if ref.group != tgt.group:
            cross_group_rejected += 1
            continue
        _require_image(config.source_root, ref.image_id)
        _require_image(config.source_root, tgt.image_id)
        pairs.append(SourcePair(ref, tgt, _string(row, "prompt", config.source_manifest, line_number), f"{ref.pose}->{tgt.pose}"))
    if not pairs:
        raise C097ExpansionError("no usable c097 source pairs")
    return tuple(pairs), source_rows, heldout_rejected, cross_group_rejected


def _select_pairs(pairs: tuple[SourcePair, ...], config: C097ExpansionConfig) -> tuple[SourcePair, ...]:
    selected: list[SourcePair] = []
    group_counts: dict[str, int] = {}
    pose_counts: dict[str, int] = {}
    for pair in sorted(pairs, key=lambda item: (item.ref.group, item.source_pose_pair, item.ref.image_id, item.tgt.image_id)):
        group_count = group_counts.get(pair.ref.group, 0)
        pose_key = f"{pair.ref.group}:{pair.source_pose_pair}"
        pose_count = pose_counts.get(pose_key, 0)
        if group_count >= config.max_rows_per_group or pose_count >= config.max_rows_per_source_pose_pair:
            continue
        selected.append(pair)
        group_counts[pair.ref.group] = group_count + 1
        pose_counts[pose_key] = pose_count + 1
    if not selected:
        raise C097ExpansionError("no c097 rows selected")
    return tuple(selected)


def _materialize_rows(selected: tuple[SourcePair, ...], config: C097ExpansionConfig) -> list[JsonObject]:
    rows: list[JsonObject] = []
    for index, pair in enumerate(selected):
        negative = _negative_for(pair, selected, index)
        ids = _output_ids(index)
        _copy_rgb_jpg(config.source_root / f"{pair.ref.image_id}.jpg", config.output_root / f"{ids['ref_id']}.jpg")
        _copy_rgb_jpg(config.source_root / f"{pair.tgt.image_id}.jpg", config.output_root / f"{ids['tgt_id']}.jpg")
        _copy_rgb_jpg(config.source_root / f"{negative.tgt.image_id}.jpg", config.output_root / f"{ids['neg_id']}.jpg")
        (config.output_root / f"{ids['tgt_id']}.txt").write_text(pair.prompt + "\n", encoding="utf-8")
        rows.append(
            ids
            | {
                "prompt": pair.prompt,
                "shape_group": pair.ref.group,
                "source_pose_pair": pair.source_pose_pair,
                "negative_shape_group": negative.tgt.group,
                "source_ref_id": pair.ref.image_id,
                "source_tgt_id": pair.tgt.image_id,
                "negative_source_id": negative.tgt.image_id,
            }
        )
    return rows


def _negative_for(pair: SourcePair, selected: tuple[SourcePair, ...], index: int) -> SourcePair:
    pool = tuple(candidate for candidate in selected if candidate.tgt.group != pair.tgt.group)
    if not pool:
        raise C097ExpansionError("c097 needs at least two shape groups for negatives")
    return pool[index % len(pool)]


def _parse_meta(image_id: str) -> ImageMeta:
    if not image_id.startswith(SOURCE_PREFIX):
        raise C097ExpansionError(f"cannot parse hard-shape id: {image_id}")
    stem = image_id.removeprefix(SOURCE_PREFIX).removeprefix("c083_")
    if "_crop" not in stem:
        raise C097ExpansionError(f"cannot parse hard-shape id: {image_id}")
    base = stem.rsplit("_crop", 1)[0]
    for pose in POSES:
        suffix = f"_{pose}"
        if base.endswith(suffix):
            return ImageMeta(image_id=image_id, group=base.removesuffix(suffix), pose=pose)
    raise C097ExpansionError(f"cannot parse hard-shape id: {image_id}")


def _output_ids(index: int) -> JsonObject:
    label = f"{OUTPUT_PREFIX}/pair_{index:03d}"
    return {"ref_id": f"{label}_ref", "tgt_id": f"{label}_target", "neg_id": f"{label}_negative"}


def _require_image(root: Path, image_id: str) -> None:
    path = root / f"{image_id}.jpg"
    if not path.is_file():
        raise C097ExpansionError(f"missing source image: {path}")


def _copy_rgb_jpg(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(source) as image:
        image.convert("RGB").save(destination, quality=95)


def _decision(selected: tuple[SourcePair, ...], group_counts: dict[str, int], config: C097ExpansionConfig) -> str:
    if len(selected) < config.minimum_rows:
        return "blocked_not_enough_rows"
    if len(group_counts) < config.minimum_groups:
        return "blocked_not_enough_shape_groups"
    return "data_gate_pass_for_deeper_siglip_encoder_training"


def _group_counts(rows: tuple[SourcePair, ...]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        counts[row.ref.group] = counts.get(row.ref.group, 0) + 1
    return counts


def _pose_pair_counts(rows: tuple[SourcePair, ...]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        key = f"{row.ref.group}:{row.source_pose_pair}"
        counts[key] = counts.get(key, 0) + 1
    return counts


def _string(row: JsonObject, field: str, path: Path, line_number: int) -> str:
    value = row.get(field)
    if not isinstance(value, str):
        raise C097ExpansionError(f"{path}:{line_number} missing {field}")
    return value


def _validate_config(config: C097ExpansionConfig) -> None:
    if config.max_rows_per_group < 1:
        raise C097ExpansionError("max_rows_per_group must be >= 1")
    if config.max_rows_per_source_pose_pair < 1:
        raise C097ExpansionError("max_rows_per_source_pose_pair must be >= 1")
    if config.minimum_groups < 2:
        raise C097ExpansionError("minimum_groups must be >= 2")


def main() -> None:
    summary = build_c097_hard_shape_expansion()
    print(json.dumps(asdict(summary), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
