# /// script
# dependencies = []
# ///
# --- How to run -----------------------------------------------------
# PYTHONPATH=. /home/wktwin/anima-lora-training-bundle/anima_lora/.venv/bin/python \
#   tools/c084_balanced_crop_pair_manifest.py

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Final

from tools.c075_tag_positive_manifest_types import PairRow
from tools.c079_manifest_io import link_or_copy, read_jsonl, write_jsonl
from tools.siglip_auto_caption_types import JsonObject


SOURCE_PREFIX: Final = "external/c083_sheet_crop_pairs/"
OUTPUT_PREFIX: Final = "external/c084_sheet_crop_pairs"
PAIR_PROMPT: Final = (
    "mrcolor_panel_style, full color manhwa comic panel, same direct-green "
    "non-human character identity, visible green skin, consistent face structure, "
    "clean webtoon coloring, cel shaded illustration, safe"
)


@dataclass(frozen=True, slots=True)
class C084ManifestError(Exception):
    detail: str

    def __str__(self) -> str:
        return self.detail


@dataclass(frozen=True, slots=True)
class C084ManifestConfig:
    reviewed_labels_path: Path
    approved_pairs_path: Path
    scratch_image_root: Path
    output_manifest_path: Path
    output_summary_path: Path
    output_report_path: Path
    max_pairs_per_group: int = 24
    max_pairs_per_source_pair: int = 8
    minimum_groups: int = 4


@dataclass(frozen=True, slots=True)
class C084ManifestSummary:
    reviewed_labels_path: str
    approved_pairs_path: str
    scratch_image_root: str
    output_manifest_path: str
    output_summary_path: str
    source_pairs: int
    selected_rows: int
    approved_groups: int
    target_positive_candidates: int
    materialized_candidate_count: int
    direct_self_pair_rows: int
    same_source_pair_rows: int
    heldout_rows_used: int
    max_pairs_per_group: int
    max_pairs_per_source_pair: int
    selected_group_counts: dict[str, int]
    selected_source_pair_counts: dict[str, int]
    raw_crop_images_committed: bool
    decision: str


def build_c084_balanced_crop_pair_manifest(
    config: C084ManifestConfig,
) -> C084ManifestSummary:
    _validate_config(config)
    reviewed = _target_positive_candidates(config.reviewed_labels_path)
    source_pairs = tuple(row for _line, row in read_jsonl(config.approved_pairs_path))
    selected = _select_pairs(source_pairs, reviewed, config)
    if len(_selected_group_counts(selected)) < config.minimum_groups:
        raise C084ManifestError("selected pairs do not cover the minimum group count")
    materialized = _materialize_selected(selected, reviewed, config.scratch_image_root)
    rows = tuple(_output_pair(row) for row in selected)
    write_jsonl(config.output_manifest_path, tuple(row.to_json() for row in rows))
    summary = _summary(config, reviewed, source_pairs, selected, materialized)
    _write_summary(config.output_summary_path, summary)
    config.output_report_path.parent.mkdir(parents=True, exist_ok=True)
    config.output_report_path.write_text(_report(summary), encoding="utf-8")
    return summary


def _target_positive_candidates(path: Path) -> dict[str, JsonObject]:
    rows: dict[str, JsonObject] = {}
    for line_number, row in read_jsonl(path):
        if row.get("manual_label") != "target_positive":
            continue
        candidate = _string(row, "candidate_id", path, line_number)
        local_path = Path(_string(row, "local_image_path", path, line_number))
        if not local_path.is_file():
            raise C084ManifestError(f"missing crop image: {local_path}")
        rows[candidate] = row
    if not rows:
        raise C084ManifestError(f"no target-positive crops: {path}")
    return rows


def _select_pairs(
    pairs: tuple[JsonObject, ...],
    reviewed: dict[str, JsonObject],
    config: C084ManifestConfig,
) -> tuple[JsonObject, ...]:
    selected: list[JsonObject] = []
    group_counts: dict[str, int] = {}
    source_pair_counts: dict[str, int] = {}
    for row in sorted(pairs, key=_pair_sort_key):
        ref = _candidate_from_id(_string(row, "ref_id", config.approved_pairs_path, 0))
        tgt = _candidate_from_id(_string(row, "tgt_id", config.approved_pairs_path, 0))
        if ref == tgt or ref not in reviewed or tgt not in reviewed:
            continue
        group = _string(row, "group_id", config.approved_pairs_path, 0)
        if group_counts.get(group, 0) >= config.max_pairs_per_group:
            continue
        source_key = _source_pair_key(reviewed[ref], reviewed[tgt])
        if source_key is None:
            continue
        if source_pair_counts.get(source_key, 0) >= config.max_pairs_per_source_pair:
            continue
        group_counts[group] = group_counts.get(group, 0) + 1
        source_pair_counts[source_key] = source_pair_counts.get(source_key, 0) + 1
        selected.append(row)
    if not selected:
        raise C084ManifestError("no c084 pairs selected")
    return tuple(selected)


def _source_pair_key(ref_row: JsonObject, tgt_row: JsonObject) -> str | None:
    ref_source = ref_row.get("source_candidate_id")
    tgt_source = tgt_row.get("source_candidate_id")
    if not isinstance(ref_source, str) or not isinstance(tgt_source, str):
        raise C084ManifestError("reviewed crop row missing source_candidate_id")
    if ref_source == tgt_source:
        return None
    left, right = sorted((ref_source, tgt_source))
    return f"{left}::{right}"


def _materialize_selected(
    selected: tuple[JsonObject, ...],
    reviewed: dict[str, JsonObject],
    scratch_root: Path,
) -> int:
    candidates = sorted(
        {
            _candidate_from_id(str(row[field]))
            for row in selected
            for field in ("ref_id", "tgt_id")
        }
    )
    for candidate in candidates:
        source = Path(str(reviewed[candidate]["local_image_path"]))
        image_id = _output_image_id(candidate)
        link_or_copy(source, scratch_root / f"{image_id}.jpg")
        (scratch_root / f"{image_id}.txt").write_text(PAIR_PROMPT + "\n", encoding="utf-8")
    return len(candidates)


def _output_pair(row: JsonObject) -> PairRow:
    return PairRow(
        ref_id=_output_image_id(_candidate_from_id(str(row["ref_id"]))),
        tgt_id=_output_image_id(_candidate_from_id(str(row["tgt_id"]))),
        prompt=PAIR_PROMPT,
    )


def _summary(
    config: C084ManifestConfig,
    reviewed: dict[str, JsonObject],
    source_pairs: tuple[JsonObject, ...],
    selected: tuple[JsonObject, ...],
    materialized: int,
) -> C084ManifestSummary:
    source_pair_counts = _selected_source_pair_counts(selected, reviewed)
    group_counts = _selected_group_counts(selected)
    return C084ManifestSummary(
        reviewed_labels_path=str(config.reviewed_labels_path),
        approved_pairs_path=str(config.approved_pairs_path),
        scratch_image_root=str(config.scratch_image_root),
        output_manifest_path=str(config.output_manifest_path),
        output_summary_path=str(config.output_summary_path),
        source_pairs=len(source_pairs),
        selected_rows=len(selected),
        approved_groups=len(group_counts),
        target_positive_candidates=len(reviewed),
        materialized_candidate_count=materialized,
        direct_self_pair_rows=sum(1 for row in selected if row["ref_id"] == row["tgt_id"]),
        same_source_pair_rows=0,
        heldout_rows_used=0,
        max_pairs_per_group=config.max_pairs_per_group,
        max_pairs_per_source_pair=config.max_pairs_per_source_pair,
        selected_group_counts=group_counts,
        selected_source_pair_counts=source_pair_counts,
        raw_crop_images_committed=False,
        decision="ready_for_c084_bounded_qwenvl_training",
    )


def _selected_group_counts(selected: tuple[JsonObject, ...]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in selected:
        group = str(row["group_id"])
        counts[group] = counts.get(group, 0) + 1
    return counts


def _selected_source_pair_counts(
    selected: tuple[JsonObject, ...],
    reviewed: dict[str, JsonObject],
) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in selected:
        key = _source_pair_key(reviewed[_candidate_from_id(str(row["ref_id"]))], reviewed[_candidate_from_id(str(row["tgt_id"]))])
        if key is not None:
            counts[key] = counts.get(key, 0) + 1
    return counts


def _candidate_from_id(image_id: str) -> str:
    if not image_id.startswith(SOURCE_PREFIX):
        raise C084ManifestError(f"unexpected c083 image id: {image_id}")
    return image_id.removeprefix(SOURCE_PREFIX)


def _output_image_id(candidate: str) -> str:
    return f"{OUTPUT_PREFIX}/{candidate}"


def _pair_sort_key(row: JsonObject) -> tuple[str, str, str]:
    return (str(row.get("group_id", "")), str(row.get("ref_id", "")), str(row.get("tgt_id", "")))


def _string(row: JsonObject, field: str, path: Path, line_number: int) -> str:
    value = row.get(field)
    if not isinstance(value, str):
        suffix = f":{line_number}" if line_number else ""
        raise C084ManifestError(f"{path}{suffix} missing {field}")
    return value


def _validate_config(config: C084ManifestConfig) -> None:
    if config.max_pairs_per_group < 1:
        raise C084ManifestError("max_pairs_per_group must be >= 1")
    if config.max_pairs_per_source_pair < 1:
        raise C084ManifestError("max_pairs_per_source_pair must be >= 1")
    if config.minimum_groups < 1:
        raise C084ManifestError("minimum_groups must be >= 1")


def _write_summary(path: Path, summary: C084ManifestSummary) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(summary), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _report(summary: C084ManifestSummary) -> str:
    fields = (
        ("decision", summary.decision),
        ("manifest", summary.output_manifest_path),
        ("scratch_image_root", summary.scratch_image_root),
        ("source_pairs", summary.source_pairs),
        ("selected_rows", summary.selected_rows),
        ("approved_groups", summary.approved_groups),
        ("target_positive_candidates", summary.target_positive_candidates),
        ("materialized_candidate_count", summary.materialized_candidate_count),
        ("direct_self_pair_rows", summary.direct_self_pair_rows),
        ("same_source_pair_rows", summary.same_source_pair_rows),
        ("heldout_rows_used", summary.heldout_rows_used),
        ("max_pairs_per_group", summary.max_pairs_per_group),
        ("max_pairs_per_source_pair", summary.max_pairs_per_source_pair),
        ("raw_crop_images_committed", summary.raw_crop_images_committed),
    )
    lines = ["# c084 balanced crop-pair manifest", ""]
    lines.extend(f"- {name}: `{value}`" for name, value in fields)
    return "\n".join((*lines, ""))


def _default_config() -> C084ManifestConfig:
    return C084ManifestConfig(
        reviewed_labels_path=Path("eval/c083_sheet_crop_identity_pair_extraction_20260613/reviewed_crop_labels.jsonl"),
        approved_pairs_path=Path("eval/c083_sheet_crop_identity_pair_extraction_20260613/approved_pair_manifest.jsonl"),
        scratch_image_root=Path(".tmp/c084_balanced_crop_pairs_root"),
        output_manifest_path=Path("training/manifests/c084_balanced_crop_pairs_20260613.jsonl"),
        output_summary_path=Path("training/manifests/c084_balanced_crop_pairs_20260613.summary.json"),
        output_report_path=Path("eval/qwenvl_c084_balanced_crop_pair_training_20260613/manifest_report.md"),
    )


def main() -> None:
    summary = build_c084_balanced_crop_pair_manifest(_default_config())
    print(json.dumps(asdict(summary), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
