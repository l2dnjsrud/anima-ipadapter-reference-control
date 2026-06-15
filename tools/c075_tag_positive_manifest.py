# /// script
# dependencies = []
# ///
# --- How to run -----------------------------------------------------
# PYTHONPATH=. /home/wktwin/anima-lora-training-bundle/anima_lora/.venv/bin/python \
#   tools/c075_tag_positive_manifest.py

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Final

from tools.c075_manifest_files import (
    external_image_id,
    materialize_external_rows,
    materialize_source_rows,
)
from tools.c075_tag_positive_manifest_types import (
    C075ManifestConfig,
    C075ManifestError,
    C075ManifestSummary,
    PairRow,
)
from tools.siglip_auto_caption_types import JsonObject, JsonValue


DEFAULT_SOURCE_MANIFEST: Final = Path(
    "training/manifests/c060_qwenvl_failure_focused_clean32_c052_20260612.jsonl"
)
DEFAULT_SOURCE_ROOT: Final = Path(
    "/home/wktwin/anima-lora-training-bundle/image_dataset_color_panel_style_v5_best"
)
DEFAULT_C074_LABELS: Final = Path(
    "eval/c074_tag_backed_direct_green_source_acquisition_20260612/"
    "reviewed_external_labels.jsonl"
)
DEFAULT_C073_LABELS: Final = Path(
    "eval/c073_external_candidate_visual_review_20260612/reviewed_external_labels.jsonl"
)
DEFAULT_CONFIG: Final = C075ManifestConfig(
    source_manifest_path=DEFAULT_SOURCE_MANIFEST,
    source_image_root=DEFAULT_SOURCE_ROOT,
    c074_labels_path=DEFAULT_C074_LABELS,
    c073_labels_path=DEFAULT_C073_LABELS,
    scratch_image_root=Path(".tmp/c075_tag_positive_direct_green_root"),
    output_manifest_path=Path(
        "training/manifests/c075_tag_positive_direct_green_20260612.jsonl"
    ),
    output_summary_path=Path(
        "training/manifests/c075_tag_positive_direct_green_20260612.summary.json"
    ),
    output_report_path=Path(
        "eval/qwenvl_c075_tag_positive_training_20260612/manifest_report.md"
    ),
)
EXTERNAL_PROMPT: Final = (
    "mrcolor_panel_style, full color manhwa comic panel, solo non-human anime "
    "character portrait, visible green skin, monster girl, tail, clean webtoon "
    "coloring, cel shaded illustration, safe"
)


def build_c075_tag_positive_manifest(
    config: C075ManifestConfig = DEFAULT_CONFIG,
) -> C075ManifestSummary:
    _validate_config(config)
    positives = _target_positive_rows(config.c074_labels_path)
    if len(positives) < config.minimum_target_positives:
        raise C075ManifestError(
            f"need at least {config.minimum_target_positives} target positives, "
            f"found {len(positives)}"
        )
    source_rows = _read_pair_rows(config.source_manifest_path)[: config.source_row_limit]
    missing = materialize_source_rows(
        source_rows,
        config.source_image_root,
        config.scratch_image_root,
    )
    if missing is not None:
        raise C075ManifestError(f"missing source asset: {missing}")
    materialize_external_rows(positives, config.scratch_image_root)
    external_rows = _external_pair_rows(positives, config.positive_repeat)
    rows = (*external_rows, *source_rows)
    _write_jsonl(config.output_manifest_path, tuple(row.to_json() for row in rows))
    summary = _summary(config, positives, source_rows, external_rows)
    _write_summary(config.output_summary_path, summary)
    config.output_report_path.parent.mkdir(parents=True, exist_ok=True)
    config.output_report_path.write_text(_report(summary), encoding="utf-8")
    return summary


def _validate_config(config: C075ManifestConfig) -> None:
    if config.source_row_limit < 0:
        raise C075ManifestError("source_row_limit must be >= 0")
    if config.positive_repeat < 1:
        raise C075ManifestError("positive_repeat must be >= 1")
    if config.minimum_target_positives < 1:
        raise C075ManifestError("minimum_target_positives must be >= 1")


def _summary(
    config: C075ManifestConfig,
    positives: tuple[JsonObject, ...],
    source_rows: tuple[PairRow, ...],
    external_rows: tuple[PairRow, ...],
) -> C075ManifestSummary:
    return C075ManifestSummary(
        source_manifest_path=str(config.source_manifest_path),
        source_image_root=str(config.source_image_root),
        c074_labels_path=str(config.c074_labels_path),
        c073_labels_path=str(config.c073_labels_path) if config.c073_labels_path else None,
        scratch_image_root=str(config.scratch_image_root),
        output_manifest_path=str(config.output_manifest_path),
        output_summary_path=str(config.output_summary_path),
        target_positive_count=len(positives),
        target_positive_training_rows=len(external_rows),
        source_training_rows=len(source_rows),
        total_rows=len(external_rows) + len(source_rows),
        heldout_rows_used=0,
        missing_paths=0,
        committed_external_image_count=0,
        positive_repeat=config.positive_repeat,
        c073_guard_label_counts=_label_counts(config.c073_labels_path),
        license_caution=_license_caution(positives),
        decision="ready_for_bounded_qwenvl_training",
    )


def _read_pair_rows(path: Path) -> tuple[PairRow, ...]:
    rows = tuple(
        PairRow(
            ref_id=_string_field(raw, "ref_id", path, line_number),
            tgt_id=_string_field(raw, "tgt_id", path, line_number),
            prompt=_string_field(raw, "prompt", path, line_number),
        )
        for line_number, raw in _read_jsonl(path)
    )
    if not rows:
        raise C075ManifestError(f"source manifest has no rows: {path}")
    return rows


def _target_positive_rows(path: Path) -> tuple[JsonObject, ...]:
    return tuple(
        _require_positive_fields(raw, path)
        for _line, raw in _read_jsonl(path)
        if raw.get("manual_label") == "target_positive"
        and raw.get("download_status") == "downloaded"
    )


def _require_positive_fields(row: JsonObject, path: Path) -> JsonObject:
    for field in ("candidate_id", "local_image_path"):
        if not isinstance(row.get(field), str):
            raise C075ManifestError(f"{path}: target positive missing {field}")
    image_path = Path(str(row["local_image_path"]))
    if not image_path.is_file():
        raise C075ManifestError(f"missing c074 target-positive image: {image_path}")
    return row


def _external_pair_rows(rows: tuple[JsonObject, ...], repeat: int) -> tuple[PairRow, ...]:
    return tuple(
        PairRow(image_id, image_id, EXTERNAL_PROMPT)
        for _ in range(repeat)
        for image_id in (external_image_id(str(row["candidate_id"])) for row in rows)
    )


def _read_jsonl(path: Path) -> tuple[tuple[int, JsonObject], ...]:
    if not path.is_file():
        raise C075ManifestError(f"jsonl not found: {path}")
    parsed: list[tuple[int, JsonObject]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            raw: JsonValue = json.loads(line)
            if not isinstance(raw, dict):
                raise C075ManifestError(f"{path}:{line_number} row must be object")
            parsed.append((line_number, raw))
    return tuple(parsed)


def _string_field(row: JsonObject, field: str, path: Path, line_number: int) -> str:
    value = row.get(field)
    if not isinstance(value, str):
        raise C075ManifestError(f"{path}:{line_number} missing {field}")
    return value


def _write_jsonl(path: Path, rows: tuple[JsonObject, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def _write_summary(path: Path, summary: C075ManifestSummary) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(asdict(summary), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _label_counts(path: Path | None) -> dict[str, int]:
    if path is None or not path.is_file():
        return {}
    counts: dict[str, int] = {}
    for _line, row in _read_jsonl(path):
        label = row.get("manual_label")
        if isinstance(label, str):
            counts[label] = counts.get(label, 0) + 1
    return counts


def _license_caution(rows: tuple[JsonObject, ...]) -> str:
    notes = {
        str(row.get("external_license_note", "unknown external license"))
        for row in rows
    }
    return " | ".join(sorted(notes))


def _report(summary: C075ManifestSummary) -> str:
    return "\n".join(
        [
            "# c075 tag-positive direct-green manifest",
            "",
            f"- decision: `{summary.decision}`",
            f"- manifest: `{summary.output_manifest_path}`",
            f"- scratch_image_root: `{summary.scratch_image_root}`",
            f"- target_positive_count: `{summary.target_positive_count}`",
            f"- target_positive_training_rows: `{summary.target_positive_training_rows}`",
            f"- source_training_rows: `{summary.source_training_rows}`",
            f"- total_rows: `{summary.total_rows}`",
            f"- heldout_rows_used: `{summary.heldout_rows_used}`",
            f"- missing_paths: `{summary.missing_paths}`",
            f"- committed_external_image_count: `{summary.committed_external_image_count}`",
            f"- license_caution: {summary.license_caution}",
            "",
        ]
    )


if __name__ == "__main__":
    print(
        json.dumps(
            asdict(build_c075_tag_positive_manifest()),
            ensure_ascii=False,
            indent=2,
        )
    )
