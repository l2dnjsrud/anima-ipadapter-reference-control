# /// script
# dependencies = []
# ///
# --- How to run -----------------------------------------------------
# PYTHONPATH=. /home/wktwin/anima-lora-training-bundle/anima_lora/.venv/bin/python \
#   tools/c079_synthetic_positive_manifest.py

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Final

from tools.c075_manifest_files import materialize_source_rows
from tools.c075_tag_positive_manifest_types import PairRow
from tools.c079_manifest_io import (
    ExternalTrainingSource,
    guard_proxy_rows,
    label_counts,
    license_notes,
    materialize_external_rows,
    read_pair_rows,
    target_positive_rows,
    write_jsonl,
)
from tools.c079_manifest_types import (
    C079ManifestConfig,
    C079ManifestError,
    C079ManifestSummary,
)
from tools.siglip_auto_caption_types import JsonObject


DEFAULT_SOURCE_MANIFEST: Final = Path(
    "training/manifests/c060_qwenvl_failure_focused_clean32_c052_20260612.jsonl"
)
DEFAULT_SOURCE_ROOT: Final = Path(
    "/home/wktwin/anima-lora-training-bundle/image_dataset_color_panel_style_v5_best"
)
TARGET_PROMPT: Final = (
    "mrcolor_panel_style, full color manhwa comic panel, solo green-skinned "
    "non-human anime character portrait, monster or demon traits, visible skin "
    "tone, tail or horns, clean webtoon coloring, cel shaded illustration, safe"
)
GUARD_PROMPT: Final = (
    "mrcolor_panel_style, full color manhwa comic panel, solo anime character "
    "reference portrait, clear species and face structure, clean webtoon coloring"
)


def build_c079_synthetic_positive_manifest(
    config: C079ManifestConfig,
) -> C079ManifestSummary:
    _validate_config(config)
    c074_rows = target_positive_rows(config.c074_labels_path, "c074")
    c078_rows = target_positive_rows(config.c078_labels_path, "c078")
    total_positive = len(c074_rows) + len(c078_rows)
    if total_positive < config.minimum_total_target_positives:
        raise C079ManifestError(
            f"need at least {config.minimum_total_target_positives} total target positives, "
            f"found {total_positive}"
        )
    guard_rows = guard_proxy_rows(config.c077_labels_path)
    source_rows = read_pair_rows(config.source_manifest_path)[: config.source_row_limit]
    missing = materialize_source_rows(
        source_rows,
        config.source_image_root,
        config.scratch_image_root,
    )
    if missing is not None:
        raise C079ManifestError(f"missing source asset: {missing}")

    target_training = materialize_external_rows(
        _target_sources(c074_rows, c078_rows),
        config.positive_repeat,
        config.scratch_image_root,
    )
    guard_training = materialize_external_rows(
        _guard_sources(guard_rows),
        config.guard_repeat,
        config.scratch_image_root,
    )
    rows = (*target_training, *guard_training, *source_rows)
    write_jsonl(config.output_manifest_path, tuple(row.to_json() for row in rows))
    summary = _summary(config, c074_rows, c078_rows, guard_rows, source_rows, target_training, guard_training)
    _write_summary(config.output_summary_path, summary)
    config.output_report_path.parent.mkdir(parents=True, exist_ok=True)
    config.output_report_path.write_text(_report(summary), encoding="utf-8")
    return summary


def _target_sources(
    c074_rows: tuple[JsonObject, ...],
    c078_rows: tuple[JsonObject, ...],
) -> tuple[ExternalTrainingSource, ...]:
    return (
        *(
            ExternalTrainingSource(row, "external/c074_real_direct_green", TARGET_PROMPT, "c074")
            for row in c074_rows
        ),
        *(
            ExternalTrainingSource(row, "external/c078_synthetic_direct_green", TARGET_PROMPT, "c078")
            for row in c078_rows
        ),
    )


def _guard_sources(rows: tuple[JsonObject, ...]) -> tuple[ExternalTrainingSource, ...]:
    return tuple(
        ExternalTrainingSource(row, "external/c077_guard_proxy", GUARD_PROMPT, "c077")
        for row in rows
    )


def _summary(
    config: C079ManifestConfig,
    c074_rows: tuple[JsonObject, ...],
    c078_rows: tuple[JsonObject, ...],
    guard_rows: tuple[JsonObject, ...],
    source_rows: tuple[PairRow, ...],
    target_training: tuple[PairRow, ...],
    guard_training: tuple[PairRow, ...],
) -> C079ManifestSummary:
    return C079ManifestSummary(
        source_manifest_path=str(config.source_manifest_path),
        source_image_root=str(config.source_image_root),
        c074_labels_path=str(config.c074_labels_path),
        c078_labels_path=str(config.c078_labels_path),
        c077_labels_path=str(config.c077_labels_path),
        scratch_image_root=str(config.scratch_image_root),
        output_manifest_path=str(config.output_manifest_path),
        output_summary_path=str(config.output_summary_path),
        c074_real_positive_count=len(c074_rows),
        c078_synthetic_target_positive_count=len(c078_rows),
        total_target_positive_count=len(c074_rows) + len(c078_rows),
        target_positive_training_rows=len(target_training),
        guard_proxy_count=len(guard_rows),
        guard_proxy_training_rows=len(guard_training),
        source_training_rows=len(source_rows),
        total_rows=len(target_training) + len(guard_training) + len(source_rows),
        heldout_rows_used=0,
        committed_raw_image_count=0,
        positive_repeat=config.positive_repeat,
        guard_repeat=config.guard_repeat,
        c077_label_counts=label_counts(config.c077_labels_path),
        license_caution=" | ".join(license_notes((*c074_rows, *c078_rows, *guard_rows))),
        decision="ready_for_c079_bounded_qwenvl_training",
    )


def _validate_config(config: C079ManifestConfig) -> None:
    if config.source_row_limit < 0:
        raise C079ManifestError("source_row_limit must be >= 0")
    if config.positive_repeat < 1:
        raise C079ManifestError("positive_repeat must be >= 1")
    if config.guard_repeat < 0:
        raise C079ManifestError("guard_repeat must be >= 0")
    if config.minimum_total_target_positives < 1:
        raise C079ManifestError("minimum_total_target_positives must be >= 1")


def _write_summary(path: Path, summary: C079ManifestSummary) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(asdict(summary), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _report(summary: C079ManifestSummary) -> str:
    return "\n".join(
        [
            "# c079 synthetic-positive direct-green manifest",
            "",
            f"- decision: `{summary.decision}`",
            f"- manifest: `{summary.output_manifest_path}`",
            f"- scratch_image_root: `{summary.scratch_image_root}`",
            f"- c074_real_positive_count: `{summary.c074_real_positive_count}`",
            f"- c078_synthetic_target_positive_count: `{summary.c078_synthetic_target_positive_count}`",
            f"- target_positive_training_rows: `{summary.target_positive_training_rows}`",
            f"- guard_proxy_count: `{summary.guard_proxy_count}`",
            f"- guard_proxy_training_rows: `{summary.guard_proxy_training_rows}`",
            f"- source_training_rows: `{summary.source_training_rows}`",
            f"- total_rows: `{summary.total_rows}`",
            f"- heldout_rows_used: `{summary.heldout_rows_used}`",
            f"- committed_raw_image_count: `{summary.committed_raw_image_count}`",
            f"- license_caution: {summary.license_caution}",
            "",
        ]
    )


def _default_config() -> C079ManifestConfig:
    return C079ManifestConfig(
        source_manifest_path=DEFAULT_SOURCE_MANIFEST,
        source_image_root=DEFAULT_SOURCE_ROOT,
        c074_labels_path=Path(
            "eval/c074_tag_backed_direct_green_source_acquisition_20260612/"
            "reviewed_external_labels.jsonl"
        ),
        c078_labels_path=Path(
            "eval/c078_synthetic_direct_green_bootstrap_20260612/reviewed_synthetic_labels.jsonl"
        ),
        c077_labels_path=Path(
            "eval/c077_direct_green_target_positive_acquisition_20260612/"
            "reviewed_external_labels.jsonl"
        ),
        scratch_image_root=Path(".tmp/c079_synthetic_positive_direct_green_root"),
        output_manifest_path=Path(
            "training/manifests/c079_synthetic_positive_direct_green_20260612.jsonl"
        ),
        output_summary_path=Path(
            "training/manifests/c079_synthetic_positive_direct_green_20260612.summary.json"
        ),
        output_report_path=Path(
            "eval/qwenvl_c079_synthetic_positive_training_20260612/manifest_report.md"
        ),
    )


def main() -> None:
    summary = build_c079_synthetic_positive_manifest(_default_config())
    print(json.dumps(asdict(summary), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
