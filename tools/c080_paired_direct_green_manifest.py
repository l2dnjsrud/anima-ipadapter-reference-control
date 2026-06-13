# /// script
# dependencies = []
# ///
# --- How to run -----------------------------------------------------
# PYTHONPATH=. /home/wktwin/anima-lora-training-bundle/anima_lora/.venv/bin/python \
#   tools/c080_paired_direct_green_manifest.py

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Final

from tools.c075_manifest_files import materialize_source_rows
from tools.c075_tag_positive_manifest_types import PairRow
from tools.c079_manifest_io import (
    ExternalTrainingSource,
    guard_proxy_rows,
    label_counts,
    license_notes,
    materialize_external,
    materialize_external_rows,
    read_pair_rows,
    target_positive_rows,
    write_jsonl,
)
from tools.siglip_auto_caption_types import JsonObject


DEFAULT_SOURCE_MANIFEST: Final = Path(
    "training/manifests/c060_qwenvl_failure_focused_clean32_c052_20260612.jsonl"
)
DEFAULT_SOURCE_ROOT: Final = Path(
    "/home/wktwin/anima-lora-training-bundle/image_dataset_color_panel_style_v5_best"
)
PAIR_PROMPT: Final = (
    "mrcolor_panel_style, full color manhwa comic panel, same green-skinned "
    "non-human anime character identity, visible green skin, tail or monster "
    "features, clean webtoon coloring, cel shaded illustration, safe"
)
GUARD_PROMPT: Final = (
    "mrcolor_panel_style, full color manhwa comic panel, solo anime character "
    "reference portrait, clear species and face structure, clean webtoon coloring"
)


@dataclass(frozen=True, slots=True)
class C080ManifestError(Exception):
    detail: str

    def __str__(self) -> str:
        return self.detail


@dataclass(frozen=True, slots=True)
class C080ManifestConfig:
    source_manifest_path: Path
    source_image_root: Path
    c074_labels_path: Path
    c078_labels_path: Path
    c077_labels_path: Path
    scratch_image_root: Path
    output_manifest_path: Path
    output_summary_path: Path
    output_report_path: Path
    source_row_limit: int = 80
    c074_pair_repeat: int = 8
    guard_repeat: int = 1
    minimum_c074_pair_sources: int = 8


@dataclass(frozen=True, slots=True)
class C080ManifestSummary:
    source_manifest_path: str
    source_image_root: str
    c074_labels_path: str
    c078_labels_path: str
    c077_labels_path: str
    scratch_image_root: str
    output_manifest_path: str
    output_summary_path: str
    c074_pair_source_count: int
    c074_paired_training_rows: int
    c078_unpaired_positive_count: int
    c078_training_rows: int
    guard_proxy_count: int
    guard_proxy_training_rows: int
    source_training_rows: int
    total_rows: int
    direct_self_pair_rows: int
    heldout_rows_used: int
    committed_raw_image_count: int
    c074_pair_repeat: int
    guard_repeat: int
    c077_label_counts: dict[str, int]
    license_caution: str
    decision: str


def build_c080_paired_direct_green_manifest(
    config: C080ManifestConfig,
) -> C080ManifestSummary:
    _validate_config(config)
    c074_rows = target_positive_rows(config.c074_labels_path, "c074")
    if len(c074_rows) < config.minimum_c074_pair_sources:
        raise C080ManifestError(
            f"need at least {config.minimum_c074_pair_sources} c074 paired sources, "
            f"found {len(c074_rows)}"
        )
    c078_rows = target_positive_rows(config.c078_labels_path, "c078")
    guard_rows = guard_proxy_rows(config.c077_labels_path)
    source_rows = read_pair_rows(config.source_manifest_path)[: config.source_row_limit]
    missing = materialize_source_rows(
        source_rows,
        config.source_image_root,
        config.scratch_image_root,
    )
    if missing is not None:
        raise C080ManifestError(f"missing source asset: {missing}")

    c074_training = _paired_c074_rows(c074_rows, config)
    guard_training = materialize_external_rows(
        _guard_sources(guard_rows),
        config.guard_repeat,
        config.scratch_image_root,
    )
    rows = (*c074_training, *guard_training, *source_rows)
    write_jsonl(config.output_manifest_path, tuple(row.to_json() for row in rows))
    summary = _summary(config, c074_rows, c078_rows, guard_rows, source_rows, c074_training, guard_training)
    _write_summary(config.output_summary_path, summary)
    config.output_report_path.parent.mkdir(parents=True, exist_ok=True)
    config.output_report_path.write_text(_report(summary), encoding="utf-8")
    return summary


def _paired_c074_rows(
    rows: tuple[JsonObject, ...],
    config: C080ManifestConfig,
) -> tuple[PairRow, ...]:
    sources = tuple(
        ExternalTrainingSource(row, "external/c080_c074_paired", PAIR_PROMPT, "c074")
        for row in rows
    )
    for source in sources:
        materialize_external(source, config.scratch_image_root)
    pair_rows: list[PairRow] = []
    for repeat_index in range(config.c074_pair_repeat):
        offset = repeat_index + 1
        for index, source in enumerate(sources):
            target = sources[(index + offset) % len(sources)]
            pair_rows.append(PairRow(_image_id(source), _image_id(target), PAIR_PROMPT))
    return tuple(pair_rows)


def _guard_sources(rows: tuple[JsonObject, ...]) -> tuple[ExternalTrainingSource, ...]:
    return tuple(
        ExternalTrainingSource(row, "external/c080_guard_proxy", GUARD_PROMPT, "c077")
        for row in rows
    )


def _image_id(source: ExternalTrainingSource) -> str:
    candidate = source.row.get("candidate_id")
    if not isinstance(candidate, str):
        raise C080ManifestError("external row missing candidate_id")
    return f"{source.prefix}/{candidate}"


def _summary(
    config: C080ManifestConfig,
    c074_rows: tuple[JsonObject, ...],
    c078_rows: tuple[JsonObject, ...],
    guard_rows: tuple[JsonObject, ...],
    source_rows: tuple[PairRow, ...],
    c074_training: tuple[PairRow, ...],
    guard_training: tuple[PairRow, ...],
) -> C080ManifestSummary:
    return C080ManifestSummary(
        source_manifest_path=str(config.source_manifest_path),
        source_image_root=str(config.source_image_root),
        c074_labels_path=str(config.c074_labels_path),
        c078_labels_path=str(config.c078_labels_path),
        c077_labels_path=str(config.c077_labels_path),
        scratch_image_root=str(config.scratch_image_root),
        output_manifest_path=str(config.output_manifest_path),
        output_summary_path=str(config.output_summary_path),
        c074_pair_source_count=len(c074_rows),
        c074_paired_training_rows=len(c074_training),
        c078_unpaired_positive_count=len(c078_rows),
        c078_training_rows=0,
        guard_proxy_count=len(guard_rows),
        guard_proxy_training_rows=len(guard_training),
        source_training_rows=len(source_rows),
        total_rows=len(c074_training) + len(guard_training) + len(source_rows),
        direct_self_pair_rows=sum(1 for row in c074_training if row.ref_id == row.tgt_id),
        heldout_rows_used=0,
        committed_raw_image_count=0,
        c074_pair_repeat=config.c074_pair_repeat,
        guard_repeat=config.guard_repeat,
        c077_label_counts=label_counts(config.c077_labels_path),
        license_caution=" | ".join(license_notes((*c074_rows, *c078_rows, *guard_rows))),
        decision="ready_for_c080_bounded_paired_qwenvl_training",
    )


def _validate_config(config: C080ManifestConfig) -> None:
    if config.source_row_limit < 0:
        raise C080ManifestError("source_row_limit must be >= 0")
    if config.c074_pair_repeat < 1:
        raise C080ManifestError("c074_pair_repeat must be >= 1")
    if config.guard_repeat < 0:
        raise C080ManifestError("guard_repeat must be >= 0")
    if config.minimum_c074_pair_sources < 2:
        raise C080ManifestError("minimum_c074_pair_sources must be >= 2")


def _write_summary(path: Path, summary: C080ManifestSummary) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(asdict(summary), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _report(summary: C080ManifestSummary) -> str:
    return "\n".join(
        [
            "# c080 paired direct-green manifest",
            "",
            f"- decision: `{summary.decision}`",
            f"- manifest: `{summary.output_manifest_path}`",
            f"- scratch_image_root: `{summary.scratch_image_root}`",
            f"- c074_pair_source_count: `{summary.c074_pair_source_count}`",
            f"- c074_paired_training_rows: `{summary.c074_paired_training_rows}`",
            f"- direct_self_pair_rows: `{summary.direct_self_pair_rows}`",
            f"- c078_unpaired_positive_count: `{summary.c078_unpaired_positive_count}`",
            f"- c078_training_rows: `{summary.c078_training_rows}`",
            f"- guard_proxy_training_rows: `{summary.guard_proxy_training_rows}`",
            f"- source_training_rows: `{summary.source_training_rows}`",
            f"- total_rows: `{summary.total_rows}`",
            f"- heldout_rows_used: `{summary.heldout_rows_used}`",
            f"- committed_raw_image_count: `{summary.committed_raw_image_count}`",
            f"- license_caution: {summary.license_caution}",
            "",
        ]
    )


def _default_config() -> C080ManifestConfig:
    return C080ManifestConfig(
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
        scratch_image_root=Path(".tmp/c080_paired_direct_green_identity_root"),
        output_manifest_path=Path(
            "training/manifests/c080_paired_direct_green_identity_20260613.jsonl"
        ),
        output_summary_path=Path(
            "training/manifests/c080_paired_direct_green_identity_20260613.summary.json"
        ),
        output_report_path=Path(
            "eval/qwenvl_c080_paired_direct_green_training_20260613/manifest_report.md"
        ),
    )


def main() -> None:
    summary = build_c080_paired_direct_green_manifest(_default_config())
    print(json.dumps(asdict(summary), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
