from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Final

from tools.c084_balanced_crop_pair_manifest import (
    C084ManifestConfig,
    C084ManifestSummary,
    build_c084_balanced_crop_pair_manifest,
)
from tools.c085_anchored_full_adapter_manifest import (
    COLOR_ROOT,
    DEFAULT_C060_MANIFEST,
    DEFAULT_HELDOUT_SUMMARY,
    C085ManifestSummary,
    build_c085_manifest,
)

ROOT: Final = Path(__file__).resolve().parents[1]
DEFAULT_REVIEWED_LABELS: Final = (
    ROOT / "eval/c083_sheet_crop_identity_pair_extraction_20260613/reviewed_crop_labels.jsonl"
)
DEFAULT_APPROVED_PAIRS: Final = (
    ROOT / "eval/c083_sheet_crop_identity_pair_extraction_20260613/approved_pair_manifest.jsonl"
)
DEFAULT_EXPANDED_ROOT: Final = ROOT / ".tmp/c087_expanded_crop_pairs_root"
DEFAULT_EXPANDED_MANIFEST: Final = ROOT / "training/manifests/c087_expanded_crop_pairs_20260613.jsonl"
DEFAULT_EXPANDED_SUMMARY: Final = (
    ROOT / "training/manifests/c087_expanded_crop_pairs_20260613.summary.json"
)
DEFAULT_TRAINING_DIR: Final = ROOT / "eval/qwenvl_c087_expanded_crop_positive_training_20260613"
DEFAULT_ANCHORED_ROOT: Final = ROOT / ".tmp/c087_expanded_crop_positive_root"
DEFAULT_ANCHORED_MANIFEST: Final = (
    ROOT / "training/manifests/c087_expanded_anchored_full_adapter_20260613.jsonl"
)
DEFAULT_ANCHORED_SUMMARY: Final = (
    ROOT / "training/manifests/c087_expanded_anchored_full_adapter_20260613.summary.json"
)


@dataclass(frozen=True, slots=True)
class C087ManifestConfig:
    reviewed_labels_path: Path = DEFAULT_REVIEWED_LABELS
    approved_pairs_path: Path = DEFAULT_APPROVED_PAIRS
    expanded_root: Path = DEFAULT_EXPANDED_ROOT
    expanded_manifest_path: Path = DEFAULT_EXPANDED_MANIFEST
    expanded_summary_path: Path = DEFAULT_EXPANDED_SUMMARY
    expanded_report_path: Path = DEFAULT_TRAINING_DIR / "crop_manifest_report.md"
    anchored_root: Path = DEFAULT_ANCHORED_ROOT
    anchored_manifest_path: Path = DEFAULT_ANCHORED_MANIFEST
    anchored_summary_path: Path = DEFAULT_ANCHORED_SUMMARY
    combined_summary_path: Path = DEFAULT_TRAINING_DIR / "manifest_stdout.json"
    c060_manifest_path: Path = DEFAULT_C060_MANIFEST
    color_root: Path = COLOR_ROOT
    heldout_summary_path: Path = DEFAULT_HELDOUT_SUMMARY
    max_pairs_per_group: int = 80
    max_pairs_per_source_pair: int = 16
    crop_row_limit: int = 320


@dataclass(frozen=True, slots=True)
class C087ManifestSummary:
    expanded_crop_summary: C084ManifestSummary
    anchored_summary: C085ManifestSummary
    decision: str


def build_c087_expanded_crop_positive_manifest(
    config: C087ManifestConfig = C087ManifestConfig(),
) -> C087ManifestSummary:
    expanded = build_c084_balanced_crop_pair_manifest(
        C084ManifestConfig(
            reviewed_labels_path=config.reviewed_labels_path,
            approved_pairs_path=config.approved_pairs_path,
            scratch_image_root=config.expanded_root,
            output_manifest_path=config.expanded_manifest_path,
            output_summary_path=config.expanded_summary_path,
            output_report_path=config.expanded_report_path,
            max_pairs_per_group=config.max_pairs_per_group,
            max_pairs_per_source_pair=config.max_pairs_per_source_pair,
        )
    )
    anchored = build_c085_manifest(
        c084_manifest=config.expanded_manifest_path,
        c060_manifest=config.c060_manifest_path,
        c084_root=config.expanded_root,
        color_root=config.color_root,
        output_root=config.anchored_root,
        output_manifest=config.anchored_manifest_path,
        output_summary=config.anchored_summary_path,
        heldout_summary=config.heldout_summary_path,
        crop_row_limit=config.crop_row_limit,
    )
    decision = (
        "ready_for_c087_expanded_crop_positive_training"
        if expanded.heldout_rows_used == 0 and anchored.heldout_rows_used == 0
        else "blocked_heldout_leakage"
    )
    summary = C087ManifestSummary(expanded, anchored, decision)
    config.combined_summary_path.parent.mkdir(parents=True, exist_ok=True)
    config.combined_summary_path.write_text(
        json.dumps(asdict(summary), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary


def main() -> None:
    summary = build_c087_expanded_crop_positive_manifest()
    print(json.dumps(asdict(summary), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
