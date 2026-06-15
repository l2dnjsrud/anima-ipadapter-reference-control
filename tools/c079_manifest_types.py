from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class C079ManifestError(Exception):
    detail: str

    def __str__(self) -> str:
        return self.detail


@dataclass(frozen=True, slots=True)
class C079ManifestConfig:
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
    positive_repeat: int = 4
    guard_repeat: int = 1
    minimum_total_target_positives: int = 24


@dataclass(frozen=True, slots=True)
class C079ManifestSummary:
    source_manifest_path: str
    source_image_root: str
    c074_labels_path: str
    c078_labels_path: str
    c077_labels_path: str
    scratch_image_root: str
    output_manifest_path: str
    output_summary_path: str
    c074_real_positive_count: int
    c078_synthetic_target_positive_count: int
    total_target_positive_count: int
    target_positive_training_rows: int
    guard_proxy_count: int
    guard_proxy_training_rows: int
    source_training_rows: int
    total_rows: int
    heldout_rows_used: int
    committed_raw_image_count: int
    positive_repeat: int
    guard_repeat: int
    c077_label_counts: dict[str, int]
    license_caution: str
    decision: str
