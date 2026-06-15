from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from tools.siglip_auto_caption_types import JsonObject


@dataclass(frozen=True, slots=True)
class C075ManifestError(Exception):
    detail: str

    def __str__(self) -> str:
        return self.detail


@dataclass(frozen=True, slots=True)
class PairRow:
    ref_id: str
    tgt_id: str
    prompt: str

    def to_json(self) -> JsonObject:
        return {"ref_id": self.ref_id, "tgt_id": self.tgt_id, "prompt": self.prompt}


@dataclass(frozen=True, slots=True)
class C075ManifestConfig:
    source_manifest_path: Path
    source_image_root: Path
    c074_labels_path: Path
    c073_labels_path: Path | None
    scratch_image_root: Path
    output_manifest_path: Path
    output_summary_path: Path
    output_report_path: Path
    source_row_limit: int = 80
    positive_repeat: int = 4
    minimum_target_positives: int = 4


@dataclass(frozen=True, slots=True)
class C075ManifestSummary:
    source_manifest_path: str
    source_image_root: str
    c074_labels_path: str
    c073_labels_path: str | None
    scratch_image_root: str
    output_manifest_path: str
    output_summary_path: str
    target_positive_count: int
    target_positive_training_rows: int
    source_training_rows: int
    total_rows: int
    heldout_rows_used: int
    missing_paths: int
    committed_external_image_count: int
    positive_repeat: int
    c073_guard_label_counts: dict[str, int]
    license_caution: str
    decision: str
