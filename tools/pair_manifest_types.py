from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Final


IMAGE_SUFFIX: Final = ".jpg"
CAPTION_SUFFIX: Final = ".txt"


@dataclass(frozen=True, slots=True)
class DatasetLayoutError(Exception):
    path: Path
    detail: str

    def __str__(self) -> str:
        return f"{self.detail}: {self.path}"


@dataclass(frozen=True, slots=True)
class CliUsageError(Exception):
    detail: str

    def __str__(self) -> str:
        return self.detail


@dataclass(frozen=True, slots=True)
class ImageEntry:
    relative_dir: str
    image_id: str
    caption: str


@dataclass(frozen=True, slots=True)
class PairRow:
    ref_id: str
    tgt_id: str
    prompt: str


@dataclass(frozen=True, slots=True)
class DatasetAudit:
    image_paths: list[Path]
    caption_count: int
    missing_captions: list[str]
    duplicate_ids: list[str]


@dataclass(frozen=True, slots=True)
class SplitRange:
    start_index: int
    end_index_exclusive: int
    rows: int


@dataclass(frozen=True, slots=True)
class SplitMetadata:
    strategy: str
    train: SplitRange
    validation: SplitRange


@dataclass(frozen=True, slots=True)
class PairManifestSummary:
    dataset_root: str
    directories: int
    source_images: int
    caption_count: int
    rows: int
    skipped_singleton_directories: int
    missing_captions: list[str]
    duplicate_ids: list[str]
    split: SplitMetadata
    output_path: str | None
    summary_path: str | None
    wrote_output: bool
    wrote_summary: bool


@dataclass(frozen=True, slots=True)
class PairBuildResult:
    rows: list[PairRow]
    directories: int
    source_images: int
    caption_count: int
    skipped_singleton_directories: int
    missing_captions: list[str]
    duplicate_ids: list[str]


def audit_dataset(dataset_root: Path) -> DatasetAudit:
    """Inspect image/caption reachability before manifest rows are emitted."""
    if not dataset_root.is_dir():
        raise DatasetLayoutError(dataset_root, "Dataset root is not a directory")
    image_paths = sorted(
        path
        for path in dataset_root.rglob("*")
        if path.is_file() and path.suffix.lower() == IMAGE_SUFFIX
    )
    caption_count = sum(
        1
        for path in dataset_root.rglob("*")
        if path.is_file() and path.suffix.lower() == CAPTION_SUFFIX
    )
    id_counts: dict[str, int] = {}
    missing_captions: list[str] = []
    for image_path in image_paths:
        image_id = image_path.relative_to(dataset_root).with_suffix("").as_posix()
        id_counts[image_id] = id_counts.get(image_id, 0) + 1
        caption_path = image_path.with_suffix(CAPTION_SUFFIX)
        if not caption_path.is_file():
            missing_captions.append(caption_path.relative_to(dataset_root).as_posix())
    duplicate_ids = sorted(image_id for image_id, count in id_counts.items() if count > 1)
    return DatasetAudit(
        image_paths=image_paths,
        caption_count=caption_count,
        missing_captions=missing_captions,
        duplicate_ids=duplicate_ids,
    )


def make_split_metadata(row_count: int) -> SplitMetadata:
    train_rows = int(row_count * 0.95)
    validation_rows = row_count - train_rows
    return SplitMetadata(
        strategy="sorted_95_5",
        train=SplitRange(
            start_index=0,
            end_index_exclusive=train_rows,
            rows=train_rows,
        ),
        validation=SplitRange(
            start_index=train_rows,
            end_index_exclusive=row_count,
            rows=validation_rows,
        ),
    )


def write_summary(summary: PairManifestSummary, summary_path: Path) -> None:
    """Write the audit and split metadata sidecar."""
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(asdict(summary), ensure_ascii=True, indent=2) + "\n",
        encoding="utf-8",
    )
