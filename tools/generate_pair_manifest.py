from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Final

import typer


IMAGE_SUFFIX: Final = ".jpg"
CAPTION_SUFFIX: Final = ".txt"
app = typer.Typer(add_completion=False)


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
class PairManifestSummary:
    dataset_root: str
    directories: int
    source_images: int
    rows: int
    skipped_singleton_directories: int
    output_path: str | None
    wrote_output: bool


@dataclass(frozen=True, slots=True)
class PairBuildResult:
    rows: list[PairRow]
    directories: int
    source_images: int
    skipped_singleton_directories: int


def build_manifest(
    dataset_root: Path,
    *,
    allow_self_pairs: bool = False,
    limit: int | None = None,
) -> PairBuildResult:
    """Build deterministic round-robin pairs from a nested image dataset."""
    if not dataset_root.is_dir():
        raise DatasetLayoutError(dataset_root, "Dataset root is not a directory")

    image_paths = sorted(
        path
        for path in dataset_root.rglob("*")
        if path.is_file() and path.suffix.lower() == IMAGE_SUFFIX
    )
    groups: dict[str, list[ImageEntry]] = {}
    for image_path in image_paths:
        entry = _load_entry(dataset_root, image_path)
        groups.setdefault(entry.relative_dir, []).append(entry)

    rows: list[PairRow] = []
    skipped_singleton_directories = 0
    for relative_dir in sorted(groups):
        entries = groups[relative_dir]
        if len(entries) == 1 and not allow_self_pairs:
            skipped_singleton_directories += 1
            continue
        for index, ref_entry in enumerate(entries):
            tgt_entry = entries[(index + 1) % len(entries)]
            rows.append(
                PairRow(
                    ref_id=ref_entry.image_id,
                    tgt_id=tgt_entry.image_id,
                    prompt=tgt_entry.caption,
                )
            )
            if limit is not None and len(rows) >= limit:
                return PairBuildResult(
                    rows=rows,
                    directories=len(groups),
                    source_images=len(image_paths),
                    skipped_singleton_directories=skipped_singleton_directories,
                )
    return PairBuildResult(
        rows=rows,
        directories=len(groups),
        source_images=len(image_paths),
        skipped_singleton_directories=skipped_singleton_directories,
    )


def build_rows(dataset_root: Path, *, allow_self_pairs: bool = True) -> list[PairRow]:
    """Build rows with the legacy self-pair behavior used by focused tests."""
    return build_manifest(dataset_root, allow_self_pairs=allow_self_pairs).rows


def write_manifest(rows: list[PairRow], output_path: Path) -> None:
    """Write JSONL rows to disk."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(asdict(row), ensure_ascii=True) + "\n")


def generate_manifest(
    dataset_root: Path,
    output_path: Path | None,
    *,
    count_only: bool,
    dry_run: bool,
    allow_self_pairs: bool,
    limit: int | None,
) -> PairManifestSummary:
    """Create pair rows and optionally persist them."""
    result = build_manifest(
        dataset_root,
        allow_self_pairs=allow_self_pairs,
        limit=limit,
    )
    wrote_output = False
    if count_only:
        return PairManifestSummary(
            dataset_root=str(dataset_root),
            directories=result.directories,
            source_images=result.source_images,
            rows=len(result.rows),
            skipped_singleton_directories=result.skipped_singleton_directories,
            output_path=None,
            wrote_output=False,
        )
    if output_path is None:
        raise CliUsageError("--output is required unless --count-only is used")
    if not dry_run:
        write_manifest(result.rows, output_path)
        wrote_output = True
    return PairManifestSummary(
        dataset_root=str(dataset_root),
        directories=result.directories,
        source_images=result.source_images,
        rows=len(result.rows),
        skipped_singleton_directories=result.skipped_singleton_directories,
        output_path=str(output_path),
        wrote_output=wrote_output,
    )


def _load_entry(dataset_root: Path, image_path: Path) -> ImageEntry:
    relative_path = image_path.relative_to(dataset_root)
    caption_path = image_path.with_suffix(CAPTION_SUFFIX)
    if not caption_path.is_file():
        raise DatasetLayoutError(caption_path, "Missing caption sidecar")
    return ImageEntry(
        relative_dir=relative_path.parent.as_posix(),
        image_id=relative_path.with_suffix("").as_posix(),
        caption=caption_path.read_text(encoding="utf-8").strip(),
    )


@app.command()
def main(
    dataset_root: Path,
    output_path: Path | None = typer.Option(
        None,
        "--output",
        "-o",
        help="Output JSONL path for ref_id/tgt_id/prompt rows.",
    ),
    count_only: bool = typer.Option(
        False,
        "--count-only",
        help="Print dataset pair counts without writing a manifest.",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Build rows and print a summary without writing output.",
    ),
    allow_self_pairs: bool = typer.Option(
        False,
        "--allow-self-pairs",
        help="Allow singleton directories to emit ref_id == tgt_id rows.",
    ),
    limit: int | None = typer.Option(
        None,
        "--limit",
        min=1,
        help="Maximum number of rows to emit after deterministic ordering.",
    ),
) -> None:
    """Generate Wenaka-style pair JSONL for local image datasets."""
    try:
        summary = generate_manifest(
            dataset_root,
            output_path,
            count_only=count_only,
            dry_run=dry_run,
            allow_self_pairs=allow_self_pairs,
            limit=limit,
        )
    except CliUsageError as error:
        raise typer.BadParameter(str(error)) from error
    except DatasetLayoutError as error:
        typer.echo(str(error), err=True)
        raise typer.Exit(code=1) from error
    typer.echo(json.dumps(asdict(summary), ensure_ascii=True))


if __name__ == "__main__":
    app()
