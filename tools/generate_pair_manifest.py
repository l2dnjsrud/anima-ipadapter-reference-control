from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import typer

if __package__:
    from tools.pair_manifest_types import (
        CAPTION_SUFFIX,
        CliUsageError,
        DatasetLayoutError,
        ImageEntry,
        PairBuildResult,
        PairManifestSummary,
        PairRow,
        audit_dataset,
        make_split_metadata,
        write_summary,
    )
else:
    from pair_manifest_types import (
        CAPTION_SUFFIX,
        CliUsageError,
        DatasetLayoutError,
        ImageEntry,
        PairBuildResult,
        PairManifestSummary,
        PairRow,
        audit_dataset,
        make_split_metadata,
        write_summary,
    )


app = typer.Typer(add_completion=False)


def build_manifest(
    dataset_root: Path,
    *,
    allow_self_pairs: bool = False,
    limit: int | None = None,
) -> PairBuildResult:
    """Build deterministic round-robin pairs from a nested image dataset."""
    if not dataset_root.is_dir():
        raise DatasetLayoutError(dataset_root, "Dataset root is not a directory")

    audit = audit_dataset(dataset_root)
    if audit.missing_captions:
        raise DatasetLayoutError(
            dataset_root / audit.missing_captions[0],
            "Missing caption sidecar",
        )
    if audit.duplicate_ids:
        raise DatasetLayoutError(
            dataset_root / audit.duplicate_ids[0],
            "Duplicate image id",
        )
    groups: dict[str, list[ImageEntry]] = {}
    for image_path in audit.image_paths:
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
                    source_images=len(audit.image_paths),
                    caption_count=audit.caption_count,
                    skipped_singleton_directories=skipped_singleton_directories,
                    missing_captions=audit.missing_captions,
                    duplicate_ids=audit.duplicate_ids,
                )
    return PairBuildResult(
        rows=rows,
        directories=len(groups),
        source_images=len(audit.image_paths),
        caption_count=audit.caption_count,
        skipped_singleton_directories=skipped_singleton_directories,
        missing_captions=audit.missing_captions,
        duplicate_ids=audit.duplicate_ids,
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
    summary_output_path: Path | None = None,
) -> PairManifestSummary:
    """Create pair rows and optionally persist them."""
    result = build_manifest(
        dataset_root,
        allow_self_pairs=allow_self_pairs,
        limit=limit,
    )
    split = make_split_metadata(len(result.rows))
    wrote_output = False
    if count_only:
        return PairManifestSummary(
            dataset_root=str(dataset_root),
            directories=result.directories,
            source_images=result.source_images,
            caption_count=result.caption_count,
            rows=len(result.rows),
            skipped_singleton_directories=result.skipped_singleton_directories,
            missing_captions=result.missing_captions,
            duplicate_ids=result.duplicate_ids,
            split=split,
            output_path=None,
            summary_path=None,
            wrote_output=False,
            wrote_summary=False,
        )
    if output_path is None:
        raise CliUsageError("--output is required unless --count-only is used")
    summary_path = summary_output_path or output_path.with_suffix(".summary.json")
    wrote_summary = False
    if not dry_run:
        write_manifest(result.rows, output_path)
        wrote_output = True
    summary = PairManifestSummary(
        dataset_root=str(dataset_root),
        directories=result.directories,
        source_images=result.source_images,
        caption_count=result.caption_count,
        rows=len(result.rows),
        skipped_singleton_directories=result.skipped_singleton_directories,
        missing_captions=result.missing_captions,
        duplicate_ids=result.duplicate_ids,
        split=split,
        output_path=str(output_path),
        summary_path=str(summary_path),
        wrote_output=wrote_output,
        wrote_summary=not dry_run,
    )
    if not dry_run:
        write_summary(summary, summary_path)
        wrote_summary = True
    if wrote_summary:
        return summary
    return summary


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
    summary_output_path: Path | None = typer.Option(
        None,
        "--summary-output",
        help="Output JSON path for audit and split metadata.",
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
            summary_output_path=summary_output_path,
        )
    except CliUsageError as error:
        raise typer.BadParameter(str(error)) from error
    except DatasetLayoutError as error:
        typer.echo(str(error), err=True)
        raise typer.Exit(code=1) from error
    typer.echo(json.dumps(asdict(summary), ensure_ascii=True))


if __name__ == "__main__":
    app()
