from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Annotated, Final, Literal

import typer


IMAGE_SUFFIX: Final = ".jpg"
CAPTION_SUFFIX: Final = ".txt"
PairLabel = Literal["positive", "negative"]


@dataclass(frozen=True, slots=True)
class ProbeImage:
    image_id: str
    group: str


@dataclass(frozen=True, slots=True)
class ProbePairRow:
    pair_id: str
    label: PairLabel
    anchor_id: str
    candidate_id: str
    anchor_group: str
    candidate_group: str


@dataclass(frozen=True, slots=True)
class ProbeManifestSummary:
    dataset_root: str
    groups: int
    usable_groups: int
    pairs: int
    positive_pairs: int
    negative_pairs: int
    output_path: str


@dataclass(frozen=True, slots=True)
class ProbeManifestError(Exception):
    detail: str

    def __str__(self) -> str:
        return self.detail


def build_probe_rows(dataset_root: Path, *, pairs_per_label: int) -> tuple[ProbePairRow, ...]:
    if pairs_per_label < 1:
        raise ProbeManifestError("pairs_per_label must be >= 1")
    groups = _group_images(dataset_root)
    usable = [(group, entries) for group, entries in sorted(groups.items()) if len(entries) >= 2]
    if len(usable) < 2:
        raise ProbeManifestError("at least two groups with two images each are required")

    rows: list[ProbePairRow] = []
    for index, (group, entries) in enumerate(usable):
        next_group, next_entries = usable[(index + 1) % len(usable)]
        pair_index = _label_count(rows, "positive")
        positive = ProbePairRow(
            pair_id=f"p{pair_index:04d}",
            label="positive",
            anchor_id=entries[0].image_id,
            candidate_id=entries[1].image_id,
            anchor_group=group,
            candidate_group=group,
        )
        negative = ProbePairRow(
            pair_id=f"n{pair_index:04d}",
            label="negative",
            anchor_id=entries[0].image_id,
            candidate_id=next_entries[0].image_id,
            anchor_group=group,
            candidate_group=next_group,
        )
        rows.extend((positive, negative))
        if _label_count(rows, "positive") >= pairs_per_label and _label_count(rows, "negative") >= pairs_per_label:
            return tuple(rows[: pairs_per_label * 2])
    return tuple(rows)


def write_probe_manifest(rows: tuple[ProbePairRow, ...], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(asdict(row), ensure_ascii=True) + "\n")


def build_and_write_manifest(
    dataset_root: Path,
    output_path: Path,
    *,
    pairs_per_label: int,
) -> ProbeManifestSummary:
    groups = _group_images(dataset_root)
    rows = build_probe_rows(dataset_root, pairs_per_label=pairs_per_label)
    write_probe_manifest(rows, output_path)
    return ProbeManifestSummary(
        dataset_root=str(dataset_root),
        groups=len(groups),
        usable_groups=sum(1 for entries in groups.values() if len(entries) >= 2),
        pairs=len(rows),
        positive_pairs=_label_count(rows, "positive"),
        negative_pairs=_label_count(rows, "negative"),
        output_path=str(output_path),
    )


def _group_images(dataset_root: Path) -> dict[str, tuple[ProbeImage, ...]]:
    if not dataset_root.is_dir():
        raise ProbeManifestError(f"dataset root is not a directory: {dataset_root}")
    grouped: dict[str, list[ProbeImage]] = {}
    for path in sorted(dataset_root.rglob(f"*{IMAGE_SUFFIX}")):
        caption_path = path.with_suffix(CAPTION_SUFFIX)
        if not caption_path.is_file():
            continue
        relative = path.relative_to(dataset_root)
        image = ProbeImage(
            image_id=relative.with_suffix("").as_posix(),
            group=relative.parent.as_posix(),
        )
        grouped.setdefault(image.group, []).append(image)
    return {group: tuple(entries) for group, entries in grouped.items()}


def _label_count(rows: tuple[ProbePairRow, ...] | list[ProbePairRow], label: PairLabel) -> int:
    return sum(1 for row in rows if row.label == label)


app = typer.Typer(add_completion=False)


@app.command()
def main(
    dataset_root: Annotated[Path, typer.Argument()],
    output_path: Annotated[Path, typer.Argument()],
    pairs_per_label: Annotated[int, typer.Option(min=1)] = 32,
) -> None:
    summary = build_and_write_manifest(
        dataset_root,
        output_path,
        pairs_per_label=pairs_per_label,
    )
    typer.echo(json.dumps(asdict(summary), ensure_ascii=True))


if __name__ == "__main__":
    app()
