from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Annotated, Final, Literal

import typer


IMAGE_SUFFIX: Final = ".jpg"
CAPTION_SUFFIX: Final = ".txt"
PANEL_KEY_PATTERN: Final = re.compile(
    r"candidate_(?P<key>\d+_SG-\d{3}-\d+_page_\d+x\d+_s\d+)"
)
PairLabel = Literal["positive", "negative"]


@dataclass(frozen=True, slots=True)
class StrictProbeImage:
    image_id: str
    group: str
    panel_key: str


@dataclass(frozen=True, slots=True)
class StrictProbePairRow:
    pair_id: str
    label: PairLabel
    anchor_id: str
    candidate_id: str
    anchor_group: str
    candidate_group: str
    anchor_panel_key: str
    candidate_panel_key: str
    mining_rule: str


@dataclass(frozen=True, slots=True)
class StrictProbeManifestSummary:
    dataset_root: str
    groups: int
    panel_keys: int
    duplicate_panel_keys: int
    pairs: int
    positive_pairs: int
    negative_pairs: int
    output_path: str


@dataclass(frozen=True, slots=True)
class StrictProbeManifestError(Exception):
    detail: str

    def __str__(self) -> str:
        return self.detail


def build_strict_probe_rows(
    dataset_root: Path,
    *,
    pairs_per_label: int,
) -> tuple[StrictProbePairRow, ...]:
    if pairs_per_label < 1:
        raise StrictProbeManifestError("pairs_per_label must be >= 1")
    images_by_panel = _group_images_by_panel(dataset_root)
    duplicates = [
        (panel_key, entries)
        for panel_key, entries in sorted(images_by_panel.items())
        if len(entries) >= 2
    ]
    if not duplicates:
        raise StrictProbeManifestError("no duplicate panel keys found")

    rows: list[StrictProbePairRow] = []
    for panel_key, entries in duplicates:
        anchor = entries[0]
        positive = entries[1]
        negative = _select_same_group_negative(
            anchor,
            images_by_panel=images_by_panel,
        )
        if negative is None:
            continue
        rows.append(
            StrictProbePairRow(
                pair_id=f"p{_label_count(rows, 'positive'):04d}",
                label="positive",
                anchor_id=anchor.image_id,
                candidate_id=positive.image_id,
                anchor_group=anchor.group,
                candidate_group=positive.group,
                anchor_panel_key=panel_key,
                candidate_panel_key=panel_key,
                mining_rule="same_panel_duplicate",
            )
        )
        rows.append(
            StrictProbePairRow(
                pair_id=f"n{_label_count(rows, 'negative'):04d}",
                label="negative",
                anchor_id=anchor.image_id,
                candidate_id=negative.image_id,
                anchor_group=anchor.group,
                candidate_group=negative.group,
                anchor_panel_key=panel_key,
                candidate_panel_key=negative.panel_key,
                mining_rule="same_group_different_panel",
            )
        )
        if _label_count(rows, "positive") >= pairs_per_label:
            return tuple(rows[: pairs_per_label * 2])
    if not rows:
        raise StrictProbeManifestError("no duplicate panels had same-group negatives")
    return tuple(rows)


def build_and_write_manifest(
    dataset_root: Path,
    output_path: Path,
    *,
    pairs_per_label: int,
) -> StrictProbeManifestSummary:
    images_by_panel = _group_images_by_panel(dataset_root)
    rows = build_strict_probe_rows(dataset_root, pairs_per_label=pairs_per_label)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(asdict(row), ensure_ascii=True) + "\n")
    return StrictProbeManifestSummary(
        dataset_root=str(dataset_root),
        groups=len({entry.group for entries in images_by_panel.values() for entry in entries}),
        panel_keys=len(images_by_panel),
        duplicate_panel_keys=sum(1 for entries in images_by_panel.values() if len(entries) >= 2),
        pairs=len(rows),
        positive_pairs=_label_count(rows, "positive"),
        negative_pairs=_label_count(rows, "negative"),
        output_path=str(output_path),
    )


def _group_images_by_panel(dataset_root: Path) -> dict[str, tuple[StrictProbeImage, ...]]:
    if not dataset_root.is_dir():
        raise StrictProbeManifestError(f"dataset root is not a directory: {dataset_root}")
    grouped: dict[str, list[StrictProbeImage]] = {}
    for path in sorted(dataset_root.rglob(f"*{IMAGE_SUFFIX}")):
        if not path.with_suffix(CAPTION_SUFFIX).is_file():
            continue
        panel_key = _parse_panel_key(path.stem)
        if panel_key is None:
            continue
        relative = path.relative_to(dataset_root)
        grouped.setdefault(panel_key, []).append(
            StrictProbeImage(
                image_id=relative.with_suffix("").as_posix(),
                group=relative.parent.as_posix(),
                panel_key=panel_key,
            )
        )
    return {panel_key: tuple(entries) for panel_key, entries in grouped.items()}


def _parse_panel_key(stem: str) -> str | None:
    matches = PANEL_KEY_PATTERN.findall(stem)
    if not matches:
        return None
    return matches[-1]


def _select_same_group_negative(
    anchor: StrictProbeImage,
    *,
    images_by_panel: dict[str, tuple[StrictProbeImage, ...]],
) -> StrictProbeImage | None:
    for panel_key, entries in sorted(images_by_panel.items()):
        if panel_key == anchor.panel_key:
            continue
        for candidate in entries:
            if candidate.group == anchor.group:
                return candidate
    return None


def _label_count(
    rows: tuple[StrictProbePairRow, ...] | list[StrictProbePairRow],
    label: PairLabel,
) -> int:
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
