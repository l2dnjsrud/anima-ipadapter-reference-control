from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from itertools import combinations
from pathlib import Path
from typing import Annotated, Final

import typer
from PIL import Image, ImageDraw


IMAGE_SUFFIX: Final = ".jpg"
CAPTION_SUFFIX: Final = ".txt"
PANEL_KEY_PATTERN: Final = re.compile(
    r"candidate_(?P<candidate>\d+)_SG-(?P<sg>\d{3})-(?P<page>\d+)_page_(?P<size>\d+x\d+)_s(?P<seg>\d+)"
)


@dataclass(frozen=True, slots=True)
class PanelImage:
    image_id: str
    group: str
    panel_key: str
    sg_page: str


@dataclass(frozen=True, slots=True)
class CandidateReviewRow:
    pair_id: str
    anchor_id: str
    candidate_id: str
    anchor_group: str
    candidate_group: str
    anchor_panel_key: str
    candidate_panel_key: str
    sg_page: str
    review_label: str
    notes: str


@dataclass(frozen=True, slots=True)
class CandidateReviewSummary:
    dataset_root: str
    rows: int
    output_path: str
    sheet_path: str | None


@dataclass(frozen=True, slots=True)
class CandidateReviewError(Exception):
    detail: str

    def __str__(self) -> str:
        return self.detail


def build_candidate_rows(dataset_root: Path, *, limit: int) -> tuple[CandidateReviewRow, ...]:
    if limit < 1:
        raise CandidateReviewError("limit must be >= 1")
    by_sg_page = _group_unique_panels(dataset_root)
    rows: list[CandidateReviewRow] = []
    for sg_page, panels in sorted(by_sg_page.items()):
        if len(panels) < 2:
            continue
        for left, right in combinations(panels, 2):
            rows.append(
                CandidateReviewRow(
                    pair_id=f"cand{len(rows):04d}",
                    anchor_id=left.image_id,
                    candidate_id=right.image_id,
                    anchor_group=left.group,
                    candidate_group=right.group,
                    anchor_panel_key=left.panel_key,
                    candidate_panel_key=right.panel_key,
                    sg_page=sg_page,
                    review_label="unlabeled",
                    notes="same_sg_page_non_duplicate_candidate",
                )
            )
            if len(rows) >= limit:
                return tuple(rows)
    if not rows:
        raise CandidateReviewError("no same-page non-duplicate candidates found")
    return tuple(rows)


def write_candidate_manifest(rows: tuple[CandidateReviewRow, ...], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(asdict(row), ensure_ascii=True) + "\n")


def write_candidate_sheet(
    rows: tuple[CandidateReviewRow, ...],
    *,
    dataset_root: Path,
    sheet_path: Path,
    thumb_size: tuple[int, int] = (220, 220),
) -> None:
    margin = 16
    label_h = 34
    row_h = thumb_size[1] + label_h + margin
    col_w = thumb_size[0] * 2 + margin * 3
    width = col_w
    height = margin + len(rows) * row_h
    sheet = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(sheet)
    for index, row in enumerate(rows):
        y = margin + index * row_h
        draw.text((margin, y), f"{row.pair_id} {row.sg_page}", fill=(0, 0, 0))
        for col, image_id in enumerate((row.anchor_id, row.candidate_id)):
            image_path = dataset_root / f"{image_id}.jpg"
            thumb = _fit_thumb(image_path, thumb_size)
            x = margin + col * (thumb_size[0] + margin)
            sheet.paste(thumb, (x, y + label_h))
    sheet_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(sheet_path, quality=92)


def build_review_outputs(
    dataset_root: Path,
    output_path: Path,
    *,
    sheet_path: Path | None,
    limit: int,
) -> CandidateReviewSummary:
    rows = build_candidate_rows(dataset_root, limit=limit)
    write_candidate_manifest(rows, output_path)
    if sheet_path is not None:
        write_candidate_sheet(rows, dataset_root=dataset_root, sheet_path=sheet_path)
    return CandidateReviewSummary(
        dataset_root=str(dataset_root),
        rows=len(rows),
        output_path=str(output_path),
        sheet_path=None if sheet_path is None else str(sheet_path),
    )


def _group_unique_panels(dataset_root: Path) -> dict[str, tuple[PanelImage, ...]]:
    if not dataset_root.is_dir():
        raise CandidateReviewError(f"dataset root is not a directory: {dataset_root}")
    by_sg_page: dict[str, dict[str, PanelImage]] = {}
    for path in sorted(dataset_root.rglob(f"*{IMAGE_SUFFIX}")):
        if not path.with_suffix(CAPTION_SUFFIX).is_file():
            continue
        panel = _parse_panel(path, dataset_root=dataset_root)
        if panel is None:
            continue
        by_sg_page.setdefault(panel.sg_page, {}).setdefault(panel.panel_key, panel)
    return {
        sg_page: tuple(sorted(panels.values(), key=lambda panel: panel.panel_key))
        for sg_page, panels in by_sg_page.items()
    }


def _parse_panel(path: Path, *, dataset_root: Path) -> PanelImage | None:
    matches = PANEL_KEY_PATTERN.findall(path.stem)
    if not matches:
        return None
    candidate, sg, page, size, seg = matches[-1]
    relative = path.relative_to(dataset_root)
    return PanelImage(
        image_id=relative.with_suffix("").as_posix(),
        group=relative.parent.as_posix(),
        panel_key=f"{candidate}_SG-{sg}-{page}_page_{size}_s{seg}",
        sg_page=f"SG-{sg}-{page}",
    )


def _fit_thumb(image_path: Path, size: tuple[int, int]) -> Image.Image:
    with Image.open(image_path) as image:
        thumb = image.convert("RGB")
        thumb.thumbnail(size, Image.Resampling.LANCZOS)
        canvas = Image.new("RGB", size, "white")
        x = (size[0] - thumb.width) // 2
        y = (size[1] - thumb.height) // 2
        canvas.paste(thumb, (x, y))
        return canvas


app = typer.Typer(add_completion=False)


@app.command()
def main(
    dataset_root: Annotated[Path, typer.Argument()],
    output_path: Annotated[Path, typer.Argument()],
    sheet_path: Annotated[Path | None, typer.Option()] = None,
    limit: Annotated[int, typer.Option(min=1)] = 24,
) -> None:
    summary = build_review_outputs(
        dataset_root,
        output_path,
        sheet_path=sheet_path,
        limit=limit,
    )
    typer.echo(json.dumps(asdict(summary), ensure_ascii=True))


if __name__ == "__main__":
    app()
