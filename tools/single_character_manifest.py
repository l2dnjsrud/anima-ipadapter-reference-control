from __future__ import annotations

import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Annotated, Final, Sequence

import typer
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.pair_manifest_types import PairRow  # noqa: E402


DEFAULT_PROMPT: Final = "mrcolor_panel_style, character panel, close-up panel"
TARGET_RATIO: Final = 0.72


@dataclass(frozen=True, slots=True)
class SingleCharacterSelectionConfig:
    dataset_root: Path
    train_count: int
    heldout_count: int
    min_ratio: float = 0.45
    max_ratio: float = 1.15
    min_short_edge: int = 512
    max_long_edge: int = 3200

    @property
    def total_count(self) -> int:
        return self.train_count + self.heldout_count


@dataclass(frozen=True, slots=True)
class SingleCharacterCandidate:
    image_id: str
    image_path: Path
    caption: str
    width: int
    height: int
    score: float


@dataclass(frozen=True, slots=True)
class SingleCharacterSelectionSummary:
    dataset_root: str
    candidates_scanned: int
    candidates_kept: int
    train_rows: int
    heldout_rows: int
    min_ratio: float
    max_ratio: float
    min_short_edge: int
    max_long_edge: int


@dataclass(frozen=True, slots=True)
class SingleCharacterSelectionResult:
    candidates: list[SingleCharacterCandidate]
    train_rows: list[PairRow]
    heldout_rows: list[PairRow]
    summary: SingleCharacterSelectionSummary


app = typer.Typer(add_completion=False)


def select_single_character_rows(
    config: SingleCharacterSelectionConfig,
) -> SingleCharacterSelectionResult:
    candidates = _collect_candidates(config)
    selected = _evenly_spaced(_best_per_directory(candidates), config.total_count)
    heldout_positions = set(_evenly_spaced_indices(len(selected), config.heldout_count))
    heldout_candidates = [
        candidate for index, candidate in enumerate(selected) if index in heldout_positions
    ]
    train_candidates = [
        candidate for index, candidate in enumerate(selected) if index not in heldout_positions
    ][: config.train_count]
    return SingleCharacterSelectionResult(
        candidates=selected,
        train_rows=[_row_for_candidate(candidate) for candidate in train_candidates],
        heldout_rows=[_row_for_candidate(candidate) for candidate in heldout_candidates],
        summary=SingleCharacterSelectionSummary(
            dataset_root=str(config.dataset_root),
            candidates_scanned=_count_images(config.dataset_root),
            candidates_kept=len(candidates),
            train_rows=len(train_candidates),
            heldout_rows=len(heldout_candidates),
            min_ratio=config.min_ratio,
            max_ratio=config.max_ratio,
            min_short_edge=config.min_short_edge,
            max_long_edge=config.max_long_edge,
        ),
    )


def write_pair_rows(rows: Sequence[PairRow], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(asdict(row), ensure_ascii=True) + "\n")


def write_summary(summary: SingleCharacterSelectionSummary, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(asdict(summary), ensure_ascii=True, indent=2) + "\n",
        encoding="utf-8",
    )


def write_candidate_sheet(
    candidates: Sequence[SingleCharacterCandidate],
    output_path: Path,
    *,
    thumb_size: tuple[int, int] = (128, 128),
    columns: int = 8,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    label_height = 40
    rows = math.ceil(len(candidates) / columns)
    sheet = Image.new(
        "RGB",
        (columns * thumb_size[0], rows * (thumb_size[1] + label_height)),
        "white",
    )
    draw = ImageDraw.Draw(sheet)
    for index, candidate in enumerate(candidates):
        col = index % columns
        row = index // columns
        x = col * thumb_size[0]
        y = row * (thumb_size[1] + label_height)
        with Image.open(candidate.image_path) as image:
            thumb = image.convert("RGB")
            thumb.thumbnail(thumb_size)
        sheet.paste(thumb, (x, y + label_height))
        draw.text((x + 4, y + 4), _short_label(candidate.image_id), fill="black")
        draw.text((x + 4, y + 20), f"{candidate.width}x{candidate.height}", fill="black")
    sheet.save(output_path)


def _collect_candidates(
    config: SingleCharacterSelectionConfig,
) -> list[SingleCharacterCandidate]:
    candidates: list[SingleCharacterCandidate] = []
    for image_path in sorted(config.dataset_root.rglob("*.jpg")):
        caption_path = image_path.with_suffix(".txt")
        if not caption_path.is_file():
            continue
        caption = caption_path.read_text(encoding="utf-8").strip() or DEFAULT_PROMPT
        with Image.open(image_path) as image:
            width, height = image.size
        ratio = width / height
        short_edge = min(width, height)
        long_edge = max(width, height)
        if not _is_single_character_caption(caption):
            continue
        if ratio < config.min_ratio or ratio > config.max_ratio:
            continue
        if short_edge < config.min_short_edge or long_edge > config.max_long_edge:
            continue
        image_id = image_path.relative_to(config.dataset_root).with_suffix("").as_posix()
        candidates.append(
            SingleCharacterCandidate(
                image_id=image_id,
                image_path=image_path,
                caption=caption,
                width=width,
                height=height,
                score=abs(ratio - TARGET_RATIO),
            )
        )
    return candidates


def _best_per_directory(
    candidates: Sequence[SingleCharacterCandidate],
) -> list[SingleCharacterCandidate]:
    by_directory: dict[str, SingleCharacterCandidate] = {}
    for candidate in candidates:
        directory = Path(candidate.image_id).parent.as_posix()
        current = by_directory.get(directory)
        if current is None or (candidate.score, candidate.image_id) < (
            current.score,
            current.image_id,
        ):
            by_directory[directory] = candidate
    return [by_directory[key] for key in sorted(by_directory)]


def _evenly_spaced(
    candidates: Sequence[SingleCharacterCandidate],
    count: int,
) -> list[SingleCharacterCandidate]:
    return [candidates[index] for index in _evenly_spaced_indices(len(candidates), count)]


def _evenly_spaced_indices(length: int, count: int) -> list[int]:
    if count <= 0 or length <= 0:
        return []
    capped = min(count, length)
    return [math.floor(index * length / capped) for index in range(capped)]


def _row_for_candidate(candidate: SingleCharacterCandidate) -> PairRow:
    return PairRow(
        ref_id=candidate.image_id,
        tgt_id=candidate.image_id,
        prompt=candidate.caption,
    )


def _is_single_character_caption(caption: str) -> bool:
    normalized = caption.lower()
    return (
        "character panel" in normalized
        and "wide panel" not in normalized
        and "background panel" not in normalized
    )


def _count_images(dataset_root: Path) -> int:
    return sum(1 for path in dataset_root.rglob("*.jpg") if path.is_file())


def _short_label(image_id: str) -> str:
    parts = image_id.split("/")
    suffix = parts[-1]
    prefix = parts[-2] if len(parts) > 1 else ""
    return f"{prefix}/{suffix}"[:30]


@app.command()
def main(
    dataset_root: Annotated[Path, typer.Argument()],
    train_output: Annotated[Path, typer.Option()],
    heldout_output: Annotated[Path, typer.Option()],
    summary_output: Annotated[Path, typer.Option()],
    sheet_output: Annotated[Path | None, typer.Option()] = None,
    train_count: Annotated[int, typer.Option(min=1)] = 64,
    heldout_count: Annotated[int, typer.Option(min=1)] = 16,
) -> None:
    result = select_single_character_rows(
        SingleCharacterSelectionConfig(
            dataset_root=dataset_root,
            train_count=train_count,
            heldout_count=heldout_count,
        )
    )
    write_pair_rows(result.train_rows, train_output)
    write_pair_rows(result.heldout_rows, heldout_output)
    write_summary(result.summary, summary_output)
    if sheet_output is not None:
        write_candidate_sheet(result.candidates, sheet_output)
    typer.echo(json.dumps(asdict(result.summary), ensure_ascii=True))


if __name__ == "__main__":
    app()
