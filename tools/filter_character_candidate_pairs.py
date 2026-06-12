from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Annotated, Final, Mapping, MutableMapping, Protocol, TypedDict

import typer
from PIL import Image, ImageDraw

from tools.build_reference_prompt_manifest import (
    DEFAULT_QWENVL_MODEL_ID,
    Qwen3VLReferenceTextScorer,
    Qwen3VLScorerConfig,
)


CHARACTER_TEXTS: Final = (
    "solo character portrait",
    "upper body character close-up",
    "human martial arts character face",
    "single anime manhwa character",
)
NON_CHARACTER_TEXTS: Final = (
    "background scenery without a character",
    "object prop close-up without a face",
    "building interior or exterior panel",
    "wide group scene with multiple characters",
)
ALL_TEXTS: Final = CHARACTER_TEXTS + NON_CHARACTER_TEXTS


class CandidateManifestRow(TypedDict):
    pair_id: str
    anchor_id: str
    candidate_id: str
    sg_page: str


class CandidateScorer(Protocol):
    def score(self, image_path: Path, candidate_texts: tuple[str, ...]) -> tuple[float, ...]: ...


@dataclass(frozen=True, slots=True)
class CharacterSideScore:
    image_id: str
    character_score: float
    character_max: float
    non_character_max: float


@dataclass(frozen=True, slots=True)
class FilteredCandidateRow:
    pair_id: str
    anchor_id: str
    candidate_id: str
    sg_page: str
    anchor_character_score: float
    candidate_character_score: float
    decision: str


@dataclass(frozen=True, slots=True)
class FilterSummary:
    input_pairs: int
    kept_pairs: int
    threshold: float


@dataclass(frozen=True, slots=True)
class FilterResult:
    rows: tuple[FilteredCandidateRow, ...]
    summary: FilterSummary


@dataclass(frozen=True, slots=True)
class FilterInputError(Exception):
    detail: str

    def __str__(self) -> str:
        return self.detail


def filter_candidate_pairs(
    manifest_path: Path,
    *,
    data_root: Path,
    scorer: CandidateScorer,
    threshold: float,
) -> FilterResult:
    cache: dict[str, CharacterSideScore] = {}
    rows: list[FilteredCandidateRow] = []
    for raw in _read_rows(manifest_path):
        anchor_id = str(raw["anchor_id"])
        candidate_id = str(raw["candidate_id"])
        anchor_score = _side_score(anchor_id, data_root=data_root, scorer=scorer, cache=cache)
        candidate_score = _side_score(candidate_id, data_root=data_root, scorer=scorer, cache=cache)
        keep = (
            anchor_score.character_score >= threshold
            and candidate_score.character_score >= threshold
        )
        rows.append(
            FilteredCandidateRow(
                pair_id=str(raw["pair_id"]),
                anchor_id=anchor_id,
                candidate_id=candidate_id,
                sg_page=str(raw["sg_page"]),
                anchor_character_score=anchor_score.character_score,
                candidate_character_score=candidate_score.character_score,
                decision="keep_character_pair_candidate" if keep else "reject_non_character_pair",
            )
        )
    return FilterResult(
        rows=tuple(rows),
        summary=FilterSummary(
            input_pairs=len(rows),
            kept_pairs=sum(1 for row in rows if row.decision == "keep_character_pair_candidate"),
            threshold=threshold,
        ),
    )


def write_filter_outputs(
    result: FilterResult,
    *,
    scored_output_path: Path,
    kept_output_path: Path,
) -> None:
    scored_output_path.parent.mkdir(parents=True, exist_ok=True)
    with scored_output_path.open("w", encoding="utf-8") as handle:
        for row in result.rows:
            handle.write(json.dumps(asdict(row), ensure_ascii=True) + "\n")
    with kept_output_path.open("w", encoding="utf-8") as handle:
        for row in result.rows:
            if row.decision == "keep_character_pair_candidate":
                handle.write(json.dumps(asdict(row), ensure_ascii=True) + "\n")


def write_filtered_sheet(
    rows: tuple[FilteredCandidateRow, ...],
    *,
    data_root: Path,
    sheet_path: Path,
    thumb_size: tuple[int, int] = (220, 220),
) -> None:
    kept = tuple(row for row in rows if row.decision == "keep_character_pair_candidate")
    margin = 16
    label_h = 38
    row_h = thumb_size[1] + label_h + margin
    width = thumb_size[0] * 2 + margin * 3
    height = margin + max(1, len(kept)) * row_h
    sheet = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(sheet)
    for index, row in enumerate(kept):
        y = margin + index * row_h
        label = f"{row.pair_id} {row.sg_page} {row.anchor_character_score:.2f}/{row.candidate_character_score:.2f}"
        draw.text((margin, y), label, fill=(0, 0, 0))
        for col, image_id in enumerate((row.anchor_id, row.candidate_id)):
            thumb = _fit_thumb(data_root / f"{image_id}.jpg", thumb_size)
            x = margin + col * (thumb_size[0] + margin)
            sheet.paste(thumb, (x, y + label_h))
    sheet_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(sheet_path, quality=92)


def _side_score(
    image_id: str,
    *,
    data_root: Path,
    scorer: CandidateScorer,
    cache: MutableMapping[str, CharacterSideScore],
) -> CharacterSideScore:
    if image_id not in cache:
        scores = scorer.score(data_root / f"{image_id}.jpg", ALL_TEXTS)
        character_max = max(scores[: len(CHARACTER_TEXTS)])
        non_character_max = max(scores[len(CHARACTER_TEXTS) :])
        cache[image_id] = CharacterSideScore(
            image_id=image_id,
            character_score=character_max - non_character_max,
            character_max=character_max,
            non_character_max=non_character_max,
        )
    return cache[image_id]


def _read_rows(manifest_path: Path) -> tuple[CandidateManifestRow, ...]:
    rows: list[CandidateManifestRow] = []
    with manifest_path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            raw = json.loads(line)
            if not isinstance(raw, dict):
                raise FilterInputError(f"row {line_number} must be an object")
            parsed = _parse_manifest_row(raw, line_number=line_number)
            rows.append(parsed)
    if not rows:
        raise FilterInputError(f"manifest has no rows: {manifest_path}")
    return tuple(rows)


def _parse_manifest_row(raw: Mapping[str, object], *, line_number: int) -> CandidateManifestRow:
    try:
        return {
            "pair_id": str(raw["pair_id"]),
            "anchor_id": str(raw["anchor_id"]),
            "candidate_id": str(raw["candidate_id"]),
            "sg_page": str(raw["sg_page"]),
        }
    except KeyError as error:
        missing_key = str(error.args[0])
        raise FilterInputError(f"row {line_number} is missing {missing_key}") from error


def _fit_thumb(image_path: Path, size: tuple[int, int]) -> Image.Image:
    with Image.open(image_path) as image:
        thumb = image.convert("RGB")
        thumb.thumbnail(size, Image.Resampling.LANCZOS)
        canvas = Image.new("RGB", size, "white")
        canvas.paste(thumb, ((size[0] - thumb.width) // 2, (size[1] - thumb.height) // 2))
        return canvas


app = typer.Typer(add_completion=False)


@app.command()
def main(
    manifest_path: Annotated[Path, typer.Argument()],
    data_root: Annotated[Path, typer.Argument()],
    scored_output_path: Annotated[Path, typer.Argument()],
    kept_output_path: Annotated[Path, typer.Argument()],
    sheet_path: Annotated[Path | None, typer.Option()] = None,
    threshold: Annotated[float, typer.Option()] = 0.0,
    model_id: Annotated[str, typer.Option()] = DEFAULT_QWENVL_MODEL_ID,
) -> None:
    scorer = Qwen3VLReferenceTextScorer(Qwen3VLScorerConfig(model_id=model_id))
    result = filter_candidate_pairs(
        manifest_path,
        data_root=data_root,
        scorer=scorer,
        threshold=threshold,
    )
    write_filter_outputs(
        result,
        scored_output_path=scored_output_path,
        kept_output_path=kept_output_path,
    )
    if sheet_path is not None:
        write_filtered_sheet(result.rows, data_root=data_root, sheet_path=sheet_path)
    typer.echo(json.dumps(asdict(result.summary), ensure_ascii=True))


if __name__ == "__main__":
    app()
