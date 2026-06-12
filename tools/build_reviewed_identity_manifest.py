from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Annotated, Final, Mapping, TypeAlias, TypedDict

import typer
from PIL import Image, ImageDraw


JsonPrimitive: TypeAlias = str | int | float | bool | None

VALID_LABELS: Final = ("same_character", "different_character", "unclear")
POSITIVE_USABLE_COLOR: Final = (198, 239, 206)
LABEL_COLORS: Final[Mapping[str, tuple[int, int, int]]] = {
    "same_character": (226, 239, 218),
    "different_character": (244, 204, 204),
    "unclear": (255, 242, 204),
}


class CandidateRow(TypedDict):
    pair_id: str
    anchor_id: str
    candidate_id: str
    sg_page: str
    anchor_character_score: float
    candidate_character_score: float


class LabelRow(TypedDict):
    pair_id: str
    review_label: str
    positive_usable: bool
    review_notes: str


@dataclass(frozen=True, slots=True)
class ReviewedIdentityRow:
    pair_id: str
    anchor_id: str
    candidate_id: str
    sg_page: str
    anchor_character_score: float
    candidate_character_score: float
    review_label: str
    positive_usable: bool
    review_notes: str


@dataclass(frozen=True, slots=True)
class ReviewSummary:
    rows: int
    same_character: int
    different_character: int
    unclear: int
    positive_usable: int


@dataclass(frozen=True, slots=True)
class ReviewInputError(Exception):
    detail: str

    def __str__(self) -> str:
        return self.detail


def build_reviewed_rows(candidate_path: Path, label_path: Path) -> tuple[ReviewedIdentityRow, ...]:
    labels = _read_labels(label_path)
    rows: list[ReviewedIdentityRow] = []
    for candidate in _read_candidates(candidate_path):
        label = labels.get(candidate["pair_id"])
        if label is None:
            raise ReviewInputError(f"missing label for pair_id {candidate['pair_id']}")
        rows.append(
            ReviewedIdentityRow(
                pair_id=candidate["pair_id"],
                anchor_id=candidate["anchor_id"],
                candidate_id=candidate["candidate_id"],
                sg_page=candidate["sg_page"],
                anchor_character_score=candidate["anchor_character_score"],
                candidate_character_score=candidate["candidate_character_score"],
                review_label=label["review_label"],
                positive_usable=label["positive_usable"],
                review_notes=label["review_notes"],
            )
        )
    return tuple(rows)


def summarize_reviewed_rows(rows: tuple[ReviewedIdentityRow, ...]) -> ReviewSummary:
    return ReviewSummary(
        rows=len(rows),
        same_character=sum(1 for row in rows if row.review_label == "same_character"),
        different_character=sum(1 for row in rows if row.review_label == "different_character"),
        unclear=sum(1 for row in rows if row.review_label == "unclear"),
        positive_usable=sum(1 for row in rows if row.positive_usable),
    )


def write_reviewed_manifest(rows: tuple[ReviewedIdentityRow, ...], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(asdict(row), ensure_ascii=True) + "\n")


def write_reviewed_sheet(
    rows: tuple[ReviewedIdentityRow, ...],
    *,
    data_root: Path,
    sheet_path: Path,
    thumb_size: tuple[int, int] = (220, 220),
) -> None:
    margin = 16
    label_h = 52
    row_h = thumb_size[1] + label_h + margin
    width = thumb_size[0] * 2 + margin * 3
    height = margin + max(1, len(rows)) * row_h
    sheet = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(sheet)
    for index, row in enumerate(rows):
        y = margin + index * row_h
        label = f"{row.pair_id} {row.review_label} usable={row.positive_usable}"
        draw.rectangle((margin, y, width - margin, y + label_h - 8), fill=_label_color(row))
        draw.text((margin + 4, y + 4), label, fill=(0, 0, 0))
        draw.text((margin + 4, y + 22), row.sg_page, fill=(0, 0, 0))
        for col, image_id in enumerate((row.anchor_id, row.candidate_id)):
            thumb = _fit_thumb(data_root / f"{image_id}.jpg", thumb_size)
            x = margin + col * (thumb_size[0] + margin)
            sheet.paste(thumb, (x, y + label_h))
    sheet_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(sheet_path, quality=92)


def _read_candidates(path: Path) -> tuple[CandidateRow, ...]:
    rows: list[CandidateRow] = []
    for line_number, raw in _read_jsonl_objects(path):
        rows.append(
            {
                "pair_id": _require_str(raw, "pair_id", line_number),
                "anchor_id": _require_str(raw, "anchor_id", line_number),
                "candidate_id": _require_str(raw, "candidate_id", line_number),
                "sg_page": _require_str(raw, "sg_page", line_number),
                "anchor_character_score": _require_score(raw, "anchor", line_number),
                "candidate_character_score": _require_score(raw, "candidate", line_number),
            }
        )
    if not rows:
        raise ReviewInputError(f"no candidate rows: {path}")
    return tuple(rows)


def _read_labels(path: Path) -> Mapping[str, LabelRow]:
    labels: dict[str, LabelRow] = {}
    for line_number, raw in _read_jsonl_objects(path):
        pair_id = _require_str(raw, "pair_id", line_number)
        review_label = _require_str(raw, "review_label", line_number)
        if review_label not in VALID_LABELS:
            raise ReviewInputError(f"row {line_number} has invalid review_label {review_label}")
        labels[pair_id] = {
            "pair_id": pair_id,
            "review_label": review_label,
            "positive_usable": _require_bool(raw, "positive_usable", line_number),
            "review_notes": _require_str(raw, "review_notes", line_number),
        }
    return labels


def _read_jsonl_objects(path: Path) -> tuple[tuple[int, Mapping[str, JsonPrimitive]], ...]:
    rows: list[tuple[int, Mapping[str, JsonPrimitive]]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            raw = json.loads(line)
            if not isinstance(raw, dict):
                raise ReviewInputError(f"row {line_number} must be an object")
            rows.append((line_number, _parse_jsonl_row(raw, line_number=line_number)))
    return tuple(rows)


def _parse_jsonl_row(
    raw: Mapping[JsonPrimitive, JsonPrimitive],
    *,
    line_number: int,
) -> Mapping[str, JsonPrimitive]:
    parsed: dict[str, JsonPrimitive] = {}
    for key, value in raw.items():
        if not isinstance(key, str):
            raise ReviewInputError(f"row {line_number} has non-string key")
        parsed[key] = value
    return parsed


def _require_str(raw: Mapping[str, JsonPrimitive], key: str, line_number: int) -> str:
    value = _require_key(raw, key, line_number)
    if not isinstance(value, str):
        raise ReviewInputError(f"row {line_number} field {key} must be a string")
    return value


def _require_float(raw: Mapping[str, JsonPrimitive], key: str, line_number: int) -> float:
    value = _require_key(raw, key, line_number)
    if isinstance(value, bool):
        raise ReviewInputError(f"row {line_number} field {key} must be numeric")
    if not isinstance(value, int | float):
        raise ReviewInputError(f"row {line_number} field {key} must be numeric")
    return float(value)


def _require_score(raw: Mapping[str, JsonPrimitive], side: str, line_number: int) -> float:
    character_key = f"{side}_character_score"
    face_key = f"{side}_face_upper_score"
    if character_key in raw:
        return _require_float(raw, character_key, line_number)
    return _require_float(raw, face_key, line_number)


def _require_bool(raw: Mapping[str, JsonPrimitive], key: str, line_number: int) -> bool:
    value = _require_key(raw, key, line_number)
    if not isinstance(value, bool):
        raise ReviewInputError(f"row {line_number} field {key} must be a boolean")
    return value


def _require_key(
    raw: Mapping[str, JsonPrimitive],
    key: str,
    line_number: int,
) -> JsonPrimitive:
    if key not in raw:
        raise ReviewInputError(f"row {line_number} missing field {key}")
    return raw[key]


def _label_color(row: ReviewedIdentityRow) -> tuple[int, int, int]:
    if row.positive_usable:
        return POSITIVE_USABLE_COLOR
    return LABEL_COLORS[row.review_label]


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
    candidate_path: Annotated[Path, typer.Argument()],
    label_path: Annotated[Path, typer.Argument()],
    output_path: Annotated[Path, typer.Argument()],
    data_root: Annotated[Path, typer.Argument()],
    sheet_path: Annotated[Path | None, typer.Option()] = None,
) -> None:
    rows = build_reviewed_rows(candidate_path, label_path)
    write_reviewed_manifest(rows, output_path)
    if sheet_path is not None:
        write_reviewed_sheet(rows, data_root=data_root, sheet_path=sheet_path)
    typer.echo(json.dumps(asdict(summarize_reviewed_rows(rows)), ensure_ascii=True))


if __name__ == "__main__":
    app()
