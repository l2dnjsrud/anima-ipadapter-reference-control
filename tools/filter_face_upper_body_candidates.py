from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Annotated, Final, Mapping, MutableMapping, Protocol, TypeAlias, TypedDict

import typer
from PIL import Image, ImageDraw

from tools.build_reference_prompt_manifest import (
    DEFAULT_QWENVL_MODEL_ID,
    Qwen3VLReferenceTextScorer,
    Qwen3VLScorerConfig,
)


JsonPrimitive: TypeAlias = str | int | float | bool | None

POSITIVE_TEXTS: Final = (
    "clear face close-up of one character",
    "upper body portrait of one character",
    "single martial arts character bust shot",
    "solo manhwa character face and shoulders",
)
NEGATIVE_TEXTS: Final = (
    "torso only without face",
    "wide group scene with multiple characters",
    "background or building scenery",
    "object or prop close-up without a face",
    "tiny distant character in a scene",
)
ALL_TEXTS: Final = POSITIVE_TEXTS + NEGATIVE_TEXTS


class CandidateManifestRow(TypedDict):
    pair_id: str
    anchor_id: str
    candidate_id: str
    sg_page: str


class CandidateScorer(Protocol):
    def score(self, image_path: Path, candidate_texts: tuple[str, ...]) -> tuple[float, ...]: ...


@dataclass(frozen=True, slots=True)
class SideQualityScore:
    image_id: str
    face_upper_score: float
    positive_max: float
    negative_max: float


@dataclass(frozen=True, slots=True)
class FaceFilteredRow:
    pair_id: str
    anchor_id: str
    candidate_id: str
    sg_page: str
    anchor_face_upper_score: float
    candidate_face_upper_score: float
    decision: str


@dataclass(frozen=True, slots=True)
class FaceFilterSummary:
    input_pairs: int
    kept_pairs: int
    threshold: float


@dataclass(frozen=True, slots=True)
class FaceFilterResult:
    rows: tuple[FaceFilteredRow, ...]
    summary: FaceFilterSummary


@dataclass(frozen=True, slots=True)
class FaceFilterInputError(Exception):
    detail: str

    def __str__(self) -> str:
        return self.detail


def filter_face_candidate_pairs(
    manifest_path: Path,
    *,
    data_root: Path,
    scorer: CandidateScorer,
    threshold: float,
) -> FaceFilterResult:
    cache: dict[str, SideQualityScore] = {}
    rows: list[FaceFilteredRow] = []
    for raw in _read_rows(manifest_path):
        anchor = _side_score(raw["anchor_id"], data_root=data_root, scorer=scorer, cache=cache)
        candidate = _side_score(
            raw["candidate_id"],
            data_root=data_root,
            scorer=scorer,
            cache=cache,
        )
        keep = anchor.face_upper_score >= threshold and candidate.face_upper_score >= threshold
        rows.append(
            FaceFilteredRow(
                pair_id=raw["pair_id"],
                anchor_id=raw["anchor_id"],
                candidate_id=raw["candidate_id"],
                sg_page=raw["sg_page"],
                anchor_face_upper_score=anchor.face_upper_score,
                candidate_face_upper_score=candidate.face_upper_score,
                decision="keep_face_upper_body_pair" if keep else "reject_weak_face_upper_body",
            )
        )
    return FaceFilterResult(
        rows=tuple(rows),
        summary=FaceFilterSummary(
            input_pairs=len(rows),
            kept_pairs=sum(1 for row in rows if row.decision == "keep_face_upper_body_pair"),
            threshold=threshold,
        ),
    )


def write_filter_outputs(
    result: FaceFilterResult,
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
            if row.decision == "keep_face_upper_body_pair":
                handle.write(json.dumps(asdict(row), ensure_ascii=True) + "\n")


def write_filtered_sheet(
    rows: tuple[FaceFilteredRow, ...],
    *,
    data_root: Path,
    sheet_path: Path,
    thumb_size: tuple[int, int] = (220, 220),
) -> None:
    kept = tuple(row for row in rows if row.decision == "keep_face_upper_body_pair")
    margin = 16
    label_h = 40
    row_h = thumb_size[1] + label_h + margin
    width = thumb_size[0] * 2 + margin * 3
    height = margin + max(1, len(kept)) * row_h
    sheet = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(sheet)
    for index, row in enumerate(kept):
        y = margin + index * row_h
        label = f"{row.pair_id} {row.sg_page} {row.anchor_face_upper_score:.2f}/{row.candidate_face_upper_score:.2f}"
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
    cache: MutableMapping[str, SideQualityScore],
) -> SideQualityScore:
    if image_id not in cache:
        scores = scorer.score(data_root / f"{image_id}.jpg", ALL_TEXTS)
        positive_max = max(scores[: len(POSITIVE_TEXTS)])
        negative_max = max(scores[len(POSITIVE_TEXTS) :])
        cache[image_id] = SideQualityScore(
            image_id=image_id,
            face_upper_score=positive_max - negative_max,
            positive_max=positive_max,
            negative_max=negative_max,
        )
    return cache[image_id]


def _read_rows(manifest_path: Path) -> tuple[CandidateManifestRow, ...]:
    rows: list[CandidateManifestRow] = []
    for line_number, raw in _read_jsonl_objects(manifest_path):
        rows.append(
            {
                "pair_id": _require_str(raw, "pair_id", line_number),
                "anchor_id": _require_str(raw, "anchor_id", line_number),
                "candidate_id": _require_str(raw, "candidate_id", line_number),
                "sg_page": _require_str(raw, "sg_page", line_number),
            }
        )
    if not rows:
        raise FaceFilterInputError(f"manifest has no rows: {manifest_path}")
    return tuple(rows)


def _read_jsonl_objects(path: Path) -> tuple[tuple[int, Mapping[str, JsonPrimitive]], ...]:
    rows: list[tuple[int, Mapping[str, JsonPrimitive]]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            raw = json.loads(line)
            if not isinstance(raw, dict):
                raise FaceFilterInputError(f"row {line_number} must be an object")
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
            raise FaceFilterInputError(f"row {line_number} has non-string key")
        parsed[key] = value
    return parsed


def _require_str(raw: Mapping[str, JsonPrimitive], key: str, line_number: int) -> str:
    if key not in raw:
        raise FaceFilterInputError(f"row {line_number} missing field {key}")
    value = raw[key]
    if not isinstance(value, str):
        raise FaceFilterInputError(f"row {line_number} field {key} must be a string")
    return value


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
    result = filter_face_candidate_pairs(
        manifest_path,
        data_root=data_root,
        scorer=scorer,
        threshold=threshold,
    )
    write_filter_outputs(result, scored_output_path=scored_output_path, kept_output_path=kept_output_path)
    if sheet_path is not None:
        write_filtered_sheet(result.rows, data_root=data_root, sheet_path=sheet_path)
    typer.echo(json.dumps(asdict(result.summary), ensure_ascii=True))


if __name__ == "__main__":
    app()
