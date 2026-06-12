from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Mapping, MutableMapping, TypeAlias, TypedDict

from PIL import Image, ImageDraw
import typer

from tools.image_feature_embedders import EncoderName, ImageEmbedder, build_image_embedder

if TYPE_CHECKING:
    import torch


JsonPrimitive: TypeAlias = str | int | float | bool | None


class CandidateRow(TypedDict):
    pair_id: str
    anchor_id: str
    candidate_id: str
    sg_page: str


@dataclass(frozen=True, slots=True)
class RankedCandidateRow:
    pair_id: str
    anchor_id: str
    candidate_id: str
    sg_page: str
    similarity: float
    rank: int


@dataclass(frozen=True, slots=True)
class RankSummary:
    input_pairs: int
    top_pairs: int
    max_similarity: float
    min_similarity: float


@dataclass(frozen=True, slots=True)
class RankResult:
    rows: tuple[RankedCandidateRow, ...]
    summary: RankSummary


@dataclass(frozen=True, slots=True)
class RankInputError(Exception):
    detail: str

    def __str__(self) -> str:
        return self.detail


def rank_candidate_pairs(
    manifest_path: Path,
    *,
    data_root: Path,
    embedder: ImageEmbedder,
    top_k: int,
) -> RankResult:
    if top_k < 1:
        raise RankInputError("top_k must be >= 1")
    cache: dict[Path, torch.Tensor] = {}
    scored = [
        _score_row(row, data_root=data_root, embedder=embedder, cache=cache)
        for row in _read_rows(manifest_path)
    ]
    ordered = tuple(
        RankedCandidateRow(
            pair_id=row.pair_id,
            anchor_id=row.anchor_id,
            candidate_id=row.candidate_id,
            sg_page=row.sg_page,
            similarity=row.similarity,
            rank=index + 1,
        )
        for index, row in enumerate(
            sorted(scored, key=lambda row: (row.similarity, row.pair_id), reverse=True)
        )
    )
    return RankResult(
        rows=ordered,
        summary=RankSummary(
            input_pairs=len(ordered),
            top_pairs=min(top_k, len(ordered)),
            max_similarity=ordered[0].similarity,
            min_similarity=ordered[-1].similarity,
        ),
    )


def write_rank_outputs(
    result: RankResult,
    *,
    scored_output_path: Path,
    top_output_path: Path,
) -> None:
    scored_output_path.parent.mkdir(parents=True, exist_ok=True)
    with scored_output_path.open("w", encoding="utf-8") as handle:
        for row in result.rows:
            handle.write(json.dumps(asdict(row), ensure_ascii=True) + "\n")
    with top_output_path.open("w", encoding="utf-8") as handle:
        for row in result.rows[: result.summary.top_pairs]:
            handle.write(json.dumps(asdict(row), ensure_ascii=True) + "\n")


def write_rank_sheet(
    rows: tuple[RankedCandidateRow, ...],
    *,
    data_root: Path,
    sheet_path: Path,
    thumb_size: tuple[int, int] = (220, 220),
) -> None:
    margin = 16
    label_h = 42
    row_h = thumb_size[1] + label_h + margin
    width = thumb_size[0] * 2 + margin * 3
    height = margin + max(1, len(rows)) * row_h
    sheet = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(sheet)
    for index, row in enumerate(rows):
        y = margin + index * row_h
        label = f"#{row.rank} {row.pair_id} {row.sg_page} sim={row.similarity:.4f}"
        draw.text((margin, y), label, fill=(0, 0, 0))
        for col, image_id in enumerate((row.anchor_id, row.candidate_id)):
            thumb = _fit_thumb(data_root / f"{image_id}.jpg", thumb_size)
            x = margin + col * (thumb_size[0] + margin)
            sheet.paste(thumb, (x, y + label_h))
    sheet_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(sheet_path, quality=92)


def _score_row(
    row: CandidateRow,
    *,
    data_root: Path,
    embedder: ImageEmbedder,
    cache: MutableMapping[Path, torch.Tensor],
) -> RankedCandidateRow:
    anchor_path = data_root / f"{row['anchor_id']}.jpg"
    candidate_path = data_root / f"{row['candidate_id']}.jpg"
    return RankedCandidateRow(
        pair_id=row["pair_id"],
        anchor_id=row["anchor_id"],
        candidate_id=row["candidate_id"],
        sg_page=row["sg_page"],
        similarity=_cosine(
            _embedding_cached(embedder, cache, anchor_path),
            _embedding_cached(embedder, cache, candidate_path),
        ),
        rank=0,
    )


def _embedding_cached(
    embedder: ImageEmbedder,
    cache: MutableMapping[Path, torch.Tensor],
    image_path: Path,
) -> torch.Tensor:
    if image_path not in cache:
        cache[image_path] = embedder.encode_image(image_path)
    return cache[image_path]


def _cosine(left: torch.Tensor, right: torch.Tensor) -> float:
    import torch

    return float(torch.nn.functional.cosine_similarity(left, right, dim=0).item())


def _read_rows(manifest_path: Path) -> tuple[CandidateRow, ...]:
    rows: list[CandidateRow] = []
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
        raise RankInputError(f"manifest has no rows: {manifest_path}")
    return tuple(rows)


def _read_jsonl_objects(path: Path) -> tuple[tuple[int, Mapping[str, JsonPrimitive]], ...]:
    rows: list[tuple[int, Mapping[str, JsonPrimitive]]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            raw = json.loads(line)
            if not isinstance(raw, dict):
                raise RankInputError(f"row {line_number} must be an object")
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
            raise RankInputError(f"row {line_number} has non-string key")
        parsed[key] = value
    return parsed


def _require_str(raw: Mapping[str, JsonPrimitive], key: str, line_number: int) -> str:
    if key not in raw:
        raise RankInputError(f"row {line_number} missing field {key}")
    value = raw[key]
    if not isinstance(value, str):
        raise RankInputError(f"row {line_number} field {key} must be a string")
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
    top_output_path: Annotated[Path, typer.Argument()],
    sheet_path: Annotated[Path | None, typer.Option()] = None,
    top_k: Annotated[int, typer.Option(min=1)] = 64,
    encoder: Annotated[EncoderName, typer.Option()] = "qwenvl",
    model_id: Annotated[str | None, typer.Option()] = None,
    device: Annotated[str, typer.Option()] = "auto",
) -> None:
    encoder_name, embedder = build_image_embedder(encoder, model_id=model_id, device=device)
    result = rank_candidate_pairs(
        manifest_path,
        data_root=data_root,
        embedder=embedder,
        top_k=top_k,
    )
    write_rank_outputs(
        result,
        scored_output_path=scored_output_path,
        top_output_path=top_output_path,
    )
    if sheet_path is not None:
        write_rank_sheet(
            result.rows[: result.summary.top_pairs],
            data_root=data_root,
            sheet_path=sheet_path,
        )
    typer.echo(json.dumps({"encoder": encoder_name, **asdict(result.summary)}, ensure_ascii=True))


if __name__ == "__main__":
    app()
