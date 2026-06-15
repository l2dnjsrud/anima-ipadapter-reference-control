from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Annotated, Literal, Mapping, TypeAlias, TypedDict

import typer


JsonPrimitive: TypeAlias = str | int | float | bool | None
PairLabel: TypeAlias = Literal["positive", "negative"]


class ReviewedRow(TypedDict):
    pair_id: str
    anchor_id: str
    candidate_id: str
    sg_page: str
    review_label: str
    positive_usable: bool


@dataclass(frozen=True, slots=True)
class PairProbeManifestRow:
    pair_id: str
    label: PairLabel
    anchor_id: str
    candidate_id: str
    anchor_group: str
    candidate_group: str
    sg_page: str
    source_review_label: str


@dataclass(frozen=True, slots=True)
class PairProbeManifestSummary:
    input_rows: int
    output_rows: int
    positive_rows: int
    negative_rows: int


@dataclass(frozen=True, slots=True)
class PairProbeManifestError(Exception):
    detail: str

    def __str__(self) -> str:
        return self.detail


def build_pair_probe_rows(reviewed_path: Path) -> tuple[PairProbeManifestRow, ...]:
    rows: list[PairProbeManifestRow] = []
    for row in _read_reviewed_rows(reviewed_path):
        label = _pair_label(row)
        if label is None:
            continue
        rows.append(
            PairProbeManifestRow(
                pair_id=row["pair_id"],
                label=label,
                anchor_id=row["anchor_id"],
                candidate_id=row["candidate_id"],
                anchor_group=_image_group(row["anchor_id"]),
                candidate_group=_image_group(row["candidate_id"]),
                sg_page=row["sg_page"],
                source_review_label=row["review_label"],
            )
        )
    if not any(row.label == "positive" for row in rows):
        raise PairProbeManifestError("reviewed manifest has no usable positive rows")
    if not any(row.label == "negative" for row in rows):
        raise PairProbeManifestError("reviewed manifest has no negative rows")
    return tuple(rows)


def summarize_pair_probe_rows(
    *,
    input_rows: int,
    rows: tuple[PairProbeManifestRow, ...],
) -> PairProbeManifestSummary:
    return PairProbeManifestSummary(
        input_rows=input_rows,
        output_rows=len(rows),
        positive_rows=sum(1 for row in rows if row.label == "positive"),
        negative_rows=sum(1 for row in rows if row.label == "negative"),
    )


def write_pair_probe_manifest(rows: tuple[PairProbeManifestRow, ...], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(asdict(row), ensure_ascii=True) + "\n")


def _read_reviewed_rows(path: Path) -> tuple[ReviewedRow, ...]:
    rows: list[ReviewedRow] = []
    for line_number, raw in _read_jsonl_objects(path):
        rows.append(
            {
                "pair_id": _require_str(raw, "pair_id", line_number),
                "anchor_id": _require_str(raw, "anchor_id", line_number),
                "candidate_id": _require_str(raw, "candidate_id", line_number),
                "sg_page": _require_str(raw, "sg_page", line_number),
                "review_label": _require_str(raw, "review_label", line_number),
                "positive_usable": _require_bool(raw, "positive_usable", line_number),
            }
        )
    if not rows:
        raise PairProbeManifestError(f"no reviewed rows: {path}")
    return tuple(rows)


def _pair_label(row: ReviewedRow) -> PairLabel | None:
    if row["positive_usable"]:
        return "positive"
    if row["review_label"] == "different_character":
        return "negative"
    return None


def _image_group(image_id: str) -> str:
    return Path(image_id).parent.as_posix()


def _read_jsonl_objects(path: Path) -> tuple[tuple[int, Mapping[str, JsonPrimitive]], ...]:
    rows: list[tuple[int, Mapping[str, JsonPrimitive]]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            raw = json.loads(line)
            if not isinstance(raw, dict):
                raise PairProbeManifestError(f"row {line_number} must be an object")
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
            raise PairProbeManifestError(f"row {line_number} has non-string key")
        parsed[key] = value
    return parsed


def _require_str(raw: Mapping[str, JsonPrimitive], key: str, line_number: int) -> str:
    value = _require_key(raw, key, line_number)
    if not isinstance(value, str):
        raise PairProbeManifestError(f"row {line_number} field {key} must be a string")
    return value


def _require_bool(raw: Mapping[str, JsonPrimitive], key: str, line_number: int) -> bool:
    value = _require_key(raw, key, line_number)
    if not isinstance(value, bool):
        raise PairProbeManifestError(f"row {line_number} field {key} must be a boolean")
    return value


def _require_key(
    raw: Mapping[str, JsonPrimitive],
    key: str,
    line_number: int,
) -> JsonPrimitive:
    if key not in raw:
        raise PairProbeManifestError(f"row {line_number} missing field {key}")
    return raw[key]


app = typer.Typer(add_completion=False)


@app.command()
def main(
    reviewed_path: Annotated[Path, typer.Argument()],
    output_path: Annotated[Path, typer.Argument()],
) -> None:
    input_rows = len(_read_reviewed_rows(reviewed_path))
    rows = build_pair_probe_rows(reviewed_path)
    write_pair_probe_manifest(rows, output_path)
    summary = summarize_pair_probe_rows(input_rows=input_rows, rows=rows)
    typer.echo(json.dumps(asdict(summary), ensure_ascii=True))


if __name__ == "__main__":
    app()
