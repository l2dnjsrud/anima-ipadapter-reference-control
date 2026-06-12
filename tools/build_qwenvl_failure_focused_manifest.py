from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Annotated, Final

import typer

from tools.siglip_auto_caption_types import JsonObject, JsonValue


FAILURE_KEYWORDS: Final[tuple[str, ...]] = (
    "side profile portrait",
    "upper body close-up portrait",
    "raised hand martial arts gesture",
    "arm thrust forward action pose",
    "folding fan in hand",
    "sharp fangs visible",
    "pale purple-skinned villain",
    "red glowing demonic eye",
    "green-skinned demon",
    "green demon",
    "monster",
    "demon",
    "non-human",
)


@dataclass(frozen=True, slots=True)
class ManifestInputError(Exception):
    detail: str

    def __str__(self) -> str:
        return self.detail


@dataclass(frozen=True, slots=True)
class PairRow:
    ref_id: str
    tgt_id: str
    prompt: str

    def to_json(self) -> JsonObject:
        return {"ref_id": self.ref_id, "tgt_id": self.tgt_id, "prompt": self.prompt}


@dataclass(frozen=True, slots=True)
class FailureManifestConfig:
    clean_manifest_path: Path
    positive_manifest_path: Path
    gate_summary_path: Path
    output_manifest_path: Path
    output_summary_path: Path
    repeat_per_failure_row: int = 2


@dataclass(frozen=True, slots=True)
class FailureManifestSummary:
    clean_manifest_path: str
    positive_manifest_path: str
    gate_summary_path: str
    output_manifest_path: str
    output_summary_path: str
    clean32_rows: int
    c052_positive_rows: int
    failure_source_rows: int
    failure_repeated_rows: int
    total_rows: int
    heldout_rows_used: int
    repeat_per_failure_row: int
    failure_keyword_counts: dict[str, int]


def build_failure_focused_manifest(
    config: FailureManifestConfig,
) -> FailureManifestSummary:
    if config.repeat_per_failure_row < 0:
        raise ManifestInputError("repeat_per_failure_row must be >= 0")
    clean_rows = _read_pair_rows(config.clean_manifest_path)
    positive_rows = _read_pair_rows(config.positive_manifest_path, allow_empty=True)
    failure_rows, keyword_counts = _failure_rows(
        clean_rows,
        _train_selected_attributes(config.gate_summary_path),
    )
    rows = [
        *clean_rows,
        *[
            row
            for row in failure_rows
            for _ in range(config.repeat_per_failure_row)
        ],
        *positive_rows,
    ]
    _write_jsonl(config.output_manifest_path, rows)
    summary = FailureManifestSummary(
        clean_manifest_path=str(config.clean_manifest_path),
        positive_manifest_path=str(config.positive_manifest_path),
        gate_summary_path=str(config.gate_summary_path),
        output_manifest_path=str(config.output_manifest_path),
        output_summary_path=str(config.output_summary_path),
        clean32_rows=len(clean_rows),
        c052_positive_rows=len(positive_rows),
        failure_source_rows=len(failure_rows),
        failure_repeated_rows=len(failure_rows) * config.repeat_per_failure_row,
        total_rows=len(rows),
        heldout_rows_used=0,
        repeat_per_failure_row=config.repeat_per_failure_row,
        failure_keyword_counts=keyword_counts,
    )
    config.output_summary_path.parent.mkdir(parents=True, exist_ok=True)
    config.output_summary_path.write_text(
        json.dumps(asdict(summary), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return summary


def _read_pair_rows(path: Path, *, allow_empty: bool = False) -> tuple[PairRow, ...]:
    if not path.is_file():
        raise ManifestInputError(f"manifest not found: {path}")
    rows: list[PairRow] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            raw: JsonValue = json.loads(line)
            if not isinstance(raw, dict):
                raise ManifestInputError(f"{path}:{line_number} row must be an object")
            rows.append(_pair_row(raw, path=path, line_number=line_number))
    if not rows and not allow_empty:
        raise ManifestInputError(f"manifest has no rows: {path}")
    return tuple(rows)


def _pair_row(row: JsonObject, *, path: Path, line_number: int) -> PairRow:
    return PairRow(
        ref_id=_string_field(row, "ref_id", path=path, line_number=line_number),
        tgt_id=_string_field(row, "tgt_id", path=path, line_number=line_number),
        prompt=_string_field(row, "prompt", path=path, line_number=line_number),
    )


def _string_field(
    row: JsonObject,
    field: str,
    *,
    path: Path,
    line_number: int,
) -> str:
    value = row.get(field)
    if not isinstance(value, str):
        raise ManifestInputError(f"{path}:{line_number} missing {field}")
    return value


def _train_selected_attributes(path: Path) -> dict[str, tuple[str, ...]]:
    if not path.is_file():
        raise ManifestInputError(f"gate summary not found: {path}")
    raw: JsonValue = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ManifestInputError(f"gate summary must be an object: {path}")
    samples = raw.get("samples")
    if not isinstance(samples, list):
        raise ManifestInputError(f"gate summary samples must be a list: {path}")
    attrs_by_ref: dict[str, tuple[str, ...]] = {}
    for sample in samples:
        if not isinstance(sample, dict):
            continue
        if sample.get("split") != "train":
            continue
        ref_id = sample.get("ref_id")
        attrs = sample.get("selected_attributes")
        if isinstance(ref_id, str) and isinstance(attrs, list):
            attrs_by_ref[ref_id] = tuple(str(attr) for attr in attrs)
    return attrs_by_ref


def _failure_rows(
    clean_rows: tuple[PairRow, ...],
    attrs_by_ref: dict[str, tuple[str, ...]],
) -> tuple[tuple[PairRow, ...], dict[str, int]]:
    failure_rows: list[PairRow] = []
    keyword_counts = {keyword: 0 for keyword in FAILURE_KEYWORDS}
    for row in clean_rows:
        matched = _matched_keywords(attrs_by_ref.get(row.ref_id, ()))
        if not matched:
            continue
        failure_rows.append(row)
        for keyword in matched:
            keyword_counts[keyword] += 1
    return tuple(failure_rows), {key: count for key, count in keyword_counts.items() if count}


def _matched_keywords(attributes: tuple[str, ...]) -> tuple[str, ...]:
    lowered = tuple(attribute.lower() for attribute in attributes)
    return tuple(
        keyword
        for keyword in FAILURE_KEYWORDS
        if any(keyword in attribute for attribute in lowered)
    )


def _write_jsonl(path: Path, rows: list[PairRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row.to_json(), ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


app = typer.Typer(add_completion=False)


@app.command()
def main(
    clean_manifest_path: Annotated[Path, typer.Option()],
    positive_manifest_path: Annotated[Path, typer.Option()],
    gate_summary_path: Annotated[Path, typer.Option()],
    output_manifest_path: Annotated[Path, typer.Option()],
    output_summary_path: Annotated[Path, typer.Option()],
    repeat_per_failure_row: Annotated[int, typer.Option()] = 2,
) -> None:
    summary = build_failure_focused_manifest(
        FailureManifestConfig(
            clean_manifest_path=clean_manifest_path,
            positive_manifest_path=positive_manifest_path,
            gate_summary_path=gate_summary_path,
            output_manifest_path=output_manifest_path,
            output_summary_path=output_summary_path,
            repeat_per_failure_row=repeat_per_failure_row,
        )
    )
    typer.echo(json.dumps(asdict(summary), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    app()
