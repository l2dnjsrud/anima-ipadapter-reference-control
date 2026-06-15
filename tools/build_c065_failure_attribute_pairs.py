from __future__ import annotations

# /// script
# requires-python = ">=3.13"
# dependencies = ["typer"]
# ///
# ─── How to run ───
# PYTHONPATH=. python tools/build_c065_failure_attribute_pairs.py --help

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Annotated, Final, Literal

import typer

from tools.siglip_auto_caption_types import JsonObject, JsonValue

PairLabel = Literal["positive", "negative"]

BUCKET_KEYWORDS: Final[dict[str, tuple[str, ...]]] = {
    "non_human_red_pale_profile_proxy": ("red glowing demonic eye", "pale purple-skinned villain"),
    "beard_headwear_crop": ("old bearded martial arts master", "middle-aged court official with black hat", "black official hat", "black mustache official face", "upper body close-up portrait"),
    "old_face_crop": ("old bearded martial arts master", "bald old monk", "elder", "elderly", "wrinkled", "upper body close-up portrait"),
}
DIRECT_GREEN_KEYWORDS: Final = ("green monster", "green non-human", "green-skinned demon", "green demon")
SAMPLE_ID_FIELDS: Final = ("ref_id", "reference_id", "image_id", "sample_id", "reference_path", "image_path", "path")


@dataclass(frozen=True, slots=True)
class ManifestInputError(Exception):
    detail: str

    def __str__(self) -> str:
        return self.detail


@dataclass(frozen=True, slots=True)
class C065PairRow:
    pair_id: str
    label: PairLabel
    anchor_id: str
    candidate_id: str
    anchor_group: str
    candidate_group: str
    attribute_bucket: str
    anchor_attributes: tuple[str, ...]
    candidate_attributes: tuple[str, ...]
    matched_keywords: tuple[str, ...]
    negative_reason: str
    source_split: str


@dataclass(frozen=True, slots=True)
class C065Config:
    gate_summary_path: Path
    train_manifest_path: Path
    heldout_manifest_path: Path
    dataset_root: Path
    output_manifest_path: Path
    output_summary_path: Path


@dataclass(frozen=True, slots=True)
class C065Summary:
    heldout_rows_used: int
    total_pairs: int
    positive_pairs: int
    negative_pairs: int
    per_bucket_counts: dict[str, dict[str, int]]
    direct_green_monster_positive_count: int
    path_verification_counts: dict[str, int]


def build_c065_failure_attribute_pairs(config: C065Config) -> C065Summary:
    train_rows = _read_manifest(config.train_manifest_path, allow_empty=False)
    heldout_rows = _read_manifest(config.heldout_manifest_path, allow_empty=True)
    known_ids = set((*train_rows, *heldout_rows))
    attrs_by_id = _read_gate_attributes(
        config.gate_summary_path,
        known_ids=known_ids,
        dataset_root=config.dataset_root,
    )
    usable_train = tuple(
        image_id for image_id in train_rows if _image_path(config.dataset_root, image_id).is_file()
    )
    rows, per_bucket_counts = _build_pair_rows(usable_train, attrs_by_id)
    _write_jsonl(config.output_manifest_path, rows)
    summary = C065Summary(
        heldout_rows_used=0,
        total_pairs=len(rows),
        positive_pairs=_label_count(rows, "positive"),
        negative_pairs=_label_count(rows, "negative"),
        per_bucket_counts=per_bucket_counts,
        direct_green_monster_positive_count=_direct_green_train_count(
            usable_train,
            attrs_by_id,
        ),
        path_verification_counts=_path_verification_counts(
            train_rows=train_rows,
            heldout_rows=heldout_rows,
            pair_rows=rows,
            dataset_root=config.dataset_root,
        ),
    )
    config.output_summary_path.parent.mkdir(parents=True, exist_ok=True)
    config.output_summary_path.write_text(
        json.dumps(asdict(summary), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return summary


def _read_manifest(path: Path, *, allow_empty: bool) -> tuple[str, ...]:
    if not path.is_file():
        raise ManifestInputError(f"manifest not found: {path}")
    rows: list[str] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            raw: JsonValue = json.loads(line)
            if not isinstance(raw, dict):
                raise ManifestInputError(f"{path}:{line_number} row must be an object")
            rows.append(_manifest_image_id(raw, path, line_number))
    if not rows and not allow_empty:
        raise ManifestInputError(f"manifest has no rows: {path}")
    return tuple(rows)


def _manifest_image_id(row: JsonObject, path: Path, line_number: int) -> str:
    for field in ("ref_id", "image_id", "sample_id", "tgt_id"):
        value = row.get(field)
        if isinstance(value, str):
            return _normalize_image_id(value, dataset_root=None)
    raise ManifestInputError(f"{path}:{line_number} missing ref_id")


def _read_gate_attributes(path: Path, *, known_ids: set[str], dataset_root: Path) -> dict[str, tuple[str, ...]]:
    if not path.is_file():
        raise ManifestInputError(f"gate summary not found: {path}")
    raw: JsonValue = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ManifestInputError(f"gate summary must be an object: {path}")
    samples = raw.get("samples")
    if not isinstance(samples, list):
        raise ManifestInputError(f"gate summary samples must be a list: {path}")
    attrs_by_id: dict[str, tuple[str, ...]] = {}
    for sample in samples:
        if not isinstance(sample, dict):
            continue
        image_id = _resolve_sample_image_id(sample, known_ids, dataset_root)
        attrs = sample.get("selected_attributes")
        if image_id is not None and isinstance(attrs, list):
            attrs_by_id[image_id] = tuple(str(attr) for attr in attrs)
    return attrs_by_id


def _resolve_sample_image_id(sample: JsonObject, known_ids: set[str], dataset_root: Path) -> str | None:
    for field in SAMPLE_ID_FIELDS:
        value = sample.get(field)
        if not isinstance(value, str):
            continue
        normalized = _normalize_image_id(value, dataset_root=dataset_root)
        if normalized in known_ids:
            return normalized
        embedded = _embedded_known_id(normalized, known_ids)
        if embedded is not None:
            return embedded
    return None


def _normalize_image_id(raw: str, *, dataset_root: Path | None) -> str:
    value = raw.replace("\\", "/")
    if dataset_root is not None:
        root = dataset_root.as_posix().rstrip("/")
        if value.startswith(f"{root}/"):
            value = value[len(root) + 1 :]
    if value.lower().endswith(".jpg"):
        value = value[:-4]
    return value.lstrip("./")


def _embedded_known_id(value: str, known_ids: set[str]) -> str | None:
    for known_id in sorted(known_ids, key=lambda item: (-len(item), item)):
        if value.endswith(f"/{known_id}") or value == known_id:
            return known_id
    return None


def _build_pair_rows(train_rows: tuple[str, ...], attrs_by_id: dict[str, tuple[str, ...]]) -> tuple[tuple[C065PairRow, ...], dict[str, dict[str, int]]]:
    rows: list[C065PairRow] = []
    per_bucket_counts: dict[str, dict[str, int]] = {}
    for bucket, keywords in BUCKET_KEYWORDS.items():
        bucket_rows = tuple(image_id for image_id in train_rows if _matched(attrs_by_id.get(image_id, ()), keywords))
        bucket_ids = set(bucket_rows)
        outside_rows = tuple(image_id for image_id in train_rows if image_id not in bucket_ids)
        before = len(rows)
        for index, anchor in enumerate(bucket_rows):
            if len(bucket_rows) >= 2:
                rows.append(_pair_row("positive", bucket, anchor, bucket_rows[(index + 1) % len(bucket_rows)], attrs_by_id, len(rows)))
            if outside_rows:
                rows.append(_pair_row("negative", bucket, anchor, outside_rows[index % len(outside_rows)], attrs_by_id, len(rows)))
        bucket_rows_out = rows[before:]
        per_bucket_counts[bucket] = {
            "source_rows": len(bucket_rows),
            "positive_pairs": _label_count(bucket_rows_out, "positive"),
            "negative_pairs": _label_count(bucket_rows_out, "negative"),
        }
    return tuple(rows), per_bucket_counts


def _pair_row(
    label: PairLabel,
    bucket: str,
    anchor: str,
    candidate: str,
    attrs_by_id: dict[str, tuple[str, ...]],
    pair_index: int,
) -> C065PairRow:
    anchor_attrs = attrs_by_id.get(anchor, ())
    candidate_attrs = attrs_by_id.get(candidate, ())
    return C065PairRow(
        pair_id=f"c065_{bucket}_{label[0]}{pair_index:04d}",
        label=label,
        anchor_id=anchor,
        candidate_id=candidate,
        anchor_group=bucket,
        candidate_group=bucket if label == "positive" else "train_not_in_bucket",
        attribute_bucket=bucket,
        anchor_attributes=anchor_attrs,
        candidate_attributes=candidate_attrs,
        matched_keywords=_matched(anchor_attrs, BUCKET_KEYWORDS[bucket]),
        negative_reason="" if label == "positive" else "candidate_not_in_attribute_bucket",
        source_split="train",
    )


def _matched(attributes: tuple[str, ...], keywords: tuple[str, ...]) -> tuple[str, ...]:
    lowered = tuple(attribute.lower() for attribute in attributes)
    return tuple(keyword for keyword in keywords if any(keyword in attr for attr in lowered))


def _direct_green_train_count(train_rows: tuple[str, ...], attrs_by_id: dict[str, tuple[str, ...]]) -> int:
    return sum(1 for image_id in train_rows if _matched(attrs_by_id.get(image_id, ()), DIRECT_GREEN_KEYWORDS))


def _path_verification_counts(*, train_rows: tuple[str, ...], heldout_rows: tuple[str, ...], pair_rows: tuple[C065PairRow, ...], dataset_root: Path) -> dict[str, int]:
    pair_existing = sum(
        1
        for row in pair_rows
        if _image_path(dataset_root, row.anchor_id).is_file()
        and _image_path(dataset_root, row.candidate_id).is_file()
    )
    return {
        "train_rows": len(train_rows),
        "train_existing_images": _existing_count(dataset_root, train_rows),
        "train_missing_images": len(train_rows) - _existing_count(dataset_root, train_rows),
        "heldout_rows": len(heldout_rows),
        "heldout_existing_images": _existing_count(dataset_root, heldout_rows),
        "heldout_missing_images": len(heldout_rows) - _existing_count(dataset_root, heldout_rows),
        "pair_rows_with_existing_paths": pair_existing,
        "pair_rows_with_missing_paths": len(pair_rows) - pair_existing,
    }


def _existing_count(dataset_root: Path, rows: tuple[str, ...]) -> int:
    return sum(1 for image_id in rows if _image_path(dataset_root, image_id).is_file())


def _image_path(dataset_root: Path, image_id: str) -> Path:
    return dataset_root / f"{image_id}.jpg"


def _label_count(rows: tuple[C065PairRow, ...] | list[C065PairRow], label: PairLabel) -> int:
    return sum(1 for row in rows if row.label == label)


def _write_jsonl(path: Path, rows: tuple[C065PairRow, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(asdict(row), ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


app = typer.Typer(add_completion=False)


@app.command()
def main(gate_summary_path: Annotated[Path, typer.Option()], train_manifest_path: Annotated[Path, typer.Option()], heldout_manifest_path: Annotated[Path, typer.Option()], dataset_root: Annotated[Path, typer.Option()], output_manifest_path: Annotated[Path, typer.Option()], output_summary_path: Annotated[Path, typer.Option()]) -> None:
    config = C065Config(gate_summary_path, train_manifest_path, heldout_manifest_path, dataset_root, output_manifest_path, output_summary_path)
    typer.echo(json.dumps(asdict(build_c065_failure_attribute_pairs(config)), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    app()
