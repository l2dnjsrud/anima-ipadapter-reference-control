from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from tools.c075_tag_positive_manifest_types import PairRow
from tools.c079_manifest_types import C079ManifestError
from tools.siglip_auto_caption_types import JsonObject, JsonValue


GUARD_LABELS: Final = frozenset({"useful_proxy_non_human", "guard_false_positive_human"})


@dataclass(frozen=True, slots=True)
class ExternalTrainingSource:
    row: JsonObject
    prefix: str
    prompt: str
    source_label: str


def target_positive_rows(path: Path, source_name: str) -> tuple[JsonObject, ...]:
    return tuple(
        require_image(row, path, f"{source_name} target-positive")
        for _line, row in read_jsonl(path)
        if row.get("manual_label") == "target_positive"
    )


def guard_proxy_rows(path: Path) -> tuple[JsonObject, ...]:
    return tuple(
        require_image(row, path, "c077 guard/proxy")
        for _line, row in read_jsonl(path)
        if row.get("manual_label") in GUARD_LABELS
    )


def read_pair_rows(path: Path) -> tuple[PairRow, ...]:
    rows = tuple(
        PairRow(
            ref_id=string_field(raw, "ref_id", path, line_number),
            tgt_id=string_field(raw, "tgt_id", path, line_number),
            prompt=string_field(raw, "prompt", path, line_number),
        )
        for line_number, raw in read_jsonl(path)
    )
    if not rows:
        raise C079ManifestError(f"source manifest has no rows: {path}")
    return rows


def materialize_external_rows(
    sources: tuple[ExternalTrainingSource, ...],
    repeat: int,
    scratch_root: Path,
) -> tuple[PairRow, ...]:
    for source in sources:
        materialize_external(source, scratch_root)
    return tuple(
        PairRow(image_id, image_id, source.prompt)
        for _ in range(repeat)
        for source in sources
        for image_id in (external_image_id(source.prefix, candidate_id(source.row)),)
    )


def write_jsonl(path: Path, rows: tuple[JsonObject, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def label_counts(path: Path) -> dict[str, int]:
    counts: dict[str, int] = {}
    for _line, row in read_jsonl(path):
        label = row.get("manual_label")
        if isinstance(label, str):
            counts[label] = counts.get(label, 0) + 1
    return counts


def license_notes(rows: tuple[JsonObject, ...]) -> tuple[str, ...]:
    notes = {
        str(row.get("external_license_note", "local synthetic/internal artifact"))
        for row in rows
    }
    return tuple(sorted(notes))


def require_image(row: JsonObject, path: Path, label: str) -> JsonObject:
    for field in ("candidate_id", "local_image_path"):
        if not isinstance(row.get(field), str):
            raise C079ManifestError(f"{path}: {label} missing {field}")
    image_path = Path(str(row["local_image_path"]))
    if not image_path.is_file():
        raise C079ManifestError(f"missing {label} image: {image_path}")
    return row


def materialize_external(source: ExternalTrainingSource, scratch_root: Path) -> None:
    image_id = external_image_id(source.prefix, candidate_id(source.row))
    source_path = Path(str(source.row["local_image_path"]))
    if not source_path.is_file():
        raise C079ManifestError(f"missing {source.source_label} image: {source_path}")
    link_or_copy(source_path, scratch_root / f"{image_id}.jpg")
    (scratch_root / f"{image_id}.txt").write_text(
        source.prompt + "\n",
        encoding="utf-8",
    )


def candidate_id(row: JsonObject) -> str:
    value = row.get("candidate_id")
    if not isinstance(value, str):
        raise C079ManifestError("external row missing candidate_id")
    return value


def external_image_id(prefix: str, row_candidate_id: str) -> str:
    return f"{prefix}/{row_candidate_id}"


def read_jsonl(path: Path) -> tuple[tuple[int, JsonObject], ...]:
    if not path.is_file():
        raise C079ManifestError(f"jsonl not found: {path}")
    parsed: list[tuple[int, JsonObject]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            raw: JsonValue = json.loads(line)
            if not isinstance(raw, dict):
                raise C079ManifestError(f"{path}:{line_number} row must be object")
            parsed.append((line_number, raw))
    return tuple(parsed)


def string_field(row: JsonObject, field: str, path: Path, line_number: int) -> str:
    value = row.get(field)
    if not isinstance(value, str):
        raise C079ManifestError(f"{path}:{line_number} missing {field}")
    return value


def link_or_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    try:
        dst.symlink_to(src.resolve())
    except OSError:
        shutil.copy2(src, dst)
