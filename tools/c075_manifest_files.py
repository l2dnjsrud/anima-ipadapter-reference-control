from __future__ import annotations

import shutil
from pathlib import Path

from tools.c075_tag_positive_manifest_types import PairRow
from tools.siglip_auto_caption_types import JsonObject


EXTERNAL_CAPTION = (
    "solo green-skinned non-human monster girl character portrait, tail, anime "
    "style, clean full color webtoon reference"
)


def materialize_source_rows(
    rows: tuple[PairRow, ...],
    source_root: Path,
    scratch_root: Path,
) -> Path | None:
    for row in rows:
        for image_id, suffix in (
            (row.ref_id, ".jpg"),
            (row.tgt_id, ".jpg"),
            (row.tgt_id, ".txt"),
        ):
            src = source_root / f"{image_id}{suffix}"
            if not src.is_file():
                return src
            _link_or_copy(src, scratch_root / f"{image_id}{suffix}")
    return None


def materialize_external_rows(
    rows: tuple[JsonObject, ...],
    scratch_root: Path,
) -> None:
    for row in rows:
        image_id = external_image_id(str(row["candidate_id"]))
        _link_or_copy(Path(str(row["local_image_path"])), scratch_root / f"{image_id}.jpg")
        caption_path = scratch_root / f"{image_id}.txt"
        caption_path.parent.mkdir(parents=True, exist_ok=True)
        caption_path.write_text(EXTERNAL_CAPTION + "\n", encoding="utf-8")


def external_image_id(candidate_id: str) -> str:
    return f"external/c074_direct_green/{candidate_id}"


def _link_or_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    try:
        dst.symlink_to(src.resolve())
    except OSError:
        shutil.copy2(src, dst)
