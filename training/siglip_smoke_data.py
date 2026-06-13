from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps

from training.siglip_smoke_types import PairPaths, PairRow, SmokeInputError


def load_pair_rows(path: Path, *, limit: int) -> list[PairRow]:
    if limit < 1:
        raise SmokeInputError("limit must be >= 1")
    if not path.is_file():
        raise SmokeInputError(f"manifest not found: {path}")
    rows: list[PairRow] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if len(rows) >= limit:
                break
            raw = json.loads(line)
            try:
                rows.append(
                    PairRow(
                        ref_id=str(raw["ref_id"]),
                        tgt_id=str(raw["tgt_id"]),
                        prompt=str(raw["prompt"]),
                        neg_id=(
                            str(raw["neg_id"])
                            if raw.get("neg_id") not in (None, "")
                            else None
                        ),
                    )
                )
            except KeyError as exc:
                raise SmokeInputError(
                    f"manifest row {line_number} missing {exc.args[0]}"
                ) from exc
    if not rows:
        raise SmokeInputError(f"manifest has no usable rows: {path}")
    return rows


def resolve_pair_paths(row: PairRow, image_root: Path) -> PairPaths:
    paths = PairPaths(
        ref_image=image_root / f"{row.ref_id}.jpg",
        target_image=image_root / f"{row.tgt_id}.jpg",
        target_caption=image_root / f"{row.tgt_id}.txt",
    )
    for path in (paths.ref_image, paths.target_image, paths.target_caption):
        if not path.is_file():
            raise SmokeInputError(f"missing pair file: {path}")
    return paths


def load_anima_pixels(path: Path, *, resolution: int) -> torch.Tensor:
    image = _load_square_rgb(path, resolution)
    array = np.asarray(image, dtype=np.float32) / 127.5 - 1.0
    return torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0).contiguous()


def load_siglip_image(path: Path, *, resolution: int = 512) -> Image.Image:
    return _load_square_rgb(path, resolution)


def _load_square_rgb(path: Path, resolution: int) -> Image.Image:
    if resolution < 64 or resolution % 8 != 0:
        raise SmokeInputError("resolution must be a multiple of 8 and at least 64")
    with Image.open(path) as image:
        rgb = image.convert("RGB")
        return ImageOps.fit(
            rgb,
            (resolution, resolution),
            method=Image.Resampling.LANCZOS,
            centering=(0.5, 0.5),
        )
