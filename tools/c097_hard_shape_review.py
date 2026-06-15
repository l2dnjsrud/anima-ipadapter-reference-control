from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps

from tools.siglip_auto_caption_types import JsonObject


def write_c097_review_sheet(rows: list[JsonObject], output_root: Path, review_sheet: Path) -> None:
    tile_w, tile_h = 180, 220
    sheet = Image.new("RGB", (tile_w * 3, tile_h * len(rows)), "white")
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()
    for y, row in enumerate(rows):
        _paste_review_row(sheet, draw, font, row, output_root, y, tile_w, tile_h)
    review_sheet.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(review_sheet, quality=92)


def _paste_review_row(
    sheet: Image.Image,
    draw: ImageDraw.ImageDraw,
    font: ImageFont.ImageFont,
    row: JsonObject,
    output_root: Path,
    y: int,
    tile_w: int,
    tile_h: int,
) -> None:
    for x, field in enumerate(("ref_id", "tgt_id", "neg_id")):
        path = output_root / f"{row[field]}.jpg"
        with Image.open(path) as image:
            thumb = ImageOps.contain(image.convert("RGB"), (tile_w - 12, tile_h - 42))
        left, top = x * tile_w + 6, y * tile_h + 24
        sheet.paste(thumb, (left, top))
        draw.text((x * tile_w + 6, y * tile_h + 6), f"{y:03d} {field}", fill="black", font=font)
