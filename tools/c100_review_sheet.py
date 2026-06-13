from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps

from tools.siglip_auto_caption_types import JsonObject


def write_c100_review_sheet(rows: tuple[JsonObject, ...], output_path: Path) -> None:
    tile_w, tile_h = 220, 274
    columns = 4
    body_rows = max(1, (len(rows) + columns - 1) // columns)
    sheet = Image.new("RGB", (tile_w * columns, tile_h * body_rows), "white")
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()
    for index, row in enumerate(rows):
        _paste_cell(sheet, draw, font, row, index, tile_w, tile_h, columns)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path, quality=92)


def _paste_cell(
    sheet: Image.Image,
    draw: ImageDraw.ImageDraw,
    font: ImageFont.ImageFont,
    row: JsonObject,
    index: int,
    tile_w: int,
    tile_h: int,
    columns: int,
) -> None:
    col = index % columns
    line = index // columns
    x = col * tile_w
    y = line * tile_h
    with Image.open(Path(str(row["image_path"]))) as image:
        thumb = ImageOps.contain(image.convert("RGB"), (tile_w - 12, tile_h - 76))
    sheet.paste(thumb, (x + 6, y + 24))
    draw.text((x + 6, y + 6), f"{index:03d} {row['source_bucket']}", fill="black", font=font)
    draw.text((x + 6, y + tile_h - 48), f"review={row['review_label']}", fill="black", font=font)
    draw.text((x + 6, y + tile_h - 32), f"g={float(row['green_ratio']):.3f} sg={float(row['strong_green_ratio']):.3f}", fill="black", font=font)
    draw.text((x + 6, y + tile_h - 16), Path(str(row["image_id"])).name[:30], fill="black", font=font)
