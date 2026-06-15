from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps

from tools.siglip_auto_caption_types import JsonObject


def write_c102_review_sheet(rows: tuple[JsonObject, ...], output_path: Path) -> None:
    tile_w, tile_h = 260, 318
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
        thumb = ImageOps.contain(image.convert("RGB"), (tile_w - 12, tile_h - 116))
    sheet.paste(thumb, (x + 6, y + 30))
    draw.rectangle((x, y, x + tile_w - 1, y + tile_h - 1), outline=_color(str(row["final_label"])))
    draw.text((x + 6, y + 6), f"{index:03d} {row['source_bucket']}", fill="black", font=font)
    draw.text((x + 6, y + tile_h - 78), f"qa={row['qa_label']}", fill="black", font=font)
    draw.text((x + 6, y + tile_h - 62), f"final={row['final_label']}", fill="black", font=font)
    draw.text((x + 6, y + tile_h - 46), f"prior={row.get('manual_label', '')}", fill="black", font=font)
    draw.text((x + 6, y + tile_h - 30), str(row.get("qa_evidence", ""))[:40], fill="black", font=font)
    draw.text((x + 6, y + tile_h - 14), Path(str(row["image_id"])).name[:34], fill="black", font=font)


def _color(label: str) -> str:
    if label == "local_positive":
        return "green"
    if label == "local_negative":
        return "red"
    return "gray"
