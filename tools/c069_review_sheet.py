from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageOps

from tools.siglip_auto_caption_types import JsonObject


def write_c069_review_sheet(
    rows: tuple[JsonObject, ...],
    output_path: Path,
    buckets: tuple[tuple[str, str], ...],
) -> None:
    cell_w = 220
    cell_h = 260
    grouped = {bucket: [row for row in rows if row["source_bucket"] == bucket] for bucket, _label in buckets}
    row_count = max((len(items) for items in grouped.values()), default=1)
    sheet = Image.new("RGB", (cell_w * len(buckets), cell_h * (row_count + 1)), "white")
    draw = ImageDraw.Draw(sheet)
    for col, (bucket, label) in enumerate(buckets):
        draw.text((col * cell_w + 4, 8), label[:30], fill="black")
        for row_index, row in enumerate(grouped[bucket]):
            _paste_cell(sheet, draw, row, col * cell_w, (row_index + 1) * cell_h, cell_w, cell_h)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path, quality=92)


def _paste_cell(
    sheet: Image.Image,
    draw: ImageDraw.ImageDraw,
    row: JsonObject,
    x: int,
    y: int,
    cell_w: int,
    cell_h: int,
) -> None:
    with Image.open(Path(str(row["image_path"]))) as image:
        thumb = ImageOps.fit(image.convert("RGB"), (cell_w, cell_h - 64))
    sheet.paste(thumb, (x, y))
    draw.text((x + 3, y + cell_h - 62), f"r{row['rank']} score={float(row['bucket_score']):.3f}", fill="black")
    draw.text((x + 3, y + cell_h - 44), f"g={float(row['green_ratio']):.2f} cg={float(row['central_green_ratio']):.2f} red={float(row['red_ratio']):.2f}", fill="black")
    draw.text((x + 3, y + cell_h - 24), Path(str(row["image_id"])).name[:28], fill="black")
