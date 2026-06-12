from __future__ import annotations

import csv
import json
import urllib.error
import urllib.request
from pathlib import Path
from typing import Callable

from PIL import Image, ImageDraw, ImageOps, UnidentifiedImageError

from tools.c071_seed_package import LABEL_SCHEMA
from tools.siglip_auto_caption_types import JsonObject, JsonValue

type FetchImage = Callable[[str, Path, float, int], bool]


def fetch_image_to_path(url: str, destination: Path, timeout_seconds: float, max_image_bytes: int) -> bool:
    try:
        with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"}), timeout=timeout_seconds) as response:
            content = response.read(max_image_bytes + 1)
    except (urllib.error.URLError, TimeoutError, OSError):
        return False
    if len(content) > max_image_bytes:
        return False
    destination.write_bytes(content)
    try:
        with Image.open(destination) as image:
            image.convert("RGB").save(destination, quality=94)
    except (UnidentifiedImageError, OSError):
        destination.unlink(missing_ok=True)
        return False
    return True


def write_sheet(rows: tuple[JsonObject, ...], output_path: Path) -> bool:
    visible = tuple(row for row in rows if Path(str(row["local_image_path"])).is_file())
    if not visible:
        return False
    cell_w, cell_h, cols = 280, 330, 4
    rows_count = (len(visible) + cols - 1) // cols
    sheet = Image.new("RGB", (cell_w * cols, cell_h * rows_count), "white")
    draw = ImageDraw.Draw(sheet)
    for index, row in enumerate(visible):
        x, y = (index % cols) * cell_w, (index // cols) * cell_h
        with Image.open(Path(str(row["local_image_path"]))) as image:
            thumb = ImageOps.contain(image.convert("RGB"), (cell_w, cell_h - 78))
        sheet.paste(thumb, (x + (cell_w - thumb.width) // 2, y))
        draw.text((x + 4, y + cell_h - 74), str(row["candidate_id"])[:38], fill="black")
        draw.text((x + 4, y + cell_h - 54), str(row.get("manual_label", row.get("suggested_label", "")))[:38], fill="black")
        draw.text((x + 4, y + cell_h - 34), str(row.get("review_source", ""))[:38], fill="black")
        draw.text((x + 4, y + cell_h - 16), str(row.get("external_license_note", ""))[:38], fill="black")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path, quality=92)
    return True


def write_template(path: Path, rows: tuple[JsonObject, ...]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("candidate_id", "image_id", "download_status", "manual_label", "manual_note", "allowed_labels"), lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "candidate_id": row["candidate_id"],
                    "image_id": row["image_id"],
                    "download_status": row["download_status"],
                    "manual_label": row.get("manual_label", ""),
                    "manual_note": "",
                    "allowed_labels": "|".join(LABEL_SCHEMA),
                }
            )


def read_label_map(path: Path) -> dict[str, JsonObject]:
    with path.open(encoding="utf-8", newline="") as handle:
        return {str(row["candidate_id"]): dict(row) for row in csv.DictReader(handle)}


def read_ids(path: Path) -> set[str]:
    return {str(row["ref_id"]) for row in read_jsonl(path) if isinstance(row.get("ref_id"), str)} if path.is_file() else set()


def read_jsonl(path: Path) -> tuple[JsonObject, ...]:
    rows: list[JsonObject] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        raw: JsonValue = json.loads(line)
        if isinstance(raw, dict):
            rows.append(raw)
    return tuple(rows)


def write_jsonl(path: Path, rows: tuple[JsonObject, ...]) -> None:
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def dimensions(path: Path) -> tuple[int, int]:
    with Image.open(path) as image:
        return image.size
