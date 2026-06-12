from __future__ import annotations

import json
import shutil
from dataclasses import asdict
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from tools.siglip_auto_caption_types import EvalConfig, JsonValue, Sample, Variant


def copy_reference(sample: Sample, config: EvalConfig) -> str:
    image_name = f"{config.out_dir.name}_{sample.label}.jpg"
    config.comfy_input.mkdir(parents=True, exist_ok=True)
    shutil.copy2(config.data_root / f"{sample.ref_id}.jpg", config.comfy_input / image_name)
    return image_name


def copy_output_image(
    image_info: dict[str, JsonValue],
    name: str,
    config: EvalConfig,
) -> Path:
    src = config.comfy_output / str(image_info["subfolder"]) / str(image_info["filename"])
    dst = config.out_dir / f"{name}.png"
    shutil.copy2(src, dst)
    return dst


def write_contact_sheet(
    samples: tuple[Sample, ...],
    variants: tuple[Variant, ...],
    config: EvalConfig,
) -> None:
    columns = ("reference", *tuple(variant.label for variant in variants))
    cell = (240, 310)
    label_h = 32
    margin = 14
    sheet = Image.new(
        "RGB",
        (margin * 2 + len(columns) * cell[0], margin * 2 + (len(samples) + 1) * (cell[1] + label_h)),
        "white",
    )
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()
    for col, title in enumerate(columns):
        draw.text((margin + col * cell[0] + 8, margin + 8), title, fill="black", font=font)
    for row_index, sample in enumerate(samples, start=1):
        y = margin + row_index * (cell[1] + label_h)
        draw.text((margin + 8, y - label_h + 8), sample.label, fill="black", font=font)
        paths = [
            config.data_root / f"{sample.ref_id}.jpg",
            *[config.out_dir / f"{sample.label}_{variant.label}.png" for variant in variants],
        ]
        for col, path in enumerate(paths):
            sheet.paste(fit_image(path, cell), (margin + col * cell[0], y))
    sheet.save(config.out_dir / "contact_sheet.jpg", quality=92)


def fit_image(path: Path, size: tuple[int, int]) -> Image.Image:
    image = Image.open(path).convert("RGB")
    image.thumbnail(size, Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", size, "white")
    canvas.paste(image, ((size[0] - image.width) // 2, (size[1] - image.height) // 2))
    return canvas


def write_summary(
    samples: tuple[Sample, ...],
    variants: tuple[Variant, ...],
    results: dict[str, JsonValue],
    config: EvalConfig,
) -> None:
    summary = {
        "base_url": config.base_url,
        "variants": [asdict(variant) for variant in variants],
        "samples": [asdict(sample) for sample in samples],
        "results": results,
        "contact_sheet": str(config.out_dir / "contact_sheet.jpg"),
    }
    (config.out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


def write_report(
    samples: tuple[Sample, ...],
    variants: tuple[Variant, ...],
    config: EvalConfig,
) -> None:
    lines = [
        f"# SigLIP Auto-Caption Runtime Evaluation: {config.out_dir.name}",
        "",
        f"- Contact sheet: `{config.out_dir / 'contact_sheet.jpg'}`",
        "- Columns: reference / " + " / ".join(variant.label for variant in variants),
        "",
        "Decision: `pending_visual_review`",
        "",
        "| sample | selected attributes |",
        "| --- | --- |",
    ]
    for sample in samples:
        attrs = ", ".join(sample.prompt_row.selected_attributes)
        lines.append(f"| {sample.label} | {attrs} |")
    (config.out_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
