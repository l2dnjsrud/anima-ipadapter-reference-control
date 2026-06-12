from __future__ import annotations

import json
import os
import shutil
from dataclasses import asdict
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from tools.comfy_api_client import upload_image, view_image_bytes
from tools.siglip_auto_caption_types import EvalConfig, JsonValue, Sample, Variant


def copy_reference(sample: Sample, config: EvalConfig) -> str:
    image_name = f"{config.out_dir.name}_{sample.label}.jpg"
    source = config.data_root / f"{sample.ref_id}.jpg"
    if _can_prepare_writable_directory(config.comfy_input):
        config.comfy_input.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, config.comfy_input / image_name)
        return image_name
    return upload_image(config.base_url, source, image_name)


def copy_output_image(
    image_info: dict[str, JsonValue],
    name: str,
    config: EvalConfig,
) -> Path:
    filename = _image_info_text(image_info, "filename", "")
    subfolder = _image_info_text(image_info, "subfolder", "")
    image_type = _image_info_text(image_info, "type", "output")
    src = config.comfy_output / subfolder / filename
    dst = config.out_dir / f"{name}.png"
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src.is_file():
        shutil.copy2(src, dst)
    else:
        dst.write_bytes(
            view_image_bytes(
                config.base_url,
                filename=filename,
                subfolder=subfolder,
                image_type=image_type,
            )
        )
    return dst


def _can_prepare_writable_directory(directory: Path) -> bool:
    probe = directory
    while not probe.exists():
        if probe.parent == probe:
            return False
        probe = probe.parent
    return probe.is_dir() and os.access(probe, os.W_OK)


def _image_info_text(
    image_info: dict[str, JsonValue],
    key: str,
    default: str,
) -> str:
    value = image_info.get(key, default)
    if isinstance(value, str):
        return value
    return default


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
