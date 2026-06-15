from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Final

import typer
from PIL import Image, ImageStat

from tools.c090_siglip_hard_shape_data import (
    BASELINE_VARIANTS,
    materialize_c090_prompt_manifest,
    siglip_variants,
)
from tools.c090_siglip_hard_shape_report import (
    write_c090_shape_metrics,
    write_extended_contact_sheet,
)
from tools.siglip_auto_caption_artifacts import copy_reference
from tools.siglip_auto_caption_eval import EvalOutputAlreadyExistsError, run_prompt
from tools.siglip_auto_caption_types import (
    DEFAULT_BASE_URL,
    DEFAULT_COMFY_INPUT,
    DEFAULT_COMFY_OUTPUT,
    DEFAULT_DATA_ROOT,
    EvalConfig,
    JsonObject,
    JsonValue,
    Sample,
    Variant,
)


DEFAULT_C090_OUT_DIR: Final = Path("eval/c090_siglip_hard_shape_generation")
PIXEL_AUDIT_NAME: Final = "pixel_nonblank_audit.json"
MIN_PIXEL_STD: Final = 5.0
C090_OUTPUT_PATTERNS: Final = (
    "auto_reference_prompts.jsonl",
    "summary.json",
    "contact_sheet_hard_shape.jpg",
    "shape_metrics.json",
    "metric_rollup.json",
    "report.md",
    PIXEL_AUDIT_NAME,
    "*.png",
    "*.api_prompt.json",
    "*.response.json",
    "*.history.json",
)


@dataclass(frozen=True, slots=True)
class MissingBaselineCandidateError(Exception):
    sample: str
    variant: str

    def __str__(self) -> str:
        return f"missing baseline candidate for {self.sample}: {self.variant}"


def run_c090_eval(probe_manifest_path: Path, config: EvalConfig) -> JsonObject:
    ensure_c090_output_writable(config.out_dir)
    config.out_dir.mkdir(parents=True, exist_ok=True)
    samples, baselines = materialize_c090_prompt_manifest(
        probe_manifest_path,
        out_dir=config.out_dir,
        reference_root=config.data_root,
    )
    ensure_baseline_candidates(samples, baselines)
    variants = siglip_variants()
    results = _run_siglip_variants(samples, variants, config)
    summary = _write_summary(probe_manifest_path, samples, variants, results, baselines, config)
    write_extended_contact_sheet(
        samples,
        variants,
        baselines,
        data_root=config.data_root,
        out_dir=config.out_dir,
        output_path=config.out_dir / "contact_sheet_hard_shape.jpg",
    )
    rollup = write_c090_shape_metrics(config.out_dir / "summary.json", config.out_dir)
    audit = write_pixel_nonblank_audit(results, config.out_dir / PIXEL_AUDIT_NAME)
    write_hard_shape_report(config.out_dir, rollup, audit)
    return summary


def ensure_c090_output_writable(out_dir: Path) -> None:
    if not out_dir.exists():
        return
    conflicts = tuple(
        sorted(
            {
                path
                for pattern in C090_OUTPUT_PATTERNS
                for path in out_dir.glob(pattern)
                if path.is_file()
            }
        )
    )
    if conflicts:
        raise EvalOutputAlreadyExistsError(out_dir=out_dir, conflicts=conflicts)


def ensure_baseline_candidates(
    samples: tuple[Sample, ...],
    baselines: dict[str, dict[str, str]],
) -> None:
    for sample in samples:
        sample_baselines = baselines.get(sample.label, {})
        for variant in BASELINE_VARIANTS:
            if variant not in sample_baselines:
                raise MissingBaselineCandidateError(sample=sample.label, variant=variant)


def write_pixel_nonblank_audit(
    results: dict[str, JsonValue],
    output_path: Path,
    *,
    min_pixel_std: float = MIN_PIXEL_STD,
) -> JsonObject:
    rows: list[JsonObject] = []
    for name, raw in sorted(results.items()):
        if isinstance(raw, dict) and isinstance(raw.get("image"), str):
            rows.append(_pixel_row(name, Path(str(raw["image"])), min_pixel_std))
    blank_count = sum(1 for row in rows if row["nonblank"] is False)
    audit: JsonObject = {
        "min_pixel_std": min_pixel_std,
        "generated_count": len(rows),
        "blank_count": blank_count,
        "low_variance_count": blank_count,
        "blank_definition": (
            "pixel_std <= min_pixel_std or pixel_max == pixel_min; this flags "
            "low-variance/collapsed outputs as well as literal blank canvases"
        ),
        "nonblank": blank_count == 0,
        "rows": rows,
    }
    _write_json(output_path, audit)
    return audit


def write_hard_shape_report(output_dir: Path, rollup: JsonObject, audit: JsonObject) -> None:
    lines = [
        "# c090 SigLIP Hard-Shape Generation Gate",
        "",
        f"- Decision: `{rollup['decision']}`",
        f"- Contact sheet: `{output_dir / 'contact_sheet_hard_shape.jpg'}`",
        f"- Pixel audit: `{output_dir / PIXEL_AUDIT_NAME}`",
        f"- Low-variance / blank-like count: `{audit['blank_count']}`",
        "",
        "| variant | mean uplift | improved rate | cases |",
        "| --- | ---: | ---: | ---: |",
    ]
    summaries = rollup["variant_summaries"]
    if isinstance(summaries, dict):
        for variant, raw in summaries.items():
            if isinstance(raw, dict):
                lines.append(
                    f"| `{variant}` | `{raw['mean_uplift']}` | `{raw['improved_rate']}` | `{raw['cases']}` |"
                )
    (output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _run_siglip_variants(
    samples: tuple[Sample, ...],
    variants: tuple[Variant, ...],
    config: EvalConfig,
) -> dict[str, JsonValue]:
    results: dict[str, JsonValue] = {}
    for sample in samples:
        image_name = copy_reference(sample, config)
        for variant in variants:
            results[f"{sample.label}_{variant.label}"] = run_prompt(sample, variant, image_name, config)
    return results


def _write_summary(
    probe_manifest_path: Path,
    samples: tuple[Sample, ...],
    variants: tuple[Variant, ...],
    results: dict[str, JsonValue],
    baselines: dict[str, dict[str, str]],
    config: EvalConfig,
) -> JsonObject:
    summary: JsonObject = {
        "base_url": config.base_url,
        "probe_manifest": str(probe_manifest_path),
        "prompt_manifest": str(config.out_dir / "auto_reference_prompts.jsonl"),
        "contact_sheet": str(config.out_dir / "contact_sheet_hard_shape.jpg"),
        "variants": [_variant_json(variant) for variant in variants],
        "samples": [_sample_json(sample, config.data_root) for sample in samples],
        "results": results,
        "baseline_candidates": _baseline_json(baselines),
    }
    _write_json(config.out_dir / "summary.json", summary)
    return summary


def _variant_json(variant: Variant) -> JsonObject:
    return {"label": variant.label, "checkpoint": variant.checkpoint, "weight": variant.weight}


def _sample_json(sample: Sample, data_root: Path) -> JsonObject:
    row = sample.prompt_row
    return {
        "label": sample.label,
        "ref_id": sample.ref_id,
        "seed": sample.seed,
        "reference_path": str(data_root / f"{sample.ref_id}.jpg"),
        "prompt_row": {
            "ref_id": row.ref_id,
            "tgt_id": row.tgt_id,
            "source_prompt": row.source_prompt,
            "prompt": row.prompt,
            "selected_attributes": list(row.selected_attributes),
        },
    }


def _baseline_json(baselines: dict[str, dict[str, str]]) -> JsonObject:
    return {sample: dict(candidates) for sample, candidates in sorted(baselines.items())}


def _pixel_row(name: str, image_path: Path, min_pixel_std: float) -> JsonObject:
    with Image.open(image_path) as raw:
        image = raw.convert("RGB")
    stats = ImageStat.Stat(image)
    extrema = image.getextrema()
    pixel_std = sum(float(value) for value in stats.stddev) / len(stats.stddev)
    pixel_mean = sum(float(value) for value in stats.mean) / len(stats.mean)
    pixel_min = min(channel[0] for channel in extrema)
    pixel_max = max(channel[1] for channel in extrema)
    return {
        "name": name,
        "image": str(image_path),
        "width": image.width,
        "height": image.height,
        "pixel_mean": pixel_mean,
        "pixel_std": pixel_std,
        "pixel_min": pixel_min,
        "pixel_max": pixel_max,
        "nonblank": pixel_std > min_pixel_std and pixel_max > pixel_min,
    }


def _write_json(path: Path, payload: JsonObject) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


app = typer.Typer(add_completion=False)


@app.command()
def main(
    probe_manifest_path: Annotated[Path, typer.Argument()],
    out_dir: Annotated[Path, typer.Option()] = DEFAULT_C090_OUT_DIR,
    base_url: Annotated[str, typer.Option()] = DEFAULT_BASE_URL,
    data_root: Annotated[Path, typer.Option()] = DEFAULT_DATA_ROOT,
    comfy_input: Annotated[Path, typer.Option()] = DEFAULT_COMFY_INPUT,
    comfy_output: Annotated[Path, typer.Option()] = DEFAULT_COMFY_OUTPUT,
) -> None:
    try:
        run_c090_eval(
            probe_manifest_path,
            EvalConfig(
                data_root=data_root,
                base_url=base_url,
                out_dir=out_dir,
                comfy_input=comfy_input,
                comfy_output=comfy_output,
            ),
        )
    except (EvalOutputAlreadyExistsError, MissingBaselineCandidateError) as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(1) from exc
    typer.echo(f"wrote {out_dir / 'contact_sheet_hard_shape.jpg'}")


if __name__ == "__main__":
    app()
