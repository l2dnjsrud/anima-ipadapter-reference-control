from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Final

import typer

from tools.c090_siglip_hard_shape_data import (
    BASELINE_VARIANTS,
    SIGLIP_PILOT_CHECKPOINT,
    materialize_c090_prompt_manifest,
)
from tools.c090_siglip_hard_shape_eval import (
    PIXEL_AUDIT_NAME,
    MissingBaselineCandidateError,
    ensure_baseline_candidates,
    ensure_c090_output_writable,
    write_pixel_nonblank_audit,
)
from tools.c090_siglip_hard_shape_report import (
    write_c090_shape_metrics,
    write_extended_contact_sheet,
)
from tools.c091_siglip_hard_shape_eval import C089_CHECKPOINT, C091_CHECKPOINT
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


C092_CHECKPOINT: Final = "anima_siglip_ip_adapter_c092_qwen_target_0064_20260613.safetensors"
DEFAULT_C092_OUT_DIR: Final = Path("eval/c092_qwen_target_siglip_generation_gate_20260613")


def c092_siglip_variants() -> tuple[Variant, ...]:
    return (
        Variant("no_ip", None, 0.0),
        Variant("siglip_pilot_w14", SIGLIP_PILOT_CHECKPOINT, 1.4),
        Variant("c089_shape_w14", C089_CHECKPOINT, 1.4),
        Variant("c091_feature_calibrator_w14", C091_CHECKPOINT, 1.4),
        Variant("c092_qwen_target_w10", C092_CHECKPOINT, 1.0),
        Variant("c092_qwen_target_w14", C092_CHECKPOINT, 1.4),
    )


def run_c092_eval(probe_manifest_path: Path, config: EvalConfig) -> JsonObject:
    ensure_c090_output_writable(config.out_dir)
    config.out_dir.mkdir(parents=True, exist_ok=True)
    samples, baselines = materialize_c090_prompt_manifest(
        probe_manifest_path,
        out_dir=config.out_dir,
        reference_root=config.data_root,
    )
    ensure_baseline_candidates(samples, baselines)
    variants = c092_siglip_variants()
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
    rollup = write_c092_shape_metrics(config.out_dir / "summary.json", config.out_dir)
    audit = write_pixel_nonblank_audit(results, config.out_dir / PIXEL_AUDIT_NAME)
    write_c092_report(config.out_dir, rollup, audit)
    return summary


def write_c092_shape_metrics(summary_path: Path, output_dir: Path) -> JsonObject:
    write_c090_shape_metrics(summary_path, output_dir)
    metrics = _read_json(output_dir / "shape_metrics.json")
    rows = metrics.get("rows")
    if not isinstance(rows, list):
        raise TypeError("shape_metrics rows must be a list")
    rollup = summarize_c092_shape_rows(tuple(row for row in rows if isinstance(row, dict)))
    metrics["rollup"] = rollup
    _write_json(output_dir / "shape_metrics.json", metrics)
    _write_json(output_dir / "metric_rollup.json", rollup)
    return rollup


def summarize_c092_shape_rows(rows: tuple[JsonObject, ...]) -> JsonObject:
    summaries = _variant_summaries(rows)
    c092 = _best_prefixed(summaries, "c092_")
    c091 = _best_named(summaries, ("c091_feature_calibrator_w14",))
    c089 = _best_named(summaries, ("c089_shape_w14",))
    qwen = _best_named(summaries, BASELINE_VARIANTS)
    return {
        "variant_summaries": summaries,
        "best_c092_variant": c092[0],
        "best_c091_variant": c091[0],
        "best_c089_variant": c089[0],
        "best_qwen_baseline_variant": qwen[0],
        "decision": _c092_decision(c092, c089, qwen),
    }


def write_c092_report(output_dir: Path, rollup: JsonObject, audit: JsonObject) -> None:
    lines = [
        "# c092 Qwen-Target SigLIP Hard-Shape Gate",
        "",
        f"- Decision: `{rollup['decision']}`",
        f"- Best c092: `{rollup['best_c092_variant']}`",
        f"- Best c091 baseline: `{rollup['best_c091_variant']}`",
        f"- Best c089 baseline: `{rollup['best_c089_variant']}`",
        f"- Best Qwen baseline: `{rollup['best_qwen_baseline_variant']}`",
        f"- Contact sheet: `{output_dir / 'contact_sheet_hard_shape.jpg'}`",
        f"- Pixel audit: `{output_dir / PIXEL_AUDIT_NAME}`",
        f"- Visual audit: `{output_dir / 'visual_audit.md'}`",
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
        "variants": [{"label": item.label, "checkpoint": item.checkpoint, "weight": item.weight} for item in variants],
        "samples": [_sample_json(sample, config.data_root) for sample in samples],
        "results": results,
        "baseline_candidates": {sample: dict(paths) for sample, paths in sorted(baselines.items())},
    }
    _write_json(config.out_dir / "summary.json", summary)
    return summary


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


def _variant_summaries(rows: tuple[JsonObject, ...]) -> JsonObject:
    by_variant: dict[str, list[float]] = {}
    for row in rows:
        by_variant.setdefault(str(row["variant"]), []).append(float(row["uplift"]))
    return {
        variant: {
            "cases": len(values),
            "mean_uplift": sum(values) / len(values),
            "improved_rate": sum(1 for value in values if value > 0.0) / len(values),
        }
        for variant, values in sorted(by_variant.items())
        if values
    }


def _best_prefixed(summaries: JsonObject, prefix: str) -> tuple[str, float]:
    return max(
        (
            (variant, float(raw["mean_uplift"]))
            for variant, raw in summaries.items()
            if isinstance(raw, dict) and variant.startswith(prefix)
        ),
        key=lambda item: item[1],
        default=("", float("-inf")),
    )


def _best_named(summaries: JsonObject, names: tuple[str, ...]) -> tuple[str, float]:
    return max(
        (
            (name, float(summaries[name]["mean_uplift"]))
            for name in names
            if isinstance(summaries.get(name), dict)
        ),
        key=lambda item: item[1],
        default=("", float("-inf")),
    )


def _c092_decision(c092: tuple[str, float], c089: tuple[str, float], qwen: tuple[str, float]) -> str:
    if c092[1] >= qwen[1] - 0.01:
        return "c092_qwen_target_candidate_for_larger_gate"
    if c092[1] >= c089[1] + 0.01:
        return "c092_improves_c089_but_not_qwen_baseline"
    if c092[1] >= c089[1] - 0.005:
        return "c092_matches_c089_not_qwen_baseline"
    return "c092_not_improved_requires_encoder_checkpoint_training"


def _read_json(path: Path) -> JsonObject:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise TypeError(f"json root must be object: {path}")
    return raw


def _write_json(path: Path, payload: JsonObject) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


app = typer.Typer(add_completion=False)


@app.command()
def main(
    probe_manifest_path: Annotated[Path, typer.Argument()],
    out_dir: Annotated[Path, typer.Option()] = DEFAULT_C092_OUT_DIR,
    base_url: Annotated[str, typer.Option()] = DEFAULT_BASE_URL,
    data_root: Annotated[Path, typer.Option()] = DEFAULT_DATA_ROOT,
    comfy_input: Annotated[Path, typer.Option()] = DEFAULT_COMFY_INPUT,
    comfy_output: Annotated[Path, typer.Option()] = DEFAULT_COMFY_OUTPUT,
) -> None:
    try:
        run_c092_eval(
            probe_manifest_path,
            EvalConfig(data_root, base_url, out_dir, comfy_input, comfy_output),
        )
    except (EvalOutputAlreadyExistsError, MissingBaselineCandidateError) as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(1) from exc
    typer.echo(f"wrote {out_dir / 'contact_sheet_hard_shape.jpg'}")


if __name__ == "__main__":
    app()
