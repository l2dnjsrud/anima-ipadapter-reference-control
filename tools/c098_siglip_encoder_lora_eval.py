from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Final

import typer

from tools.c090_siglip_hard_shape_data import BASELINE_VARIANTS, materialize_c090_prompt_manifest
from tools.c090_siglip_hard_shape_eval import (
    PIXEL_AUDIT_NAME,
    MissingBaselineCandidateError,
    ensure_baseline_candidates,
    ensure_c090_output_writable,
    write_pixel_nonblank_audit,
)
from tools.c090_siglip_hard_shape_report import write_c090_shape_metrics, write_extended_contact_sheet
from tools.c092_siglip_qwen_target_eval import _read_json, _run_siglip_variants, _write_json, _write_summary
from tools.c094_shape_eval import C094_CHECKPOINT
from tools.c095_feature_bridge_eval import C095_CHECKPOINT
from tools.c096_siglip_encoder_lora_eval import (
    C096_ENCODER_LORA,
    _best_named,
    _best_prefixed,
    _blank_like_rows,
    _uplift,
    _variant_summaries,
)
from tools.siglip_auto_caption_eval import EvalOutputAlreadyExistsError
from tools.siglip_auto_caption_types import DEFAULT_BASE_URL, DEFAULT_COMFY_INPUT, DEFAULT_COMFY_OUTPUT, DEFAULT_DATA_ROOT
from tools.siglip_auto_caption_types import EvalConfig, JsonObject, Variant


C098_ENCODER_LORA: Final = "anima_siglip_encoder_lora_c098_rank8_layer4_0224_20260613.safetensors"
DEFAULT_C098_OUT_DIR: Final = Path("eval/c098_siglip_encoder_lora_generation_gate_20260613")


def c098_siglip_variants() -> tuple[Variant, ...]:
    return (
        Variant("no_ip", None, 0.0),
        Variant("c094_shape_supervised_w14", C094_CHECKPOINT, 1.4),
        Variant("c095_feature_bridge_w14", C095_CHECKPOINT, 1.4),
        Variant("c096_lora_c094_w14", C094_CHECKPOINT, 1.4, C096_ENCODER_LORA),
        Variant("c098_lora_c094_w08", C094_CHECKPOINT, 0.8, C098_ENCODER_LORA),
        Variant("c098_lora_c094_w10", C094_CHECKPOINT, 1.0, C098_ENCODER_LORA),
        Variant("c098_lora_c094_w12", C094_CHECKPOINT, 1.2, C098_ENCODER_LORA),
        Variant("c098_lora_c094_w14", C094_CHECKPOINT, 1.4, C098_ENCODER_LORA),
    )


def run_c098_eval(probe_manifest_path: Path, config: EvalConfig) -> JsonObject:
    ensure_c090_output_writable(config.out_dir)
    config.out_dir.mkdir(parents=True, exist_ok=True)
    samples, baselines = materialize_c090_prompt_manifest(
        probe_manifest_path,
        out_dir=config.out_dir,
        reference_root=config.data_root,
    )
    ensure_baseline_candidates(samples, baselines)
    variants = c098_siglip_variants()
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
    audit = write_pixel_nonblank_audit(results, config.out_dir / PIXEL_AUDIT_NAME)
    rollup = write_c098_shape_metrics(config.out_dir / "summary.json", config.out_dir, audit)
    write_c098_report(config.out_dir, rollup, audit)
    return summary


def write_c098_shape_metrics(summary_path: Path, output_dir: Path, pixel_audit: JsonObject) -> JsonObject:
    write_c090_shape_metrics(summary_path, output_dir)
    metrics = _read_json(output_dir / "shape_metrics.json")
    rows = metrics.get("rows")
    if not isinstance(rows, list):
        raise TypeError("shape_metrics rows must be a list")
    rollup = summarize_c098_shape_rows(tuple(row for row in rows if isinstance(row, dict)), pixel_audit)
    metrics["rollup"] = rollup
    _write_json(output_dir / "shape_metrics.json", metrics)
    _write_json(output_dir / "metric_rollup.json", rollup)
    return rollup


def summarize_c098_shape_rows(rows: tuple[JsonObject, ...], pixel_audit: JsonObject) -> JsonObject:
    summaries = _variant_summaries(rows)
    c098 = _best_prefixed(summaries, "c098_")
    c096 = _best_named(summaries, ("c096_lora_c094_w14",))
    c094 = _best_named(summaries, ("c094_shape_supervised_w14",))
    c095 = _best_named(summaries, ("c095_feature_bridge_w14",))
    qwen = _best_named(summaries, BASELINE_VARIANTS)
    heldout = {
        "best_c098_uplift": _uplift(rows, "heldout07", c098[0]),
        "c096_w14_uplift": _uplift(rows, "heldout07", "c096_lora_c094_w14"),
        "c094_w14_uplift": _uplift(rows, "heldout07", "c094_shape_supervised_w14"),
        "c095_w14_uplift": _uplift(rows, "heldout07", "c095_feature_bridge_w14"),
    }
    blank_rows = _blank_like_rows(pixel_audit, "c098_")
    return {
        "variant_summaries": summaries,
        "best_c098_variant": c098[0],
        "best_c096_variant": c096[0],
        "best_c094_variant": c094[0],
        "best_c095_variant": c095[0],
        "best_qwen_baseline_variant": qwen[0],
        "heldout07": heldout,
        "c098_blank_like_rows": blank_rows,
        "decision": _decision(c098, c096, c094, c095, qwen, heldout, blank_rows),
    }


def write_c098_report(output_dir: Path, rollup: JsonObject, audit: JsonObject) -> None:
    lines = [
        "# c098 SigLIP Encoder LoRA Hard-Shape Gate",
        "",
        f"- Decision: `{rollup['decision']}`",
        f"- Best c098: `{rollup['best_c098_variant']}`",
        f"- C096 baseline: `{rollup['best_c096_variant']}`",
        f"- C094 baseline: `{rollup['best_c094_variant']}`",
        f"- C095 baseline: `{rollup['best_c095_variant']}`",
        f"- Best Qwen baseline: `{rollup['best_qwen_baseline_variant']}`",
        f"- Heldout07: `{json.dumps(rollup['heldout07'], ensure_ascii=False)}`",
        f"- C098 blank-like rows: `{json.dumps(rollup['c098_blank_like_rows'], ensure_ascii=False)}`",
        f"- Low-variance / blank-like count: `{audit['blank_count']}`",
        "",
        "| variant | mean uplift | improved rate | cases |",
        "| --- | ---: | ---: | ---: |",
    ]
    summaries = rollup["variant_summaries"]
    if isinstance(summaries, dict):
        for variant, raw in summaries.items():
            if isinstance(raw, dict):
                lines.append(f"| `{variant}` | `{raw['mean_uplift']}` | `{raw['improved_rate']}` | `{raw['cases']}` |")
    (output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _decision(
    c098: tuple[str, float],
    c096: tuple[str, float],
    c094: tuple[str, float],
    c095: tuple[str, float],
    qwen: tuple[str, float],
    heldout: JsonObject,
    blank_rows: list[str],
) -> str:
    heldout_c098 = float(heldout["best_c098_uplift"])
    heldout_baseline = max(
        float(heldout["c096_w14_uplift"]),
        float(heldout["c094_w14_uplift"]),
        float(heldout["c095_w14_uplift"]),
    )
    adapter_baseline = max(c096[1], c094[1], c095[1])
    if blank_rows:
        return "c098_encoder_lora_not_promoted_blank_outputs"
    if c098[1] >= max(adapter_baseline, qwen[1]) + 0.010 and heldout_c098 >= heldout_baseline + 0.020:
        return "c098_encoder_lora_quality_candidate_for_larger_gate"
    if c098[1] >= c096[1] + 0.005 and heldout_c098 >= float(heldout["c096_w14_uplift"]) - 0.005:
        return "c098_encoder_lora_improves_c096_needs_larger_gate"
    if c098[1] >= qwen[1] - 0.010 and heldout_c098 >= 0.030:
        return "c098_encoder_lora_near_qwen_candidate"
    return "c098_encoder_lora_not_promoted_requires_stronger_encoder_or_better_data"


app = typer.Typer(add_completion=False)


@app.command()
def main(
    probe_manifest_path: Annotated[Path, typer.Argument()],
    out_dir: Annotated[Path, typer.Option()] = DEFAULT_C098_OUT_DIR,
    base_url: Annotated[str, typer.Option()] = DEFAULT_BASE_URL,
    data_root: Annotated[Path, typer.Option()] = DEFAULT_DATA_ROOT,
    comfy_input: Annotated[Path, typer.Option()] = DEFAULT_COMFY_INPUT,
    comfy_output: Annotated[Path, typer.Option()] = DEFAULT_COMFY_OUTPUT,
) -> None:
    try:
        run_c098_eval(probe_manifest_path, EvalConfig(data_root, base_url, out_dir, comfy_input, comfy_output))
    except (EvalOutputAlreadyExistsError, MissingBaselineCandidateError) as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(1) from exc
    typer.echo(f"wrote {out_dir / 'contact_sheet_hard_shape.jpg'}")


if __name__ == "__main__":
    app()
