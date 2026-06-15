from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Final

import typer

from tools.c090_siglip_hard_shape_data import (
    BASELINE_VARIANTS,
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
from tools.c092_siglip_qwen_target_eval import (
    _read_json,
    _run_siglip_variants,
    _write_json,
    _write_summary,
)
from tools.c094_shape_eval import C094_CHECKPOINT
from tools.c095_feature_bridge_eval import C095_CHECKPOINT
from tools.siglip_auto_caption_eval import EvalOutputAlreadyExistsError
from tools.siglip_auto_caption_types import (
    DEFAULT_BASE_URL,
    DEFAULT_COMFY_INPUT,
    DEFAULT_COMFY_OUTPUT,
    DEFAULT_DATA_ROOT,
    EvalConfig,
    JsonObject,
    Variant,
)


C096_ENCODER_LORA: Final = "anima_siglip_encoder_lora_c096_rank8_0096_20260613.safetensors"
DEFAULT_C096_OUT_DIR: Final = Path("eval/c096_siglip_encoder_lora_generation_gate_20260613")


def c096_siglip_variants() -> tuple[Variant, ...]:
    return (
        Variant("no_ip", None, 0.0),
        Variant("c094_shape_supervised_w14", C094_CHECKPOINT, 1.4),
        Variant("c095_feature_bridge_w14", C095_CHECKPOINT, 1.4),
        Variant("c096_lora_c094_w08", C094_CHECKPOINT, 0.8, C096_ENCODER_LORA),
        Variant("c096_lora_c094_w10", C094_CHECKPOINT, 1.0, C096_ENCODER_LORA),
        Variant("c096_lora_c094_w12", C094_CHECKPOINT, 1.2, C096_ENCODER_LORA),
        Variant("c096_lora_c094_w14", C094_CHECKPOINT, 1.4, C096_ENCODER_LORA),
    )


def run_c096_eval(probe_manifest_path: Path, config: EvalConfig) -> JsonObject:
    ensure_c090_output_writable(config.out_dir)
    config.out_dir.mkdir(parents=True, exist_ok=True)
    samples, baselines = materialize_c090_prompt_manifest(
        probe_manifest_path,
        out_dir=config.out_dir,
        reference_root=config.data_root,
    )
    ensure_baseline_candidates(samples, baselines)
    variants = c096_siglip_variants()
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
    rollup = write_c096_shape_metrics(config.out_dir / "summary.json", config.out_dir, audit)
    write_c096_report(config.out_dir, rollup, audit)
    return summary


def write_c096_shape_metrics(
    summary_path: Path,
    output_dir: Path,
    pixel_audit: JsonObject,
) -> JsonObject:
    write_c090_shape_metrics(summary_path, output_dir)
    metrics = _read_json(output_dir / "shape_metrics.json")
    rows = metrics.get("rows")
    if not isinstance(rows, list):
        raise TypeError("shape_metrics rows must be a list")
    rollup = summarize_c096_shape_rows(tuple(row for row in rows if isinstance(row, dict)), pixel_audit)
    metrics["rollup"] = rollup
    _write_json(output_dir / "shape_metrics.json", metrics)
    _write_json(output_dir / "metric_rollup.json", rollup)
    return rollup


def summarize_c096_shape_rows(rows: tuple[JsonObject, ...], pixel_audit: JsonObject) -> JsonObject:
    summaries = _variant_summaries(rows)
    c096 = _best_prefixed(summaries, "c096_")
    c094 = _best_named(summaries, ("c094_shape_supervised_w14",))
    c095 = _best_named(summaries, ("c095_feature_bridge_w14",))
    qwen = _best_named(summaries, BASELINE_VARIANTS)
    heldout = {
        "best_c096_uplift": _uplift(rows, "heldout07", c096[0]),
        "c094_w14_uplift": _uplift(rows, "heldout07", "c094_shape_supervised_w14"),
        "c095_w14_uplift": _uplift(rows, "heldout07", "c095_feature_bridge_w14"),
    }
    blank_rows = _blank_like_rows(pixel_audit, "c096_")
    return {
        "variant_summaries": summaries,
        "best_c096_variant": c096[0],
        "best_c094_variant": c094[0],
        "best_c095_variant": c095[0],
        "best_qwen_baseline_variant": qwen[0],
        "heldout07": heldout,
        "c096_blank_like_rows": blank_rows,
        "decision": _decision(c096, c094, c095, qwen, heldout, blank_rows),
    }


def write_c096_report(output_dir: Path, rollup: JsonObject, audit: JsonObject) -> None:
    lines = [
        "# c096 SigLIP Encoder LoRA Hard-Shape Gate",
        "",
        f"- Decision: `{rollup['decision']}`",
        f"- Best c096: `{rollup['best_c096_variant']}`",
        f"- Best c094 baseline: `{rollup['best_c094_variant']}`",
        f"- Best c095 baseline: `{rollup['best_c095_variant']}`",
        f"- Best Qwen baseline: `{rollup['best_qwen_baseline_variant']}`",
        f"- Heldout07: `{json.dumps(rollup['heldout07'], ensure_ascii=False)}`",
        f"- C096 blank-like rows: `{json.dumps(rollup['c096_blank_like_rows'], ensure_ascii=False)}`",
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
    }


def _best_prefixed(summaries: JsonObject, prefix: str) -> tuple[str, float]:
    return max(
        ((key, float(value["mean_uplift"])) for key, value in summaries.items() if key.startswith(prefix) and isinstance(value, dict)),
        key=lambda item: item[1],
        default=("", float("-inf")),
    )


def _best_named(summaries: JsonObject, names: tuple[str, ...]) -> tuple[str, float]:
    return max(
        ((name, float(summaries[name]["mean_uplift"])) for name in names if isinstance(summaries.get(name), dict)),
        key=lambda item: item[1],
        default=("", float("-inf")),
    )


def _uplift(rows: tuple[JsonObject, ...], sample: str, variant: str) -> float:
    for row in rows:
        if row.get("sample") == sample and row.get("variant") == variant:
            return float(row["uplift"])
    return float("-inf")


def _blank_like_rows(pixel_audit: JsonObject, prefix: str) -> list[str]:
    rows = pixel_audit.get("rows", [])
    if not isinstance(rows, list):
        return []
    return [
        str(row.get("name", ""))
        for row in rows
        if isinstance(row, dict)
        and f"_{prefix}" in str(row.get("name", ""))
        and row.get("nonblank") is False
    ]


def _decision(
    c096: tuple[str, float],
    c094: tuple[str, float],
    c095: tuple[str, float],
    qwen: tuple[str, float],
    heldout: JsonObject,
    blank_rows: list[str],
) -> str:
    heldout_best = float(heldout["best_c096_uplift"])
    heldout_base = max(float(heldout["c094_w14_uplift"]), float(heldout["c095_w14_uplift"]))
    if blank_rows:
        return "c096_encoder_lora_not_promoted_blank_outputs"
    if c096[1] >= max(c094[1], c095[1]) + 0.010 and heldout_best >= heldout_base + 0.020:
        return "c096_encoder_lora_candidate_for_larger_gate"
    if c096[1] >= qwen[1] - 0.010 and heldout_best >= 0.030:
        return "c096_encoder_lora_near_qwen_candidate"
    return "c096_encoder_lora_not_promoted_requires_data_expansion_or_deeper_encoder_training"


app = typer.Typer(add_completion=False)


@app.command()
def main(
    probe_manifest_path: Annotated[Path, typer.Argument()],
    out_dir: Annotated[Path, typer.Option()] = DEFAULT_C096_OUT_DIR,
    base_url: Annotated[str, typer.Option()] = DEFAULT_BASE_URL,
    data_root: Annotated[Path, typer.Option()] = DEFAULT_DATA_ROOT,
    comfy_input: Annotated[Path, typer.Option()] = DEFAULT_COMFY_INPUT,
    comfy_output: Annotated[Path, typer.Option()] = DEFAULT_COMFY_OUTPUT,
) -> None:
    try:
        run_c096_eval(
            probe_manifest_path,
            EvalConfig(data_root, base_url, out_dir, comfy_input, comfy_output),
        )
    except (EvalOutputAlreadyExistsError, MissingBaselineCandidateError) as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(1) from exc
    typer.echo(f"wrote {out_dir / 'contact_sheet_hard_shape.jpg'}")


if __name__ == "__main__":
    app()
