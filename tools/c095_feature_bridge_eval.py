from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Final

import typer

from tools.c088_shape_metrics import _cosine, _shape_feature
from tools.c090_siglip_hard_shape_data import (
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
from tools.c090_siglip_hard_shape_report import write_extended_contact_sheet
from tools.c091_siglip_hard_shape_eval import C089_CHECKPOINT, C091_CHECKPOINT
from tools.c092_siglip_qwen_target_eval import (
    C092_CHECKPOINT,
    _read_json,
    _run_siglip_variants,
    _write_json,
    _write_summary,
)
from tools.c093_siglip_anti_collapse_eval import C093_CHECKPOINT
from tools.c094_shape_eval import C094_CHECKPOINT, write_c094_shape_metrics
from tools.c095_feature_bridge_metrics import summarize_c095_shape_rows
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


C095_CHECKPOINT: Final = "anima_siglip_ip_adapter_c095_feature_bridge_b128_0096_20260613.safetensors"
DEFAULT_C095_OUT_DIR: Final = Path("eval/c095_siglip_feature_bridge_generation_gate_20260613")


def c095_siglip_variants() -> tuple[Variant, ...]:
    return (
        Variant("no_ip", None, 0.0),
        Variant("siglip_pilot_w14", SIGLIP_PILOT_CHECKPOINT, 1.4),
        Variant("c089_shape_w14", C089_CHECKPOINT, 1.4),
        Variant("c091_feature_calibrator_w14", C091_CHECKPOINT, 1.4),
        Variant("c092_qwen_target_w10", C092_CHECKPOINT, 1.0),
        Variant("c092_qwen_target_w14", C092_CHECKPOINT, 1.4),
        Variant("c093_anti_collapse_w14", C093_CHECKPOINT, 1.4),
        Variant("c094_shape_supervised_w14", C094_CHECKPOINT, 1.4),
        Variant("c095_feature_bridge_w08", C095_CHECKPOINT, 0.8),
        Variant("c095_feature_bridge_w10", C095_CHECKPOINT, 1.0),
        Variant("c095_feature_bridge_w12", C095_CHECKPOINT, 1.2),
        Variant("c095_feature_bridge_w14", C095_CHECKPOINT, 1.4),
    )


def run_c095_eval(probe_manifest_path: Path, config: EvalConfig) -> JsonObject:
    ensure_c090_output_writable(config.out_dir)
    config.out_dir.mkdir(parents=True, exist_ok=True)
    samples, baselines = materialize_c090_prompt_manifest(
        probe_manifest_path,
        out_dir=config.out_dir,
        reference_root=config.data_root,
    )
    ensure_baseline_candidates(samples, baselines)
    variants = c095_siglip_variants()
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
    rollup = write_c095_shape_metrics(config.out_dir / "summary.json", config.out_dir, audit)
    write_c095_report(config.out_dir, rollup, audit)
    return summary


def write_c095_shape_metrics(
    summary_path: Path,
    output_dir: Path,
    pixel_audit: JsonObject,
) -> JsonObject:
    write_c094_shape_metrics(summary_path, output_dir, pixel_audit)
    metrics = _read_json(output_dir / "shape_metrics.json")
    rows = metrics.get("rows")
    if not isinstance(rows, list):
        raise TypeError("shape_metrics rows must be a list")
    rollup = summarize_c095_shape_rows(
        tuple(row for row in rows if isinstance(row, dict)),
        diversity_by_variant=_diversity_by_variant(_read_json(summary_path)),
        pixel_audit=pixel_audit,
    )
    metrics["rollup"] = rollup
    _write_json(output_dir / "shape_metrics.json", metrics)
    _write_json(output_dir / "metric_rollup.json", rollup)
    return rollup


def write_c095_report(output_dir: Path, rollup: JsonObject, audit: JsonObject) -> None:
    lines = [
        "# c095 SigLIP Feature-Bridge Hard-Shape Gate",
        "",
        f"- Decision: `{rollup['decision']}`",
        f"- Best c095: `{rollup['best_c095_variant']}`",
        f"- Best c094 baseline: `{rollup['best_c094_variant']}`",
        f"- Best c093 baseline: `{rollup['best_c093_variant']}`",
        f"- Best c092 baseline: `{rollup['best_c092_variant']}`",
        f"- Best Qwen baseline: `{rollup['best_qwen_baseline_variant']}`",
        f"- Heldout07: `{json.dumps(rollup['heldout07'], ensure_ascii=False)}`",
        f"- C095 blank-like rows: `{json.dumps(rollup['c095_blank_like_rows'], ensure_ascii=False)}`",
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


def _diversity_by_variant(summary: JsonObject) -> dict[str, float]:
    samples = tuple(item for item in summary.get("samples", []) if isinstance(item, dict))
    labels = tuple(str(item["label"]) for item in samples if str(item.get("label", "")).startswith("crop_pair"))
    variants = tuple(
        str(item["label"])
        for item in summary.get("variants", [])
        if isinstance(item, dict) and item.get("checkpoint") is not None
    )
    return {variant: _mean_pairwise_distance(_variant_paths(summary, labels, variant)) for variant in variants}


def _variant_paths(summary: JsonObject, labels: tuple[str, ...], variant: str) -> tuple[Path, ...]:
    results = summary.get("results", {})
    if not isinstance(results, dict):
        return ()
    paths: list[Path] = []
    for label in labels:
        raw = results.get(f"{label}_{variant}")
        if isinstance(raw, dict) and isinstance(raw.get("image"), str):
            paths.append(Path(str(raw["image"])))
    return tuple(paths)


def _mean_pairwise_distance(paths: tuple[Path, ...]) -> float:
    features = [_shape_feature(path) for path in paths if path.is_file()]
    if len(features) < 2:
        return 0.0
    distances = [
        1.0 - _cosine(left, right)
        for index, left in enumerate(features)
        for right in features[index + 1 :]
    ]
    return sum(distances) / len(distances)


app = typer.Typer(add_completion=False)


@app.command()
def main(
    probe_manifest_path: Annotated[Path, typer.Argument()],
    out_dir: Annotated[Path, typer.Option()] = DEFAULT_C095_OUT_DIR,
    base_url: Annotated[str, typer.Option()] = DEFAULT_BASE_URL,
    data_root: Annotated[Path, typer.Option()] = DEFAULT_DATA_ROOT,
    comfy_input: Annotated[Path, typer.Option()] = DEFAULT_COMFY_INPUT,
    comfy_output: Annotated[Path, typer.Option()] = DEFAULT_COMFY_OUTPUT,
) -> None:
    try:
        run_c095_eval(
            probe_manifest_path,
            EvalConfig(data_root, base_url, out_dir, comfy_input, comfy_output),
        )
    except (EvalOutputAlreadyExistsError, MissingBaselineCandidateError) as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(1) from exc
    typer.echo(f"wrote {out_dir / 'contact_sheet_hard_shape.jpg'}")


if __name__ == "__main__":
    app()
