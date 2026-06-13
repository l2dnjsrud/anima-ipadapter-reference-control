# /// script
# dependencies = ["typer"]
# ///
# --- How to run -----------------------------------------------------
# PYTHONPATH=. /home/wktwin/anima-lora-training-bundle/anima_lora/.venv/bin/python \
#   tools/c088_report.py --output-dir eval/c088_shape_silhouette_feature_probe_20260613

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated

import typer

from tools.c088_probe_io import DEFAULT_OUT_DIR, write_json
from tools.siglip_auto_caption_types import JsonObject


FEATURE_FILES = {
    "edge_projection_silhouette": "shape_silhouette_metrics.json",
    "qwenvl": "qwenvl_embedding_metrics.json",
    "siglip2": "siglip_embedding_metrics.json",
    "pe": "pe_embedding_metrics.json",
}


@dataclass(frozen=True, slots=True)
class C088ReportError(Exception):
    detail: str

    def __str__(self) -> str:
        return self.detail


def build_c088_rollup(output_dir: Path) -> JsonObject:
    metrics = {feature: _read_metric(output_dir / filename) for feature, filename in FEATURE_FILES.items()}
    summaries = {
        feature: _summary_row(feature, metric)
        for feature, metric in metrics.items()
    }
    rollup = {
        "experiment": "c088_shape_silhouette_feature_probe",
        "feature_summaries": summaries,
        "hard_case_notes": _hard_case_notes(metrics),
        "decision": _decision(summaries),
        "next_direction": _next_direction(summaries),
    }
    write_json(output_dir / "metrics.json", {"features": metrics})
    write_json(output_dir / "metric_rollup.json", rollup)
    (output_dir / "report.md").write_text(_report(rollup), encoding="utf-8")
    return rollup


def _read_metric(path: Path) -> JsonObject:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise C088ReportError(f"metric root must be object: {path}")
    return raw


def _summary_row(feature: str, metric: JsonObject) -> JsonObject:
    summary = metric.get("summary")
    if not isinstance(summary, dict):
        raise C088ReportError(f"metric missing summary: {feature}")
    return {
        "cases": int(summary["cases"]),
        "supported_cases": int(summary["supported_cases"]),
        "support_rate": float(summary["support_rate"]),
        "source_decision": str(summary["decision"]),
    }


def _decision(summaries: dict[str, JsonObject]) -> str:
    shape_rate = float(summaries["edge_projection_silhouette"]["support_rate"])
    embedding_supports = sum(
        1 for name in ("qwenvl", "siglip2", "pe")
        if float(summaries[name]["support_rate"]) >= 0.5
    )
    if shape_rate >= 0.5 and embedding_supports >= 2:
        return "c089_supervised_feature_objective_viable"
    if shape_rate >= 0.5:
        return "shape_signal_present_encoder_embedding_not_enough"
    return "encoder_side_checkpoint_adaptation_required"


def _hard_case_notes(metrics: dict[str, JsonObject]) -> JsonObject:
    notes: dict[str, JsonObject] = {}
    for feature, metric in metrics.items():
        decisions = metric.get("case_decisions")
        if not isinstance(decisions, list):
            continue
        for raw in decisions:
            if isinstance(raw, dict) and raw.get("sample") == "heldout07":
                notes[feature] = {
                    "best_variant": str(raw["best_variant"]),
                    "best_uplift": float(raw["best_uplift"]),
                    "top_margin": float(raw["top_margin"]),
                    "decision": str(raw["decision"]),
                }
    return notes


def _next_direction(summaries: dict[str, JsonObject]) -> str:
    decision = _decision(summaries)
    match decision:
        case "c089_supervised_feature_objective_viable":
            return "train_shape_silhouette_supervised_feature_objective"
        case "shape_signal_present_encoder_embedding_not_enough":
            return "distill_edge_silhouette_or_train_encoder_side_shape_checkpoint"
        case "encoder_side_checkpoint_adaptation_required":
            return "encoder_side_checkpoint_adaptation"
        case unexpected:
            raise C088ReportError(f"unknown c088 decision: {unexpected}")


def _report(rollup: JsonObject) -> str:
    summaries = rollup["feature_summaries"]
    if not isinstance(summaries, dict):
        raise C088ReportError("rollup feature_summaries must be object")
    hard_cases = rollup["hard_case_notes"]
    if not isinstance(hard_cases, dict):
        raise C088ReportError("rollup hard_case_notes must be object")
    lines = [
        "# c088 Shape/Silhouette Feature Probe",
        "",
        f"- Decision: `{rollup['decision']}`",
        f"- Next direction: `{rollup['next_direction']}`",
        "",
        "## Feature Summaries",
        "",
        "| feature | cases | supported | support rate | source decision |",
        "| --- | ---: | ---: | ---: | --- |",
    ]
    for feature, raw in summaries.items():
        if not isinstance(raw, dict):
            raise C088ReportError("feature summary must be object")
        lines.append(
            f"| `{feature}` | `{raw['cases']}` | `{raw['supported_cases']}` | "
            f"`{raw['support_rate']}` | `{raw['source_decision']}` |"
        )
    lines.extend(
        [
            "",
            "## heldout07 Hard Case",
            "",
            "| feature | best variant | uplift | margin | decision |",
            "| --- | --- | ---: | ---: | --- |",
        ]
    )
    for feature, raw in hard_cases.items():
        if not isinstance(raw, dict):
            raise C088ReportError("hard case summary must be object")
        lines.append(
            f"| `{feature}` | `{raw['best_variant']}` | `{raw['best_uplift']}` | "
            f"`{raw['top_margin']}` | `{raw['decision']}` |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "c088 checks whether the reference-control failure is merely an adapter head issue or whether the image encoder feature space is missing shape/silhouette signal.",
            "The edge/projection/silhouette metric and PE contain partial shape signal, but QwenVL and SigLIP2 do not reach the support threshold on this c087 hard-shape set.",
            "Because the active native adapter path depends on QwenVL/SigLIP-like image embeddings, the next step should not repeat broad adapter continuation; it should either distill explicit shape features or train an encoder-side shape checkpoint.",
            "",
        ]
    )
    return "\n".join(lines)


app = typer.Typer(add_completion=False)


@app.command()
def main(
    output_dir: Annotated[Path, typer.Option()] = DEFAULT_OUT_DIR,
) -> None:
    rollup = build_c088_rollup(output_dir)
    typer.echo(json.dumps(rollup, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    app()
