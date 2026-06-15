# /// script
# dependencies = ["typer"]
# ///
# --- How to run -----------------------------------------------------
# PYTHONPATH=. /home/wktwin/anima-lora-training-bundle/anima_lora/.venv/bin/python \
#   tools/c104_expanded_qwen_target_siglip_probe.py build
# PYTHONPATH=. /home/wktwin/anima-lora-training-bundle/anima_lora/.venv/bin/python \
#   tools/c104_expanded_qwen_target_siglip_probe.py summarize

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from math import isfinite
from pathlib import Path
from typing import Annotated, Final

import typer

from tools.c079_manifest_io import read_jsonl, write_jsonl
from tools.siglip_auto_caption_types import JsonObject


DEFAULT_SOURCE_MANIFEST: Final = Path("training/manifests/c097_siglip_hard_shape_expanded_pairs_20260613.jsonl")
DEFAULT_IMAGE_ROOT: Final = Path(".tmp/c097_siglip_hard_shape_expanded_root")
DEFAULT_OUT_DIR: Final = Path("eval/c104_expanded_qwen_target_siglip_probe_20260613")
DEFAULT_TOKEN_SCORE_PATH: Final = DEFAULT_OUT_DIR / "token_scores.json"
C098_BEST_MEAN_UPLIFT: Final = 0.0865313863
QWEN_BASELINE_MEAN_UPLIFT: Final = 0.1089544056
MIN_PASS_AUC: Final = 0.85


class C104ProbeError(Exception):
    def __init__(self, detail: str) -> None:
        self.detail = detail
        super().__init__(detail)


@dataclass(frozen=True, slots=True)
class C104ProbeConfig:
    source_manifest: Path = DEFAULT_SOURCE_MANIFEST
    image_root: Path = DEFAULT_IMAGE_ROOT
    out_dir: Path = DEFAULT_OUT_DIR

    @property
    def probe_manifest(self) -> Path:
        return self.out_dir / "probe_manifest.jsonl"

@dataclass(frozen=True, slots=True)
class C104ManifestSummary:
    source: str
    source_manifest: str
    image_root: str
    probe_manifest: str
    selected_rows: int
    positive_rows: int
    explicit_negative_rows: int
    token_probe_rows: int
    heldout_rows_used: int
    heldout_rows_rejected: int
    missing_path_count: int
    c098_best_mean_uplift: float
    qwen_baseline_mean_uplift: float
    decision: str


def build_c104_probe_manifest(config: C104ProbeConfig = C104ProbeConfig()) -> C104ManifestSummary:
    manifest_rows: list[JsonObject] = []
    missing: list[str] = []
    heldout_rejected = 0
    selected_rows = 0
    for line_number, raw in read_jsonl(config.source_manifest):
        ids = _row_ids(raw, line_number, config.source_manifest)
        if any("heldout" in image_id for image_id in ids):
            heldout_rejected += 1
            continue
        for image_id in ids:
            path = config.image_root / f"{image_id}.jpg"
            if not path.is_file():
                missing.append(str(path))
        pair_id = f"pair_{selected_rows:03d}"
        manifest_rows.extend(_token_rows(pair_id, raw, ids))
        selected_rows += 1
    if missing:
        raise C104ProbeError("missing c104 image paths: " + ", ".join(missing[:5]))
    if selected_rows == 0:
        raise C104ProbeError("c104 source manifest has no usable rows")
    write_jsonl(config.probe_manifest, tuple(manifest_rows))
    summary = C104ManifestSummary(
        source="C097/C087/C092",
        source_manifest=str(config.source_manifest),
        image_root=str(config.image_root),
        probe_manifest=str(config.probe_manifest),
        selected_rows=selected_rows,
        positive_rows=selected_rows,
        explicit_negative_rows=selected_rows,
        token_probe_rows=len(manifest_rows),
        heldout_rows_used=0,
        heldout_rows_rejected=heldout_rejected,
        missing_path_count=0,
        c098_best_mean_uplift=C098_BEST_MEAN_UPLIFT,
        qwen_baseline_mean_uplift=QWEN_BASELINE_MEAN_UPLIFT,
        decision="ready_for_c104_siglip_token_probe",
    )
    _write_json(config.out_dir / "manifest_summary.json", asdict(summary))
    return summary


def write_c104_probe_summary(
    *,
    token_score_path: Path,
    out_dir: Path,
    manifest_summary_path: Path,
    c098_best_mean_uplift: float = C098_BEST_MEAN_UPLIFT,
    qwen_baseline_mean_uplift: float = QWEN_BASELINE_MEAN_UPLIFT,
) -> JsonObject:
    scores = _read_json(token_score_path)
    metric_summaries = _metric_summaries(scores)
    best_metric, best_margin, best_auc = _best_metric(metric_summaries)
    pair_rows = _pair_rows(scores)
    finite_metrics = all(
        isfinite(float(value))
        for row in pair_rows
        for metric in ("pooled", "mean_max_token", "topk_token")
        for value in (row[f"{metric}_margin"],)
    )
    pass_threshold = max(qwen_baseline_mean_uplift, c098_best_mean_uplift + 0.01)
    decision = (
        "c104_probe_pass_prepare_training"
        if finite_metrics and best_margin >= pass_threshold and best_auc >= MIN_PASS_AUC
        else "c104_probe_not_enough_signal"
    )
    summary: JsonObject = {
        "token_score_path": str(token_score_path),
        "manifest_summary_path": str(manifest_summary_path),
        "rows_evaluated": len(pair_rows),
        "explicit_negative_rows": len(pair_rows),
        "finite_metrics": finite_metrics,
        "c098_best_mean_uplift": c098_best_mean_uplift,
        "qwen_baseline_mean_uplift": qwen_baseline_mean_uplift,
        "pass_threshold": pass_threshold,
        "min_pass_auc": MIN_PASS_AUC,
        "best_siglip_metric": best_metric,
        "best_siglip_margin_or_uplift": best_margin,
        "best_pairwise_auc": best_auc,
        "metric_summaries": metric_summaries,
        "decision": decision,
        "next_branch": (
            "c105_small_qwen_target_siglip_training"
            if decision == "c104_probe_pass_prepare_training"
            else "stronger_encoder_checkpoint_or_manual_external_annotation"
        ),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(out_dir / "per_row_metrics.jsonl", tuple(pair_rows))
    _write_json(out_dir / "summary.json", summary)
    (out_dir / "report.md").write_text(_report(summary), encoding="utf-8")
    return summary


def _row_ids(row: JsonObject, line_number: int, path: Path) -> tuple[str, str, str]:
    return (
        _string(row, "ref_id", line_number, path),
        _string(row, "tgt_id", line_number, path),
        _string(row, "neg_id", line_number, path),
    )


def _string(row: JsonObject, key: str, line_number: int, path: Path) -> str:
    value = row.get(key)
    if not isinstance(value, str):
        raise C104ProbeError(f"{path}:{line_number} missing {key}")
    return value


def _token_rows(pair_id: str, row: JsonObject, ids: tuple[str, str, str]) -> tuple[JsonObject, JsonObject]:
    ref_id, tgt_id, neg_id = ids
    base: JsonObject = {
        "pair_id": pair_id,
        "anchor_id": ref_id,
        "shape_group": str(row.get("shape_group", "")),
        "negative_shape_group": str(row.get("negative_shape_group", "")),
        "source_pose_pair": str(row.get("source_pose_pair", "")),
    }
    return (
        base | {"label": "positive", "candidate_id": tgt_id},
        base | {"label": "negative", "candidate_id": neg_id},
    )


def _metric_summaries(score_json: JsonObject) -> JsonObject:
    raw = score_json.get("summaries")
    if not isinstance(raw, dict):
        raise C104ProbeError("token score JSON missing summaries")
    return raw


def _best_metric(summaries: JsonObject) -> tuple[str, float, float]:
    best_name = ""
    best_margin = float("-inf")
    best_auc = 0.0
    for name, raw in summaries.items():
        if isinstance(raw, dict):
            margin = float(raw.get("separation_margin", float("-inf")))
            auc = float(raw.get("pairwise_auc", 0.0))
            if margin > best_margin:
                best_name = str(name)
                best_margin = margin
                best_auc = auc
    if not best_name:
        raise C104ProbeError("token score JSON has no metric summaries")
    return best_name, best_margin, best_auc


def _pair_rows(score_json: JsonObject) -> list[JsonObject]:
    grouped: dict[str, dict[str, JsonObject]] = {}
    rows = score_json.get("rows")
    if not isinstance(rows, list):
        raise C104ProbeError("token score JSON missing rows")
    for raw in rows:
        if isinstance(raw, dict):
            grouped.setdefault(str(raw["pair_id"]), {})[str(raw["label"])] = raw
    return [_pair_metric(pair_id, pair) for pair_id, pair in sorted(grouped.items()) if "positive" in pair and "negative" in pair]


def _pair_metric(pair_id: str, pair: dict[str, JsonObject]) -> JsonObject:
    positive = _scores(pair["positive"])
    negative = _scores(pair["negative"])
    return {
        "pair_id": pair_id,
        "positive": positive,
        "negative": negative,
        "pooled_margin": float(positive["pooled"]) - float(negative["pooled"]),
        "mean_max_token_margin": float(positive["mean_max_token"]) - float(negative["mean_max_token"]),
        "topk_token_margin": float(positive["topk_token"]) - float(negative["topk_token"]),
    }


def _scores(row: JsonObject) -> JsonObject:
    raw = row.get("scores")
    if not isinstance(raw, dict):
        raise C104ProbeError("token score row missing scores")
    return raw


def _report(summary: JsonObject) -> str:
    return "\n".join(
        [
            "# C104 Expanded Qwen-Target SigLIP Probe",
            "",
            f"- Decision: `{summary['decision']}`",
            f"- Rows evaluated: `{summary['rows_evaluated']}`",
            f"- Best metric: `{summary['best_siglip_metric']}`",
            f"- Best margin: `{summary['best_siglip_margin_or_uplift']}`",
            f"- Best AUC: `{summary['best_pairwise_auc']}`",
            f"- Pass threshold: `{summary['pass_threshold']}`",
            f"- Next branch: `{summary['next_branch']}`",
            "",
        ]
    )


def _read_json(path: Path) -> JsonObject:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise C104ProbeError(f"json root must be object: {path}")
    return raw


def _write_json(path: Path, payload: JsonObject) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


app = typer.Typer(add_completion=False)


@app.command()
def build(
    source_manifest: Annotated[Path, typer.Option()] = DEFAULT_SOURCE_MANIFEST,
    image_root: Annotated[Path, typer.Option()] = DEFAULT_IMAGE_ROOT,
    out_dir: Annotated[Path, typer.Option()] = DEFAULT_OUT_DIR,
) -> None:
    summary = build_c104_probe_manifest(C104ProbeConfig(source_manifest, image_root, out_dir))
    typer.echo(f"wrote {summary.token_probe_rows} token rows to {summary.probe_manifest}")


@app.command()
def summarize(
    token_score_path: Annotated[Path, typer.Option()] = DEFAULT_TOKEN_SCORE_PATH,
    out_dir: Annotated[Path, typer.Option()] = DEFAULT_OUT_DIR,
    manifest_summary_path: Annotated[Path, typer.Option()] = DEFAULT_OUT_DIR / "manifest_summary.json",
) -> None:
    summary = write_c104_probe_summary(
        token_score_path=token_score_path,
        out_dir=out_dir,
        manifest_summary_path=manifest_summary_path,
    )
    typer.echo(f"decision {summary['decision']} from {token_score_path}")


if __name__ == "__main__":
    app()
