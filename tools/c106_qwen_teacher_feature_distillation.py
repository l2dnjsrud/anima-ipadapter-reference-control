# /// script
# dependencies = ["typer"]
# ///
# --- How to run -----------------------------------------------------
# PYTHONPATH=. /home/wktwin/anima-lora-training-bundle/anima_lora/.venv/bin/python \
#   tools/c106_qwen_teacher_feature_distillation.py build
# PYTHONPATH=. /home/wktwin/anima-lora-training-bundle/anima_lora/.venv/bin/python \
#   tools/c106_qwen_teacher_feature_distillation.py summarize

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from math import isfinite
from pathlib import Path
from typing import Annotated, Final

import typer

from tools.c079_manifest_io import read_jsonl, write_jsonl
from tools.siglip_auto_caption_types import JsonObject, JsonValue


DEFAULT_SOURCE_MANIFEST: Final = Path("training/manifests/c097_siglip_hard_shape_expanded_pairs_20260613.jsonl")
DEFAULT_IMAGE_ROOT: Final = Path(".tmp/c097_siglip_hard_shape_expanded_root")
DEFAULT_OUT_DIR: Final = Path("eval/c106_qwen_teacher_feature_distillation_20260613")
DEFAULT_PLAN_PATH: Final = Path("docs/c106_qwen_teacher_feature_distillation_plan_ko.md")
DEFAULT_PAIR_SCORE_PATH: Final = DEFAULT_OUT_DIR / "qwenvl_pair_scores.json"
C104_MARGIN: Final = 0.01997534079211105
MIN_TEACHER_MARGIN: Final = 0.05
MIN_TEACHER_AUC: Final = 0.85


@dataclass(frozen=True, slots=True)
class C106ProbeError(Exception):
    detail: str

    def __str__(self) -> str:
        return self.detail


@dataclass(frozen=True, slots=True)
class C106ProbeConfig:
    source_manifest: Path = DEFAULT_SOURCE_MANIFEST
    image_root: Path = DEFAULT_IMAGE_ROOT
    out_dir: Path = DEFAULT_OUT_DIR
    plan_path: Path | None = None

    @property
    def probe_manifest(self) -> Path:
        return self.out_dir / "probe_manifest.jsonl"

    @property
    def resolved_plan_path(self) -> Path:
        if self.plan_path is not None:
            return self.plan_path
        if self.out_dir == DEFAULT_OUT_DIR:
            return DEFAULT_PLAN_PATH
        return self.out_dir / "c106_qwen_teacher_feature_distillation_plan_ko.md"


@dataclass(frozen=True, slots=True)
class C106ManifestSummary:
    source: str
    source_manifest: str
    image_root: str
    probe_manifest: str
    selected_rows: int
    positive_rows: int
    explicit_negative_rows: int
    pair_probe_rows: int
    heldout_rows_used: int
    heldout_rows_rejected: int
    missing_path_count: int
    c105_selected_route: str
    c104_margin: float
    minimum_teacher_margin: float
    minimum_teacher_auc: float
    decision: str


def build_c106_probe_manifest(config: C106ProbeConfig = C106ProbeConfig()) -> C106ManifestSummary:
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
        manifest_rows.extend(_pair_rows(pair_id, raw, ids))
        selected_rows += 1
    if missing:
        raise C106ProbeError("missing c106 image paths: " + ", ".join(missing[:5]))
    if selected_rows == 0:
        raise C106ProbeError(f"c106 source manifest has no usable rows: {config.source_manifest}")
    write_jsonl(config.probe_manifest, tuple(manifest_rows))
    summary = C106ManifestSummary(
        source="C097/C087/C105",
        source_manifest=str(config.source_manifest),
        image_root=str(config.image_root),
        probe_manifest=str(config.probe_manifest),
        selected_rows=selected_rows,
        positive_rows=selected_rows,
        explicit_negative_rows=selected_rows,
        pair_probe_rows=len(manifest_rows),
        heldout_rows_used=0,
        heldout_rows_rejected=heldout_rejected,
        missing_path_count=0,
        c105_selected_route="qwen_teacher_distillation",
        c104_margin=C104_MARGIN,
        minimum_teacher_margin=MIN_TEACHER_MARGIN,
        minimum_teacher_auc=MIN_TEACHER_AUC,
        decision="ready_for_c106_qwen_teacher_probe",
    )
    _write_json(config.out_dir / "manifest_summary.json", asdict(summary))
    _write_plan(config.resolved_plan_path, summary)
    return summary


def write_c106_probe_summary(
    *,
    score_path: Path,
    out_dir: Path,
    manifest_summary_path: Path,
) -> JsonObject:
    blockers: list[str] = []
    if not score_path.is_file():
        blockers.append(f"missing score file: {score_path}")
        return _write_summary(out_dir, _blocked_summary(score_path, manifest_summary_path, blockers))
    scores = _read_json(score_path)
    summary = _summary_object(scores)
    positive_pairs = int(summary.get("positive_pairs", 0))
    negative_pairs = int(summary.get("negative_pairs", 0))
    margin = float(summary.get("separation_margin", float("nan")))
    auc = float(summary.get("pairwise_auc", float("nan")))
    finite_metrics = all(isfinite(value) for value in (margin, auc))
    if positive_pairs != negative_pairs or positive_pairs == 0:
        blockers.append("positive and explicit negative pair counts must match and be nonzero")
    decision = _decision(finite_metrics=finite_metrics, blockers=tuple(blockers), margin=margin, auc=auc)
    payload: JsonObject = {
        "score_path": str(score_path),
        "manifest_summary_path": str(manifest_summary_path),
        "rows_evaluated": positive_pairs,
        "explicit_negative_rows": negative_pairs,
        "finite_metrics": finite_metrics,
        "teacher_margin": margin,
        "teacher_auc": auc,
        "minimum_teacher_margin": MIN_TEACHER_MARGIN,
        "minimum_teacher_auc": MIN_TEACHER_AUC,
        "c104_margin": C104_MARGIN,
        "blockers": blockers,
        "decision": decision,
        "next_branch": _next_branch(decision),
    }
    return _write_summary(out_dir, payload)


def _row_ids(row: JsonObject, line_number: int, path: Path) -> tuple[str, str, str]:
    return (
        _string(row, "ref_id", line_number, path),
        _string(row, "tgt_id", line_number, path),
        _string(row, "neg_id", line_number, path),
    )


def _string(row: JsonObject, key: str, line_number: int, path: Path) -> str:
    value = row.get(key)
    if not isinstance(value, str):
        raise C106ProbeError(f"{path}:{line_number} missing {key}")
    return value


def _pair_rows(pair_id: str, row: JsonObject, ids: tuple[str, str, str]) -> tuple[JsonObject, JsonObject]:
    ref_id, tgt_id, neg_id = ids
    group = str(row.get("shape_group", ""))
    negative_group = str(row.get("negative_shape_group", ""))
    base: JsonObject = {
        "pair_id": pair_id,
        "anchor_id": ref_id,
        "anchor_group": group,
        "prompt": str(row.get("prompt", "")),
        "source_pose_pair": str(row.get("source_pose_pair", "")),
        "teacher_route": "qwen_teacher_distillation",
    }
    return (
        base | {"label": "positive", "candidate_id": tgt_id, "candidate_group": group},
        base | {"label": "negative", "candidate_id": neg_id, "candidate_group": negative_group},
    )


def _summary_object(scores: JsonObject) -> JsonObject:
    raw = scores.get("summary")
    if not isinstance(raw, dict):
        raise C106ProbeError("QwenVL pair score JSON missing summary")
    return raw


def _decision(*, finite_metrics: bool, blockers: tuple[str, ...], margin: float, auc: float) -> str:
    if not blockers and finite_metrics and margin >= MIN_TEACHER_MARGIN and auc >= MIN_TEACHER_AUC and margin > C104_MARGIN:
        return "c106_probe_pass_prepare_training"
    return "c106_probe_not_enough_signal_or_blocked"


def _next_branch(decision: str) -> str:
    if decision == "c106_probe_pass_prepare_training":
        return "c107_bounded_qwen_teacher_distillation_training"
    return "manual_external_annotation_or_stronger_teacher_checkpoint"


def _blocked_summary(score_path: Path, manifest_summary_path: Path, blockers: list[str]) -> JsonObject:
    return {
        "score_path": str(score_path),
        "manifest_summary_path": str(manifest_summary_path),
        "rows_evaluated": 0,
        "explicit_negative_rows": 0,
        "finite_metrics": False,
        "teacher_margin": None,
        "teacher_auc": None,
        "minimum_teacher_margin": MIN_TEACHER_MARGIN,
        "minimum_teacher_auc": MIN_TEACHER_AUC,
        "c104_margin": C104_MARGIN,
        "blockers": blockers,
        "decision": "c106_probe_not_enough_signal_or_blocked",
        "next_branch": "manual_external_annotation_or_stronger_teacher_checkpoint",
    }


def _write_summary(out_dir: Path, payload: JsonObject) -> JsonObject:
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_json(out_dir / "probe_summary.json", payload)
    (out_dir / "report.md").write_text(_report(payload), encoding="utf-8")
    return payload


def _report(summary: JsonObject) -> str:
    return f"""# C106 Qwen Teacher Feature Distillation Probe

- Decision: `{summary['decision']}`
- Rows evaluated / explicit negatives: `{summary['rows_evaluated']}` / `{summary['explicit_negative_rows']}`
- Teacher margin: `{summary['teacher_margin']}`
- Teacher AUC: `{summary['teacher_auc']}`
- Next branch: `{summary['next_branch']}`
"""


def _write_plan(path: Path, summary: C106ManifestSummary) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f"""# C106 Qwen Teacher Feature Distillation Plan

C105에서 선택한 `qwen_teacher_distillation` 경로의 첫 번째 문턱 실험으로, C097 hard-shape positive/explicit-negative 쌍을 QwenVL pair feature probe로 변환한다.

- Source / manifest / output: `{summary.source}` / `{summary.source_manifest}` / `{summary.probe_manifest}`
- Selected / pair-probe / heldout rows: `{summary.selected_rows}` / `{summary.pair_probe_rows}` / `{summary.heldout_rows_used}`
- Gate: margin >= `{summary.minimum_teacher_margin}`, AUC >= `{summary.minimum_teacher_auc}`, C104 margin `{summary.c104_margin}` 초과
- Branches: pass `c107_bounded_qwen_teacher_distillation_training`, stop `manual_external_annotation_or_stronger_teacher_checkpoint`
""",
        encoding="utf-8",
    )


def _read_json(path: Path) -> JsonObject:
    raw: JsonValue = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise C106ProbeError(f"json root must be object: {path}")
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
    plan_path: Annotated[Path, typer.Option()] = DEFAULT_PLAN_PATH,
) -> None:
    summary = build_c106_probe_manifest(C106ProbeConfig(source_manifest, image_root, out_dir, plan_path))
    typer.echo(f"wrote {summary.pair_probe_rows} pair rows to {summary.probe_manifest}")


@app.command()
def summarize(
    score_path: Annotated[Path, typer.Option()] = DEFAULT_PAIR_SCORE_PATH,
    out_dir: Annotated[Path, typer.Option()] = DEFAULT_OUT_DIR,
    manifest_summary_path: Annotated[Path, typer.Option()] = DEFAULT_OUT_DIR / "manifest_summary.json",
) -> None:
    summary = write_c106_probe_summary(
        score_path=score_path,
        out_dir=out_dir,
        manifest_summary_path=manifest_summary_path,
    )
    typer.echo(f"decision {summary['decision']} from {score_path}")


if __name__ == "__main__":
    app()
