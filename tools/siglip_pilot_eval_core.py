from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Final

import torch
from safetensors.torch import load_file


MIN_RELATIVE_DELTA: Final = 1e-8


@dataclass(frozen=True, slots=True)
class CheckpointDelta:
    smoke_checkpoint: str
    pilot_checkpoint: str
    smoke_tensors: int
    pilot_tensors: int
    common_tensors: int
    missing_in_pilot: tuple[str, ...]
    extra_in_pilot: tuple[str, ...]
    shape_mismatches: tuple[str, ...]
    key_match: bool
    changed_tensors: int
    unchanged_tensors: int
    relative_l2_delta: float
    mean_abs_delta: float
    max_abs_delta: float


@dataclass(frozen=True, slots=True)
class PeBaseline:
    summary_path: str
    passed: bool
    best_scale: float | None
    generated_count: int
    best_mean_uplift: float | None
    best_improved_rate: float | None


@dataclass(frozen=True, slots=True)
class ScaleDecision:
    label: str
    scale_next: bool
    quality_proven: bool
    reason: str


@dataclass(frozen=True, slots=True)
class PilotEvaluation:
    metrics_path: Path
    report_path: Path
    checkpoint_delta: CheckpointDelta
    pe_baseline: PeBaseline
    decision: ScaleDecision


@dataclass(frozen=True, slots=True)
class _DeltaAccumulator:
    changed: int
    unchanged: int
    l2_delta_sq: float
    smoke_l2_sq: float
    abs_sum: float
    max_abs: float
    total_values: int
    shape_mismatches: tuple[str, ...]


def compare_checkpoints(smoke_checkpoint: Path, pilot_checkpoint: Path) -> CheckpointDelta:
    smoke = load_file(str(smoke_checkpoint), device="cpu")
    pilot = load_file(str(pilot_checkpoint), device="cpu")
    smoke_keys = set(smoke)
    pilot_keys = set(pilot)
    common = tuple(sorted(smoke_keys & pilot_keys))
    missing = tuple(sorted(smoke_keys - pilot_keys))
    extra = tuple(sorted(pilot_keys - smoke_keys))
    totals = _measure_common_tensors(smoke, pilot, common)
    l2_delta = math.sqrt(totals.l2_delta_sq)
    smoke_l2 = math.sqrt(totals.smoke_l2_sq)
    relative = l2_delta / smoke_l2 if smoke_l2 > 0.0 else 0.0
    mean_abs = totals.abs_sum / totals.total_values if totals.total_values > 0 else 0.0
    return CheckpointDelta(
        smoke_checkpoint=str(smoke_checkpoint),
        pilot_checkpoint=str(pilot_checkpoint),
        smoke_tensors=len(smoke),
        pilot_tensors=len(pilot),
        common_tensors=len(common),
        missing_in_pilot=missing,
        extra_in_pilot=extra,
        shape_mismatches=totals.shape_mismatches,
        key_match=not missing and not extra and not totals.shape_mismatches,
        changed_tensors=totals.changed,
        unchanged_tensors=totals.unchanged,
        relative_l2_delta=relative,
        mean_abs_delta=mean_abs,
        max_abs_delta=totals.max_abs,
    )


def read_pe_baseline(summary_path: Path) -> PeBaseline:
    raw = json.loads(summary_path.read_text(encoding="utf-8"))
    scale_summaries = raw.get("scale_summaries", [])
    best_scale = raw.get("best_scale")
    best_row = _find_best_scale_row(scale_summaries, best_scale)
    return PeBaseline(
        summary_path=str(summary_path),
        passed=bool(raw.get("pass", False)),
        best_scale=float(best_scale) if best_scale is not None else None,
        generated_count=int(raw.get("generated_count", 0)),
        best_mean_uplift=_optional_float(best_row.get("mean_uplift") if best_row else None),
        best_improved_rate=_optional_float(best_row.get("improved_rate") if best_row else None),
    )


def decide_scale(
    *,
    key_match: bool,
    changed_tensors: int,
    relative_l2_delta: float,
    pe_baseline_pass: bool,
    siglip_visual_eval_available: bool,
) -> ScaleDecision:
    moved = key_match and changed_tensors > 0 and relative_l2_delta > MIN_RELATIVE_DELTA
    scale_next = moved and pe_baseline_pass
    if scale_next:
        reason = "pilot moved from smoke, but SigLIP image-generation workflow is not evaluated yet"
        if siglip_visual_eval_available:
            reason = (
                "pilot moved from smoke; visual availability flags are not accepted as "
                "quality proof without scored contact-sheet artifacts"
            )
        return ScaleDecision(
            label="scale_after_siglip_workflow_eval",
            scale_next=True,
            quality_proven=False,
            reason=reason,
        )
    return ScaleDecision(
        label="stop_or_fix_before_scale",
        scale_next=False,
        quality_proven=False,
        reason="checkpoint movement, key compatibility, or PE baseline evidence is insufficient",
    )


def evaluate_pilot(
    *,
    smoke_checkpoint: Path,
    pilot_checkpoint: Path,
    pe_summary_path: Path,
    out_dir: Path,
    siglip_visual_eval_available: bool,
) -> PilotEvaluation:
    delta = compare_checkpoints(smoke_checkpoint, pilot_checkpoint)
    pe_baseline = read_pe_baseline(pe_summary_path)
    decision = decide_scale(
        key_match=delta.key_match,
        changed_tensors=delta.changed_tensors,
        relative_l2_delta=delta.relative_l2_delta,
        pe_baseline_pass=pe_baseline.passed,
        siglip_visual_eval_available=siglip_visual_eval_available,
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = out_dir / "metrics.json"
    report_path = out_dir / "report.md"
    metrics = {
        "methodology": "checkpoint_delta_proxy_plus_existing_pe_contact_sheet_baseline",
        "siglip_visual_eval_available": siglip_visual_eval_available,
        "checkpoint_delta": asdict(delta),
        "pe_baseline": asdict(pe_baseline),
        "decision": asdict(decision),
    }
    metrics_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    report_path.write_text(_render_report(delta, pe_baseline, decision), encoding="utf-8")
    return PilotEvaluation(metrics_path, report_path, delta, pe_baseline, decision)


def _measure_common_tensors(smoke, pilot, common: tuple[str, ...]) -> _DeltaAccumulator:
    changed = 0
    unchanged = 0
    l2_delta_sq = 0.0
    smoke_l2_sq = 0.0
    abs_sum = 0.0
    max_abs = 0.0
    total_values = 0
    shape_mismatches: list[str] = []
    for key in common:
        smoke_tensor = smoke[key].detach().to(dtype=torch.float64)
        pilot_tensor = pilot[key].detach().to(dtype=torch.float64)
        if smoke_tensor.shape != pilot_tensor.shape:
            shape_mismatches.append(key)
            changed += 1
            continue
        delta = pilot_tensor - smoke_tensor
        abs_delta = delta.abs()
        l2_delta_sq += float(torch.sum(delta * delta).item())
        smoke_l2_sq += float(torch.sum(smoke_tensor * smoke_tensor).item())
        abs_sum += float(torch.sum(abs_delta).item())
        total_values += delta.numel()
        if delta.numel() > 0:
            max_abs = max(max_abs, float(torch.max(abs_delta).item()))
        if torch.equal(smoke_tensor, pilot_tensor):
            unchanged += 1
        else:
            changed += 1
    return _DeltaAccumulator(
        changed=changed,
        unchanged=unchanged,
        l2_delta_sq=l2_delta_sq,
        smoke_l2_sq=smoke_l2_sq,
        abs_sum=abs_sum,
        max_abs=max_abs,
        total_values=total_values,
        shape_mismatches=tuple(shape_mismatches),
    )


def _find_best_scale_row(rows, best_scale):
    for row in rows:
        if row.get("scale") == best_scale:
            return row
    return rows[0] if rows else None


def _optional_float(value) -> float | None:
    return float(value) if value is not None else None


def _render_report(delta: CheckpointDelta, baseline: PeBaseline, decision: ScaleDecision) -> str:
    return "\n".join(
        [
            "# SigLIP Color Pilot Evaluation",
            "",
            f"**Decision:** `{decision.label}`",
            f"**Quality proven:** `{decision.quality_proven}`",
            "",
            "## Method",
            "",
            "This is a proxy evaluation, not a final visual quality pass. The pilot checkpoint",
            "is compared against the one-step SigLIP smoke checkpoint by tensor deltas, then",
            "anchored to the existing PE/no-IP contact-sheet baseline. A real SigLIP UI/API",
            "image workflow is still required before calling the model usable.",
            "",
            "## Checkpoint Delta",
            "",
            f"- Smoke: `{delta.smoke_checkpoint}`",
            f"- Pilot: `{delta.pilot_checkpoint}`",
            f"- Key match: `{delta.key_match}`",
            f"- Common tensors: `{delta.common_tensors}`",
            f"- Changed tensors: `{delta.changed_tensors}`",
            f"- Relative L2 delta: `{delta.relative_l2_delta:.10f}`",
            f"- Mean abs delta: `{delta.mean_abs_delta:.10f}`",
            f"- Max abs delta: `{delta.max_abs_delta:.10f}`",
            "",
            "## PE Baseline Anchor",
            "",
            f"- Summary: `{baseline.summary_path}`",
            f"- PE/no-IP contact-sheet pass: `{baseline.passed}`",
            f"- Best PE scale: `{baseline.best_scale}`",
            f"- Generated images: `{baseline.generated_count}`",
            f"- Best mean uplift: `{baseline.best_mean_uplift}`",
            f"- Best improved rate: `{baseline.best_improved_rate}`",
            "",
            "## Scale Decision",
            "",
            f"- Scale next: `{decision.scale_next}`",
            f"- Reason: {decision.reason}",
            "",
            "## Required Next Gate",
            "",
            "Create a native SigLIP ComfyUI/API workflow and generate contact sheets against",
            "no-IP and PE-Core baselines before any `quality` or `ready to use` claim.",
        ]
    ) + "\n"
