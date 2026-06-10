from __future__ import annotations

import json
from pathlib import Path

import torch
from safetensors.torch import save_file

from tools.siglip_pilot_eval_core import (
    compare_checkpoints,
    decide_scale,
    evaluate_pilot,
)


def _write_tensors(path: Path, weight: float, bias: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    save_file(
        {
            "layer.weight": torch.tensor([[weight, 0.0], [0.0, weight]]),
            "layer.bias": torch.tensor([bias, bias]),
        },
        str(path),
        metadata={"format": "pt"},
    )


def test_checkpoint_delta_flags_changed_tensors_when_pilot_moves(tmp_path: Path) -> None:
    # Given: two loadable safetensor checkpoints with the same keys.
    smoke = tmp_path / "smoke.safetensors"
    pilot = tmp_path / "pilot.safetensors"
    _write_tensors(smoke, weight=1.0, bias=0.0)
    _write_tensors(pilot, weight=1.25, bias=0.5)

    # When: the checkpoint proxy metrics are computed.
    metrics = compare_checkpoints(smoke, pilot)

    # Then: every shared tensor is measured as changed with positive deltas.
    assert metrics.key_match is True
    assert metrics.common_tensors == 2
    assert metrics.changed_tensors == 2
    assert metrics.relative_l2_delta > 0.0
    assert metrics.max_abs_delta == 0.5


def test_decision_requires_visual_workflow_before_quality_claim() -> None:
    # Given: a pilot checkpoint moved, and the PE baseline already passed.
    decision = decide_scale(
        key_match=True,
        changed_tensors=4,
        relative_l2_delta=0.05,
        pe_baseline_pass=True,
        siglip_visual_eval_available=False,
    )

    # When/Then: the decision permits scaling but refuses to call quality proven.
    assert decision.scale_next is True
    assert decision.quality_proven is False
    assert decision.label == "scale_after_siglip_workflow_eval"


def test_decision_rejects_bare_visual_flag_as_quality_proof() -> None:
    # Given: a caller claims visual evaluation is available without scored artifacts.
    decision = decide_scale(
        key_match=True,
        changed_tensors=4,
        relative_l2_delta=0.05,
        pe_baseline_pass=True,
        siglip_visual_eval_available=True,
    )

    # When/Then: the proxy gate still refuses to set quality_proven.
    assert decision.scale_next is True
    assert decision.quality_proven is False
    assert "not accepted as quality proof" in decision.reason


def test_evaluate_pilot_writes_metrics_and_report(tmp_path: Path) -> None:
    # Given: tiny smoke/pilot checkpoints and a passing PE baseline summary.
    smoke = tmp_path / "smoke.safetensors"
    pilot = tmp_path / "pilot.safetensors"
    pe_summary = tmp_path / "pe_summary.json"
    out_dir = tmp_path / "eval"
    _write_tensors(smoke, weight=1.0, bias=0.0)
    _write_tensors(pilot, weight=1.1, bias=0.2)
    pe_summary.write_text(
        json.dumps(
            {
                "pass": True,
                "best_scale": 1.0,
                "generated_count": 40,
                "scale_summaries": [
                    {"scale": 1.0, "mean_uplift": 0.0937, "improved_rate": 0.875}
                ],
            }
        ),
        encoding="utf-8",
    )

    # When: the pilot evaluation CLI core writes artifacts.
    result = evaluate_pilot(
        smoke_checkpoint=smoke,
        pilot_checkpoint=pilot,
        pe_summary_path=pe_summary,
        out_dir=out_dir,
        siglip_visual_eval_available=False,
    )

    # Then: durable JSON and markdown artifacts describe the proxy decision.
    assert result.metrics_path.exists()
    assert result.report_path.exists()
    metrics = json.loads(result.metrics_path.read_text(encoding="utf-8"))
    report = result.report_path.read_text(encoding="utf-8")
    assert metrics["decision"]["label"] == "scale_after_siglip_workflow_eval"
    assert metrics["decision"]["quality_proven"] is False
    assert "proxy" in report.lower()
