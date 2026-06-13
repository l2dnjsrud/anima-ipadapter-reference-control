from __future__ import annotations

import json
from pathlib import Path

from tools.c095_feature_bridge_training_report import write_c095_training_report


def test_c095_training_report_accepts_bridge_only_summary(tmp_path: Path) -> None:
    stdout = tmp_path / "train_stdout.txt"
    stdout.write_text(json.dumps(_summary()) + "\n", encoding="utf-8")

    report = write_c095_training_report(stdout, tmp_path / "out")

    assert report["decision"] == "proceed_to_c095_generation_gate"
    assert report["failures"] == []
    assert report["training_summary"]["heldout07_rows"] == []
    assert (tmp_path / "out" / "report.md").is_file()


def test_c095_training_report_rejects_non_bridge_trainable(tmp_path: Path) -> None:
    stdout = tmp_path / "train_stdout.txt"
    payload = _summary()
    payload["trainable_parameter_names"] = ["resampler.proj_in.weight"]
    stdout.write_text(json.dumps(payload) + "\n", encoding="utf-8")

    report = write_c095_training_report(stdout, tmp_path / "out")

    assert report["decision"] == "c095_training_gate_failed"
    assert "non-bridge parameters trainable" in report["failures"]


def _summary() -> dict[str, object]:
    return {
        "steps": 96,
        "rows_loaded": 10,
        "first_loss": 0.3,
        "final_loss": 0.2,
        "mean_loss": 0.25,
        "mean_base_loss": 0.2,
        "mean_contrastive_loss": 0.1,
        "mean_shape_loss": 0.05,
        "finite_loss": True,
        "explicit_negative_rows": 10,
        "trainable_parameters": 114816,
        "trainable_parameter_names": [
            "feature_bridge.norm.weight",
            "feature_bridge.norm.bias",
            "feature_bridge.down.weight",
            "feature_bridge.up.weight",
        ],
        "frozen_base_parameters": 1,
        "checkpoint": {
            "output_path": "checkpoints/anima_siglip_ip_adapter_c095_feature_bridge_b128_0096_20260613.safetensors",
            "loadable": True,
            "pe_checkpoint_rejected": True,
        },
        "init_checkpoint_path": "checkpoints/anima_siglip_ip_adapter_c094_shape_supervised_0064_20260613.safetensors",
        "contrastive_weight": 0.25,
        "contrastive_margin": 0.08,
        "shape_weight": 0.2,
        "reference_shape_weight": 0.35,
        "feature_bridge_bottleneck_dim": 128,
        "train_feature_bridge_only": True,
    }
