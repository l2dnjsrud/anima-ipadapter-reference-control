from __future__ import annotations

import json
from pathlib import Path

from tools.c094_shape_training_report import write_c094_training_report


def test_write_c094_training_report_accepts_valid_summary(tmp_path: Path) -> None:
    stdout = _write_stdout(tmp_path, _valid_summary())

    result = write_c094_training_report(stdout, tmp_path)

    assert result["decision"] == "proceed_to_c094_generation_gate"
    assert (tmp_path / "training_summary.json").is_file()
    assert (tmp_path / "summary.json").is_file()
    assert "proceed_to_c094_generation_gate" in (tmp_path / "report.md").read_text()


def test_write_c094_training_report_accepts_multiline_console_json(tmp_path: Path) -> None:
    stdout = tmp_path / "train_stdout.txt"
    stdout.write_text(
        "loading...\n" + json.dumps(_valid_summary(), indent=2) + "\n",
        encoding="utf-8",
    )

    result = write_c094_training_report(stdout, tmp_path)

    assert result["decision"] == "proceed_to_c094_generation_gate"


def test_write_c094_training_report_rejects_wrong_init_checkpoint(tmp_path: Path) -> None:
    payload = _valid_summary()
    payload["init_checkpoint_path"] = "checkpoints/anima_siglip_ip_adapter_c092_qwen_target_0064_20260613.safetensors"
    stdout = _write_stdout(tmp_path, payload)

    result = write_c094_training_report(stdout, tmp_path)

    assert result["decision"] == "c094_training_gate_failed"
    assert "wrong init checkpoint" in result["failures"]


def test_write_c094_training_report_rejects_missing_shape_loss(tmp_path: Path) -> None:
    payload = _valid_summary()
    payload["shape_weight"] = 0.0
    payload["mean_shape_loss"] = 0.0
    stdout = _write_stdout(tmp_path, payload)

    result = write_c094_training_report(stdout, tmp_path)

    assert result["decision"] == "c094_training_gate_failed"
    assert "shape supervision disabled" in result["failures"]
    assert "mean shape loss not positive" in result["failures"]


def _write_stdout(tmp_path: Path, payload: dict[str, object]) -> Path:
    path = tmp_path / "train_stdout.txt"
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    return path


def _valid_summary() -> dict[str, object]:
    return {
        "steps": 64,
        "rows_loaded": 10,
        "first_loss": 1.0,
        "final_loss": 0.8,
        "mean_loss": 0.9,
        "mean_base_loss": 0.7,
        "mean_contrastive_loss": 0.1,
        "mean_shape_loss": 0.2,
        "finite_loss": True,
        "explicit_negative_rows": 10,
        "trainable_parameters": 1,
        "frozen_base_parameters": 2,
        "checkpoint": {
            "output_path": "checkpoints/anima_siglip_ip_adapter_c094_shape_supervised_0064_20260613.safetensors",
            "loadable": True,
            "pe_checkpoint_rejected": True,
        },
        "init_checkpoint_path": "checkpoints/anima_siglip_ip_adapter_c093_qwen_target_anti_collapse_0048_20260613.safetensors",
        "contrastive_weight": 0.25,
        "contrastive_margin": 0.08,
        "shape_weight": 0.2,
        "reference_shape_weight": 0.35,
    }
