from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from training.siglip_shape_contrastive_cli import app
from training.siglip_shape_contrastive_smoke import _summary, _validate_weights
from training.siglip_smoke_types import CheckpointVerification, SmokeConfig, SmokeInputError


def test_summary_records_shape_fields_and_rows_loaded(tmp_path: Path) -> None:
    summary = _summary(
        _config(tmp_path),
        rows_loaded=10,
        losses=[1.0, 0.5],
        base_losses=[0.8, 0.4],
        contrastive_losses=[0.2, 0.1],
        shape_losses=[0.3, 0.2],
        explicit_negative_rows=10,
        trainable_parameters=12,
        frozen_params=34,
        checkpoint=CheckpointVerification("out.safetensors", True, True),
        contrastive_weight=0.25,
        contrastive_margin=0.08,
        shape_weight=0.2,
        reference_shape_weight=0.35,
    )

    assert summary.rows_loaded == 10
    assert summary.mean_shape_loss == pytest.approx(0.25)
    assert summary.shape_weight == 0.2
    assert summary.reference_shape_weight == 0.35
    assert summary.explicit_negative_rows == 10


def test_validate_weights_rejects_negative_shape_weight() -> None:
    with pytest.raises(SmokeInputError, match="shape_weight"):
        _validate_weights(-0.1, 0.0)


def test_shape_contrastive_cli_help_exposes_c094_knobs() -> None:
    result = CliRunner().invoke(app, ["--help"])

    assert result.exit_code == 0
    assert "--shape-weight" in result.output
    assert "--reference-shape" in result.output
    assert "--contrastive-weight" in result.output
    assert "--init-checkpoint-path" in result.output


def _config(tmp_path: Path) -> SmokeConfig:
    return SmokeConfig(
        manifest_path=tmp_path / "manifest.jsonl",
        image_root=tmp_path,
        output_path=tmp_path / "out.safetensors",
        dit_path=tmp_path / "dit.safetensors",
        text_encoder_path=tmp_path / "text.safetensors",
        vae_path=tmp_path / "vae.safetensors",
        pe_checkpoint_path=tmp_path / "pe.safetensors",
        siglip_model_id="local",
        device="cpu",
        steps=2,
        resolution=256,
        lr=1e-5,
        seed=1,
        max_rows=10,
        init_checkpoint_path=tmp_path / "init.safetensors",
    )
