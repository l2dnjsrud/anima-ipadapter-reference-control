from __future__ import annotations

from pathlib import Path

import pytest
import torch

from qwenvl_checkpoint import QwenVLCheckpointError, build_qwenvl_adapter_from_state
from qwenvl_feature_calibration import (
    CalibratedIPAdapterQwenVL,
    QwenVLFeatureCalibrator,
    wrap_qwenvl_with_calibrator,
)
from qwenvl_model import IPAdapterQwenVL
from training.qwenvl_smoke_checkpoint import (
    load_trainable_qwenvl_adapter,
    save_qwenvl_adapter_checkpoint,
)
from training.siglip_smoke_types import SmokeConfig


def test_qwenvl_feature_calibrator_is_identity_at_initialization() -> None:
    embedding = torch.randn(2, 12)
    calibrator = QwenVLFeatureCalibrator(embedding_dim=12, bottleneck_dim=4)

    calibrated = calibrator(embedding)

    assert calibrated.shape == embedding.shape
    assert torch.allclose(calibrated, embedding)


def test_calibrated_qwenvl_adapter_state_round_trips() -> None:
    adapter = _tiny_calibrated_qwenvl_adapter()

    loaded = build_qwenvl_adapter_from_state(adapter.state_dict())

    assert isinstance(loaded, CalibratedIPAdapterQwenVL)
    assert loaded.encode_ref(torch.randn(1, 12), timestep=torch.tensor([0.5])).shape == (
        1,
        3,
        16,
    )


def test_qwenvl_checkpoint_rejects_malformed_calibration_shape() -> None:
    state = dict(_tiny_calibrated_qwenvl_adapter().state_dict())
    state["feature_calibrator.down.weight"] = torch.zeros(4, 11)

    with pytest.raises(QwenVLCheckpointError, match="calibration"):
        build_qwenvl_adapter_from_state(state)


def test_wrap_qwenvl_with_calibrator_preserves_base_weights() -> None:
    base = _tiny_qwenvl_adapter()

    wrapped = wrap_qwenvl_with_calibrator(base, bottleneck_dim=4)

    assert isinstance(wrapped, CalibratedIPAdapterQwenVL)
    assert torch.allclose(wrapped.resampler.latents, base.resampler.latents)
    assert "feature_calibrator.down.weight" in wrapped.state_dict()


def test_trainable_loader_can_add_qwenvl_calibrator(tmp_path: Path) -> None:
    checkpoint = tmp_path / "base.safetensors"
    save_qwenvl_adapter_checkpoint(_tiny_qwenvl_adapter(), checkpoint)

    adapter = load_trainable_qwenvl_adapter(
        _smoke_config(tmp_path, init_checkpoint_path=checkpoint),
        torch.device("cpu"),
        calibrator_bottleneck_dim=4,
    )

    assert isinstance(adapter, CalibratedIPAdapterQwenVL)
    assert adapter.training is True
    assert all(parameter.requires_grad for parameter in adapter.parameters())


def _tiny_qwenvl_adapter() -> IPAdapterQwenVL:
    return IPAdapterQwenVL(
        embedding_dim=12,
        dit_dim=16,
        num_blocks=2,
        num_queries=3,
        resampler_depth=1,
        resampler_heads=2,
        resampler_dim=16,
        resampler_dim_head=8,
        ip_heads=4,
        time_embed_dim=10,
    )


def _tiny_calibrated_qwenvl_adapter() -> CalibratedIPAdapterQwenVL:
    return CalibratedIPAdapterQwenVL(
        embedding_dim=12,
        dit_dim=16,
        num_blocks=2,
        num_queries=3,
        resampler_depth=1,
        resampler_heads=2,
        resampler_dim=16,
        resampler_dim_head=8,
        ip_heads=4,
        time_embed_dim=10,
        calibrator_bottleneck_dim=4,
    )


def _smoke_config(tmp_path: Path, *, init_checkpoint_path: Path) -> SmokeConfig:
    return SmokeConfig(
        manifest_path=tmp_path / "manifest.jsonl",
        image_root=tmp_path,
        output_path=tmp_path / "out.safetensors",
        dit_path=tmp_path / "dit.safetensors",
        text_encoder_path=tmp_path / "text.safetensors",
        vae_path=tmp_path / "vae.safetensors",
        pe_checkpoint_path=tmp_path / "pe.safetensors",
        siglip_model_id="Qwen/Qwen3-VL-Embedding-2B",
        device="cpu",
        steps=1,
        resolution=128,
        lr=1e-5,
        seed=1,
        max_rows=1,
        init_checkpoint_path=init_checkpoint_path,
    )
