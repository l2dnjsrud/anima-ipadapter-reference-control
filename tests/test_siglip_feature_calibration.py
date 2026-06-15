from __future__ import annotations

from pathlib import Path

import pytest
import torch

from siglip_checkpoint import (
    SigLIPCheckpointError,
    build_siglip_adapter_from_state,
)
from siglip_feature_calibration import (
    CalibratedIPAdapterSigLIP,
    SigLIPFeatureCalibrator,
    wrap_siglip_with_calibrator,
)
from siglip_model import IPAdapterSigLIP, SigLIPFeatures
from training.siglip_real_smoke import load_trainable_adapter, save_adapter_checkpoint
from training.siglip_smoke_types import SmokeConfig


def _tiny_adapter() -> IPAdapterSigLIP:
    return IPAdapterSigLIP(
        siglip_dim=8,
        siglip_shallow_dim=8,
        dit_dim=16,
        num_blocks=2,
        num_queries=3,
        resampler_depth=1,
        resampler_heads=2,
        resampler_dim=16,
        resampler_dim_head=8,
        intermediate_dim=8,
        intermediate_layers=1,
        intermediate_heads=2,
        ip_heads=4,
        time_embed_dim=10,
        use_intermediate_encoder=True,
    )


def _tiny_calibrated_adapter() -> CalibratedIPAdapterSigLIP:
    return CalibratedIPAdapterSigLIP(
        siglip_dim=8,
        siglip_shallow_dim=8,
        dit_dim=16,
        num_blocks=2,
        num_queries=3,
        resampler_depth=1,
        resampler_heads=2,
        resampler_dim=16,
        resampler_dim_head=8,
        intermediate_dim=8,
        intermediate_layers=1,
        intermediate_heads=2,
        ip_heads=4,
        time_embed_dim=10,
        use_intermediate_encoder=True,
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
        siglip_model_id="google/siglip2-base-patch16-512",
        device="cpu",
        steps=1,
        resolution=128,
        lr=1e-5,
        seed=1,
        max_rows=1,
        init_checkpoint_path=init_checkpoint_path,
    )


def test_feature_calibrator_is_identity_at_initialization() -> None:
    features = SigLIPFeatures(
        deep=torch.randn(2, 5, 8),
        shallow=torch.randn(2, 7, 8),
    )
    calibrator = SigLIPFeatureCalibrator(
        deep_dim=8,
        shallow_dim=8,
        bottleneck_dim=4,
    )

    calibrated = calibrator(features)

    assert calibrated.deep.shape == features.deep.shape
    assert calibrated.shallow is not None
    assert calibrated.shallow.shape == features.shallow.shape
    assert torch.allclose(calibrated.deep, features.deep)
    assert torch.allclose(calibrated.shallow, features.shallow)


def test_calibrated_adapter_state_contains_feature_calibrator_keys() -> None:
    adapter = _tiny_calibrated_adapter()
    features = SigLIPFeatures(
        deep=torch.randn(1, 5, 8),
        shallow=torch.randn(1, 7, 8),
    )

    tokens = adapter.encode_ref(features, timestep=torch.tensor([0.5]))
    state = adapter.state_dict()

    assert tokens.shape == (1, 3, 16)
    assert "feature_calibrator.deep_down.weight" in state
    assert "feature_calibrator.shallow_up.weight" in state


def test_checkpoint_builder_round_trips_calibrated_state() -> None:
    state = _tiny_calibrated_adapter().state_dict()

    loaded = build_siglip_adapter_from_state(state)

    assert isinstance(loaded, CalibratedIPAdapterSigLIP)
    assert loaded.encode_ref(
        SigLIPFeatures(torch.randn(1, 5, 8), torch.randn(1, 7, 8)),
        timestep=torch.tensor([0.5]),
    ).shape == (1, 3, 16)


def test_checkpoint_builder_keeps_uncalibrated_state_type() -> None:
    loaded = build_siglip_adapter_from_state(_tiny_adapter().state_dict())

    assert type(loaded) is IPAdapterSigLIP


def test_wrap_siglip_with_calibrator_preserves_base_weights() -> None:
    base = _tiny_adapter()
    wrapped = wrap_siglip_with_calibrator(base, bottleneck_dim=4)

    assert isinstance(wrapped, CalibratedIPAdapterSigLIP)
    assert torch.allclose(wrapped.resampler.latents, base.resampler.latents)
    assert "feature_calibrator.deep_down.weight" in wrapped.state_dict()


def test_wrap_siglip_with_calibrator_keeps_existing_calibrator() -> None:
    calibrated = _tiny_calibrated_adapter()

    wrapped = wrap_siglip_with_calibrator(calibrated, bottleneck_dim=2)

    assert wrapped is calibrated


def test_trainable_loader_can_add_siglip_calibrator(tmp_path: Path) -> None:
    checkpoint = tmp_path / "base.safetensors"
    save_adapter_checkpoint(_tiny_adapter(), checkpoint)

    adapter = load_trainable_adapter(
        _smoke_config(tmp_path, init_checkpoint_path=checkpoint),
        torch.device("cpu"),
        torch.bfloat16,
        calibrator_bottleneck_dim=4,
    )

    assert isinstance(adapter, CalibratedIPAdapterSigLIP)
    assert adapter.training is True
    assert all(parameter.requires_grad for parameter in adapter.parameters())


def test_trainable_loader_can_freeze_siglip_except_calibrator(tmp_path: Path) -> None:
    checkpoint = tmp_path / "base.safetensors"
    save_adapter_checkpoint(_tiny_adapter(), checkpoint)

    adapter = load_trainable_adapter(
        _smoke_config(tmp_path, init_checkpoint_path=checkpoint),
        torch.device("cpu"),
        torch.bfloat16,
        calibrator_bottleneck_dim=4,
        train_calibrator_only=True,
    )

    trainable_names = {
        name for name, parameter in adapter.named_parameters() if parameter.requires_grad
    }
    assert trainable_names == {
        "feature_calibrator.deep_norm.weight",
        "feature_calibrator.deep_norm.bias",
        "feature_calibrator.deep_down.weight",
        "feature_calibrator.deep_up.weight",
        "feature_calibrator.shallow_norm.weight",
        "feature_calibrator.shallow_norm.bias",
        "feature_calibrator.shallow_down.weight",
        "feature_calibrator.shallow_up.weight",
    }


def test_trainable_loader_rejects_siglip_calibrator_only_without_calibrator(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "base.safetensors"
    save_adapter_checkpoint(_tiny_adapter(), checkpoint)

    with pytest.raises(ValueError, match="requires a calibrated SigLIP adapter"):
        load_trainable_adapter(
            _smoke_config(tmp_path, init_checkpoint_path=checkpoint),
            torch.device("cpu"),
            torch.bfloat16,
            train_calibrator_only=True,
        )


def test_checkpoint_builder_rejects_malformed_calibration_shape() -> None:
    state = dict(_tiny_calibrated_adapter().state_dict())
    state["feature_calibrator.deep_down.weight"] = torch.zeros(4, 7)

    with pytest.raises(SigLIPCheckpointError, match="calibration"):
        build_siglip_adapter_from_state(state)
