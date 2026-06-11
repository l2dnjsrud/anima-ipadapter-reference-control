from __future__ import annotations

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


def test_checkpoint_builder_rejects_malformed_calibration_shape() -> None:
    state = dict(_tiny_calibrated_adapter().state_dict())
    state["feature_calibrator.deep_down.weight"] = torch.zeros(4, 7)

    with pytest.raises(SigLIPCheckpointError, match="calibration"):
        build_siglip_adapter_from_state(state)
