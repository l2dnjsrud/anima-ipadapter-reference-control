from __future__ import annotations

from pathlib import Path

import pytest
import torch

from siglip_checkpoint import SigLIPCheckpointError, build_siglip_adapter_from_state
from siglip_feature_bridge import (
    BridgedIPAdapterSigLIP,
    SigLIPFeatureBridge,
    wrap_siglip_with_feature_bridge,
)
from siglip_model import IPAdapterSigLIP, SigLIPFeatures
from training.siglip_real_smoke import load_trainable_adapter, save_adapter_checkpoint
from training.siglip_smoke_types import SmokeConfig


def test_feature_bridge_is_identity_at_initialization() -> None:
    tokens = torch.randn(2, 9, 8)
    bridge = SigLIPFeatureBridge(token_dim=8, bottleneck_dim=4)

    bridged = bridge(tokens)

    assert bridged.shape == tokens.shape
    assert torch.allclose(bridged, tokens)


def test_bridged_adapter_state_contains_feature_bridge_keys() -> None:
    adapter = _tiny_bridged_adapter()
    features = SigLIPFeatures(torch.randn(1, 5, 8), torch.randn(1, 7, 8))

    tokens = adapter.encode_ref(features, timestep=torch.tensor([0.5]))
    state = adapter.state_dict()

    assert tokens.shape == (1, 3, 16)
    assert "feature_bridge.down.weight" in state
    assert "feature_bridge.up.weight" in state


def test_checkpoint_builder_round_trips_bridged_state() -> None:
    state = _tiny_bridged_adapter().state_dict()

    loaded = build_siglip_adapter_from_state(state)

    assert isinstance(loaded, BridgedIPAdapterSigLIP)
    assert loaded.encode_ref(
        SigLIPFeatures(torch.randn(1, 5, 8), torch.randn(1, 7, 8)),
        timestep=torch.tensor([0.5]),
    ).shape == (1, 3, 16)


def test_wrap_siglip_with_feature_bridge_preserves_base_weights() -> None:
    base = _tiny_adapter()

    wrapped = wrap_siglip_with_feature_bridge(base, bottleneck_dim=4)

    assert isinstance(wrapped, BridgedIPAdapterSigLIP)
    assert torch.allclose(wrapped.resampler.latents, base.resampler.latents)
    assert "feature_bridge.down.weight" in wrapped.state_dict()


def test_trainable_loader_can_freeze_siglip_except_feature_bridge(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "base.safetensors"
    save_adapter_checkpoint(_tiny_adapter(), checkpoint)

    adapter = load_trainable_adapter(
        _smoke_config(tmp_path, init_checkpoint_path=checkpoint),
        torch.device("cpu"),
        torch.bfloat16,
        feature_bridge_bottleneck_dim=4,
        train_feature_bridge_only=True,
    )

    trainable_names = {
        name for name, parameter in adapter.named_parameters() if parameter.requires_grad
    }
    assert trainable_names == {
        "feature_bridge.norm.weight",
        "feature_bridge.norm.bias",
        "feature_bridge.down.weight",
        "feature_bridge.up.weight",
    }


def test_checkpoint_builder_rejects_malformed_feature_bridge_shape() -> None:
    state = dict(_tiny_bridged_adapter().state_dict())
    state["feature_bridge.down.weight"] = torch.zeros(4, 7)

    with pytest.raises(SigLIPCheckpointError, match="feature bridge"):
        build_siglip_adapter_from_state(state)


def test_checkpoint_builder_rejects_combined_calibrator_and_bridge() -> None:
    state = dict(_tiny_bridged_adapter().state_dict())
    state.update(
        {
            "feature_calibrator.deep_norm.weight": torch.ones(8),
            "feature_calibrator.deep_norm.bias": torch.zeros(8),
            "feature_calibrator.deep_down.weight": torch.zeros(4, 8),
            "feature_calibrator.deep_up.weight": torch.zeros(8, 4),
            "feature_calibrator.shallow_norm.weight": torch.ones(8),
            "feature_calibrator.shallow_norm.bias": torch.zeros(8),
            "feature_calibrator.shallow_down.weight": torch.zeros(4, 8),
            "feature_calibrator.shallow_up.weight": torch.zeros(8, 4),
        }
    )

    with pytest.raises(SigLIPCheckpointError, match="cannot be combined"):
        build_siglip_adapter_from_state(state)


def test_wrap_siglip_with_feature_bridge_rejects_no_intermediate_route() -> None:
    adapter = IPAdapterSigLIP(
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
        use_intermediate_encoder=False,
    )

    with pytest.raises(RuntimeError, match="CrossLayerEncoder"):
        wrap_siglip_with_feature_bridge(adapter, bottleneck_dim=4)


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


def _tiny_bridged_adapter() -> BridgedIPAdapterSigLIP:
    return BridgedIPAdapterSigLIP(
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
        feature_bridge_bottleneck_dim=4,
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
