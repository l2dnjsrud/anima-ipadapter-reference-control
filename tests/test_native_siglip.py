from __future__ import annotations

from pathlib import Path

import pytest
import torch

from native_siglip import (
    AnimaSigLIPIPAdapterLoader,
    AnimaSigLIPIPAdapterApply,
    SIGLIP_NODE_CLASS_MAPPINGS,
    SigLIPFeatures,
    SigLIPKVAttn2Patch,
)
from siglip_checkpoint import (
    SigLIPCheckpointError,
    build_siglip_adapter_from_state,
    detect_siglip_checkpoint,
)
from siglip_model import IPAdapterSigLIP


ROOT = Path(__file__).resolve().parents[1]
PE_CHECKPOINT = ROOT / "checkpoints" / "anima_ip_adapter_quality_20260610.safetensors"


class FakeModelPatcher:
    def __init__(self) -> None:
        self.patches: list[torch.nn.Module] = []

    def clone(self) -> FakeModelPatcher:
        return FakeModelPatcher()

    def set_model_attn2_patch(self, patch: torch.nn.Module) -> None:
        self.patches.append(patch)


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


def test_siglip_adapter_shapes_when_shallow_and_deep_features_are_present() -> None:
    adapter = _tiny_adapter()
    features = SigLIPFeatures(deep=torch.randn(2, 5, 8), shallow=torch.randn(2, 7, 8))
    timestep = torch.tensor([0.25, 0.75])

    image_tokens = adapter.encode_ref(features, timestep=timestep)
    out = adapter.forward_block(1, torch.randn(2, 4, 16), image_tokens)

    assert image_tokens.shape == (2, 3, 16)
    assert out.shape == (2, 4, 16)


def test_siglip_checkpoint_detection_round_trips_tiny_state() -> None:
    state = _tiny_adapter().state_dict()

    spec = detect_siglip_checkpoint(state)
    loaded = build_siglip_adapter_from_state(state)

    assert spec.num_blocks == 2
    assert spec.num_queries == 3
    assert spec.use_intermediate_encoder is True
    assert loaded.encode_ref(
        SigLIPFeatures(torch.randn(1, 5, 8), torch.randn(1, 7, 8)),
        timestep=torch.tensor([0.5]),
    ).shape == (1, 3, 16)


def test_siglip_checkpoint_detection_rejects_malformed_state() -> None:
    state = {"resampler.time_proj.weight": torch.empty(10, 1)}

    with pytest.raises(SigLIPCheckpointError, match="missing"):
        detect_siglip_checkpoint(state)


def test_siglip_checkpoint_detection_rejects_pe_core_checkpoint() -> None:
    safetensors = pytest.importorskip("safetensors.torch")
    state = safetensors.load_file(str(PE_CHECKPOINT), device="cpu")

    with pytest.raises(SigLIPCheckpointError, match="PE-Core"):
        detect_siglip_checkpoint(state)


def test_siglip_loader_uses_ipadapter_model_selector() -> None:
    inputs = AnimaSigLIPIPAdapterLoader.INPUT_TYPES()["required"]

    assert "ipadapter_name" in inputs
    assert "ipadapter_path" not in inputs


def test_siglip_loader_rejects_pe_core_selected_model(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("native_siglip._model_path", lambda _folder, _name: PE_CHECKPOINT)

    with pytest.raises(SigLIPCheckpointError, match="PE-Core"):
        AnimaSigLIPIPAdapterLoader().load("anima_ip_adapter_quality_20260610.safetensors")


def test_siglip_apply_uses_comfy_model_patch_semantics() -> None:
    adapter = _tiny_adapter()
    features = SigLIPFeatures(torch.randn(1, 5, 8), torch.randn(1, 7, 8))
    source = FakeModelPatcher()

    (patched,) = AnimaSigLIPIPAdapterApply().apply(source, adapter, features, 1.25, 0.0, 1.0)

    assert patched is not source
    assert len(patched.patches) == 1
    assert isinstance(patched.patches[0], SigLIPKVAttn2Patch)


def test_siglip_attn2_patch_appends_ip_key_value_tokens() -> None:
    adapter = _tiny_adapter()
    features = SigLIPFeatures(torch.randn(1, 5, 8), torch.randn(1, 7, 8))
    patch = SigLIPKVAttn2Patch(adapter, features, weight=1.0, start_at=0.0, end_at=1.0)

    q, k, v = patch(
        torch.randn(1, 4, 16),
        torch.randn(1, 6, 16),
        torch.randn(1, 6, 16),
        {"block": ("double", 1), "timestep": torch.tensor([0.5])},
    )

    assert q.shape == (1, 4, 16)
    assert k.shape == (1, 9, 16)
    assert v.shape == (1, 9, 16)


def test_siglip_nodes_and_training_doc_exist() -> None:
    assert {
        "AnimaSigLIPIPAdapterLoader",
        "AnimaSigLIPEncodeImage",
        "AnimaSigLIPIPAdapterApply",
    } <= set(SIGLIP_NODE_CLASS_MAPPINGS)
    training_doc = ROOT / "docs" / "siglip_training.md"
    assert training_doc.exists()
    text = training_doc.read_text(encoding="utf-8")
    assert "Wenaka/anima-ip-adapter-dataset" in text
