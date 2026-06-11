from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from native_siglip import (
    AnimaSigLIPIPAdapterApply,
    AnimaSigLIPIPAdapterLoader,
    SIGLIP_NODE_CLASS_MAPPINGS,
    SigLIPFeatures,
)
from siglip_checkpoint import (
    SigLIPCheckpointError,
    build_siglip_adapter_from_state,
    detect_siglip_checkpoint,
)
from siglip_model import IPAdapterSigLIP


ROOT = Path(__file__).resolve().parents[1]
PE_CHECKPOINT = ROOT / "checkpoints" / "anima_ip_adapter_quality_20260610.safetensors"


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


def test_siglip_loader_rejects_pe_core_selected_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "native_siglip._model_path", lambda _folder, _name: PE_CHECKPOINT
    )

    with pytest.raises(SigLIPCheckpointError, match="PE-Core"):
        AnimaSigLIPIPAdapterLoader().load(
            "anima_ip_adapter_quality_20260610.safetensors"
        )


def test_siglip_apply_wraps_sampling_with_runtime_patch() -> None:
    class FakeCrossAttention:
        def forward(
            self,
            x: torch.Tensor,
            attn_params: object | None = None,
            context: torch.Tensor | None = None,
            rope_cos_sin: object | None = None,
            **_kwargs: object,
        ) -> torch.Tensor:
            del attn_params, context, rope_cos_sin
            return x

    class FakeDIT:
        prepare_embedded_sequence = object()
        unpatchify = object()
        patch_spatial = 2
        patch_temporal = 1

        def __init__(self) -> None:
            self.blocks = [
                SimpleNamespace(cross_attn=FakeCrossAttention()) for _ in range(2)
            ]

    class FakeModelPatcher:
        def __init__(self, dit: FakeDIT) -> None:
            self.model = SimpleNamespace(diffusion_model=dit)
            self.model_options: dict[str, object] = {}
            self.wrapper = None

        def clone(self) -> FakeModelPatcher:
            clone = FakeModelPatcher(self.model.diffusion_model)
            clone.model_options = self.model_options.copy()
            return clone

        def get_model_object(self, name: str):
            assert name == "model_sampling"
            return SimpleNamespace(percent_to_sigma=lambda percent: 1.0 - percent)

        def set_model_unet_function_wrapper(self, wrapper) -> None:
            self.wrapper = wrapper

    torch.manual_seed(0)
    adapter = _tiny_adapter()
    for scale in adapter.ip_scales:
        scale.data.fill_(1.0)
    features = SigLIPFeatures(torch.randn(1, 5, 8), torch.randn(1, 7, 8))
    dit = FakeDIT()
    source = FakeModelPatcher(dit)

    (patched,) = AnimaSigLIPIPAdapterApply().apply(
        source, adapter, features, 1.25, 0.0, 1.0
    )

    assert patched is not source
    assert patched.wrapper is not None

    input_x = torch.randn(1, 4, 16)

    def apply_model(
        model_input: torch.Tensor,
        timestep: torch.Tensor,
        **_kwargs: object,
    ) -> torch.Tensor:
        del timestep
        output = model_input
        for block in dit.blocks:
            output = block.cross_attn.forward(output, object(), output, None)
        return output

    baseline = apply_model(input_x, torch.tensor([0.5]))
    output = patched.wrapper(
        apply_model,
        {"input": input_x, "timestep": torch.tensor([0.5]), "c": {}},
    )
    restored = apply_model(input_x, torch.tensor([0.5]))

    assert output.shape == baseline.shape
    assert not torch.allclose(output, baseline)
    assert torch.allclose(restored, baseline)


def test_siglip_apply_skips_when_weight_is_zero() -> None:
    class FakeDIT:
        prepare_embedded_sequence = object()
        unpatchify = object()
        patch_spatial = 2
        patch_temporal = 1
        blocks = []

    class FakeModelPatcher:
        def __init__(self) -> None:
            self.model = SimpleNamespace(diffusion_model=FakeDIT())
            self.model_options: dict[str, object] = {}
            self.wrapper = None

        def clone(self) -> FakeModelPatcher:
            clone = FakeModelPatcher()
            clone.model_options = self.model_options.copy()
            return clone

        def get_model_object(self, name: str):
            assert name == "model_sampling"
            return SimpleNamespace(percent_to_sigma=lambda percent: 1.0 - percent)

        def set_model_unet_function_wrapper(self, wrapper) -> None:
            self.wrapper = wrapper

    adapter = _tiny_adapter()
    features = SigLIPFeatures(torch.randn(1, 5, 8), torch.randn(1, 7, 8))
    (patched,) = AnimaSigLIPIPAdapterApply().apply(
        FakeModelPatcher(), adapter, features, 0.0, 0.0, 1.0
    )

    def apply_model(
        model_input: torch.Tensor, timestep: torch.Tensor, **_kwargs: object
    ) -> torch.Tensor:
        del timestep
        return model_input + 1

    output = patched.wrapper(
        apply_model,
        {"input": torch.zeros(1, 4, 16), "timestep": torch.tensor([0.5]), "c": {}},
    )

    assert torch.equal(output, torch.ones(1, 4, 16))


def test_siglip_nodes_and_training_doc_exist() -> None:
    assert {
        "AnimaSigLIPIPAdapterLoader",
        "AnimaSigLIPIPAdapterApply",
        "AnimaSigLIPEncodeImage",
    } <= set(SIGLIP_NODE_CLASS_MAPPINGS)
    training_doc = ROOT / "docs" / "siglip_training.md"
    assert training_doc.exists()
    text = training_doc.read_text(encoding="utf-8")
    assert "Wenaka/anima-ip-adapter-dataset" in text
