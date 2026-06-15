from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from native_qwenvl import (
    AnimaQwenVLIPAdapterApply,
    AnimaQwenVLIPAdapterLoader,
    AnimaQwenVLEncodeImage,
    QWENVL_NODE_CLASS_MAPPINGS,
)
from qwenvl_checkpoint import QwenVLCheckpointError
from qwenvl_model import IPAdapterQwenVL


ROOT = Path(__file__).resolve().parents[1]
PE_CHECKPOINT = ROOT / "checkpoints" / "anima_ip_adapter_quality_20260610.safetensors"


def test_qwenvl_loader_uses_ipadapter_model_selector() -> None:
    inputs = AnimaQwenVLIPAdapterLoader.INPUT_TYPES()["required"]

    assert "ipadapter_name" in inputs
    assert "ipadapter_path" not in inputs


def test_qwenvl_loader_rejects_pe_core_selected_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "native_qwenvl._model_path", lambda _folder, _name: PE_CHECKPOINT
    )

    with pytest.raises(QwenVLCheckpointError, match="PE-Core"):
        AnimaQwenVLIPAdapterLoader().load(
            "anima_ip_adapter_quality_20260610.safetensors"
        )


def test_qwenvl_encode_image_returns_embedding_tensor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeSentenceModel:
        def encode(
            self,
            inputs: list[dict[str, object]],
            *,
            normalize_embeddings: bool,
            convert_to_tensor: bool,
            prompt: str,
        ) -> torch.Tensor:
            assert len(inputs) == 2
            assert normalize_embeddings is True
            assert convert_to_tensor is True
            assert "manhwa" in prompt
            return torch.ones(2, 2048)

    monkeypatch.setattr(
        AnimaQwenVLEncodeImage,
        "_embedding_stack",
        lambda self, _model_id: FakeSentenceModel(),
    )
    image = torch.zeros(2, 8, 8, 3)

    (embedding,) = AnimaQwenVLEncodeImage().encode(
        image,
        "Qwen/Qwen3-VL-Embedding-2B",
        "Represent this manhwa reference image.",
    )

    assert embedding.shape == (2, 2048)


def test_qwenvl_apply_wraps_sampling_with_runtime_patch() -> None:
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
    adapter = _tiny_qwenvl_adapter()
    for scale in adapter.ip_scales:
        scale.data.fill_(1.0)
    embedding = torch.randn(1, 12)
    dit = FakeDIT()
    source = FakeModelPatcher(dit)

    (patched,) = AnimaQwenVLIPAdapterApply().apply(
        source, adapter, embedding, 1.25, 0.0, 1.0
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


def test_qwenvl_nodes_are_registered() -> None:
    assert {
        "AnimaQwenVLIPAdapterLoader",
        "AnimaQwenVLEncodeImage",
        "AnimaQwenVLIPAdapterApply",
    } <= set(QWENVL_NODE_CLASS_MAPPINGS)


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
