from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file

from native_pe import (
    AnimaPEFeatures,
    AnimaPEIPAdapterSpec,
    AnimaPEIPAdapterApply,
    AnimaPEIPAdapterLoader,
    find_anima_diffusion_model,
    load_pe_adapter_spec,
)


PE_METADATA = {
    "ss_network_spec": "ip_adapter",
    "ss_encoder": "pe",
    "ss_encoder_dim": "1024",
    "ss_context_dim": "1024",
    "ss_num_ip_tokens": "16",
    "ss_num_blocks": "28",
    "ss_hidden_size": "2048",
    "ss_num_heads": "16",
}


def _checkpoint(path: Path, metadata: dict[str, str]) -> Path:
    save_file({"ip_centroid": torch.zeros(1024)}, str(path), metadata=metadata)
    return path


def test_pe_loader_exposes_ipadapter_model_selector() -> None:
    inputs = AnimaPEIPAdapterLoader.INPUT_TYPES()["required"]

    assert "ipadapter_name" in inputs
    assert "anima_ip_adapter_quality_20260610.safetensors" in inputs["ipadapter_name"][0]


def test_pe_checkpoint_metadata_is_accepted(tmp_path: Path) -> None:
    path = _checkpoint(tmp_path / "adapter.safetensors", PE_METADATA)

    spec = load_pe_adapter_spec(path)

    assert spec.path == path
    assert spec.metadata["ss_encoder"] == "pe"


def test_siglip_or_malformed_checkpoint_is_rejected(tmp_path: Path) -> None:
    path = _checkpoint(
        tmp_path / "siglip.safetensors",
        {
            **PE_METADATA,
            "ss_encoder": "siglip2",
        },
    )

    with pytest.raises(ValueError, match="PE-Core Anima IP-Adapter"):
        load_pe_adapter_spec(path)


def test_non_safetensors_checkpoint_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "adapter.bin"
    path.write_bytes(b"not a safetensors file")

    with pytest.raises(ValueError, match=".safetensors"):
        load_pe_adapter_spec(path)


def test_find_anima_diffusion_model_walks_common_wrappers() -> None:
    class DIT:
        blocks = []
        prepare_embedded_sequence = object()
        unpatchify = object()
        patch_spatial = 2
        patch_temporal = 1

    class Inner:
        diffusion_model = DIT()

    class Patcher:
        model = Inner()

    assert find_anima_diffusion_model(Patcher()) is Patcher.model.diffusion_model


def test_apply_rejects_feature_dimension_before_loading_models(tmp_path: Path) -> None:
    path = _checkpoint(tmp_path / "adapter.safetensors", PE_METADATA)
    spec = load_pe_adapter_spec(path)
    features = AnimaPEFeatures(
        features=torch.zeros(1, 4, 512),
        encoder_name="pe",
        source_size=(64, 64),
    )

    with pytest.raises(ValueError, match=r"\[B,T,1024\]"):
        AnimaPEIPAdapterApply().apply(
            model=object(),
            adapter=spec,
            features=features,
            strength=1.0,
            start_percent=0.0,
            end_percent=1.0,
            preserve_wrapper=True,
        )


def test_apply_uses_comfy_attention_signature(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeCrossAttention:
        is_selfattn = False
        context_dim = 4
        n_heads = 2
        head_dim = 2

        def __init__(self) -> None:
            self.output_proj = torch.nn.Identity()
            self.output_dropout = torch.nn.Identity()
            self.forward = self.original_forward

        def original_forward(
            self,
            x: torch.Tensor,
            context: torch.Tensor | None = None,
            rope_emb: torch.Tensor | None = None,
            transformer_options: dict | None = None,
        ) -> torch.Tensor:
            q, k, v = self.compute_qkv(x, context, rope_emb=rope_emb)
            return self.output_dropout(self.output_proj(self.attn_op(q, k, v, transformer_options=transformer_options)))

        def compute_qkv(
            self,
            x: torch.Tensor,
            context: torch.Tensor | None = None,
            rope_emb: torch.Tensor | None = None,
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            del rope_emb
            source = x if context is None else context
            return (
                x.reshape(x.shape[0], x.shape[1], self.n_heads, self.head_dim),
                source.reshape(source.shape[0], source.shape[1], self.n_heads, self.head_dim),
                source.reshape(source.shape[0], source.shape[1], self.n_heads, self.head_dim),
            )

        def attn_op(
            self,
            q: torch.Tensor,
            k: torch.Tensor,
            v: torch.Tensor,
            transformer_options: dict | None = None,
        ) -> torch.Tensor:
            del k, v, transformer_options
            return q.reshape(q.shape[0], q.shape[1], -1)

    class FakeDIT:
        prepare_embedded_sequence = object()
        unpatchify = object()
        patch_spatial = 2
        patch_temporal = 1

        def __init__(self) -> None:
            self.blocks = [SimpleNamespace(cross_attn=FakeCrossAttention())]

    class FakeNetwork:
        num_blocks = 1
        context_dim = 4
        num_heads = 2
        head_dim = 2
        ip_scale = 1.0
        multiplier = 1.0

        def __init__(self, dit: FakeDIT) -> None:
            self.dit = dit
            self._cross_attn_modules = []
            self._original_forwards = []
            self._patched = False
            self._diag_enabled = False
            self.ip_gate = torch.nn.Parameter(torch.ones(1))

        def to(self, device: torch.device, dtype: torch.dtype) -> "FakeNetwork":
            self.device = device
            self.dtype = dtype
            return self

        def apply_to(self, *_args, **_kwargs) -> None:
            cross_attn = self.dit.blocks[0].cross_attn
            self._cross_attn_modules.append(cross_attn)
            self._original_forwards.append(cross_attn.forward)

            def training_forward(x, attn_params, context, rope_cos_sin=None):
                del attn_params, context, rope_cos_sin
                return x

            cross_attn.forward = training_forward
            self._patched = True

        def encode_ip_tokens(self, image_features: torch.Tensor) -> torch.Tensor:
            return image_features

        def set_ip_tokens(self, ip_tokens: torch.Tensor | None) -> None:
            if ip_tokens is None:
                return
            for cross_attn in self._cross_attn_modules:
                cross_attn._ip_k_cached = torch.zeros(1, self.num_heads, 2, self.head_dim)
                cross_attn._ip_v_cached = torch.ones(1, self.num_heads, 2, self.head_dim)
                cross_attn._ip_gate_scale_cached = torch.tensor(1.0)
                cross_attn._ip_diag_ratio_sum = None
                cross_attn._ip_diag_count = None

        def clear_ip_tokens(self) -> None:
            self.set_ip_tokens(None)

        def remove_from(self) -> None:
            for cross_attn, original in zip(self._cross_attn_modules, self._original_forwards):
                cross_attn.forward = original
            self._cross_attn_modules.clear()
            self._original_forwards.clear()
            self._patched = False

    class FakeModel:
        def __init__(self) -> None:
            self.model_options = {}
            self.wrapper = None

        def get_model_object(self, name: str):
            assert name == "model_sampling"
            return SimpleNamespace(percent_to_sigma=lambda percent: 1.0 - percent)

        def clone(self) -> "FakeModel":
            return self

        def set_model_unet_function_wrapper(self, wrapper) -> None:
            self.wrapper = wrapper

    dit = FakeDIT()
    network = FakeNetwork(dit)
    monkeypatch.setattr("native_pe.find_anima_diffusion_model", lambda _model: dit)
    monkeypatch.setattr("native_pe._load_network", lambda _adapter, _strength: network)
    model = FakeModel()
    adapter = AnimaPEIPAdapterSpec(
        name="adapter.safetensors",
        path=Path("adapter.safetensors"),
        metadata={**PE_METADATA, "ss_encoder_dim": "4", "ss_context_dim": "4", "ss_num_blocks": "1", "ss_num_heads": "2"},
    )
    features = AnimaPEFeatures(
        features=torch.zeros(1, 2, 4),
        encoder_name="pe",
        source_size=(64, 64),
    )

    patched_model = AnimaPEIPAdapterApply().apply(
        model=model,
        adapter=adapter,
        features=features,
        strength=1.0,
        start_percent=0.0,
        end_percent=1.0,
        preserve_wrapper=True,
    )[0]

    def apply_model(input_x: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        del timestep
        return dit.blocks[0].cross_attn.forward(
            input_x,
            context=input_x,
            rope_emb=None,
            transformer_options={},
        )

    output = patched_model.wrapper(
        apply_model,
        {"input": torch.zeros(1, 3, 4), "timestep": torch.tensor([0.5]), "c": {}},
    )

    assert output.shape == (1, 3, 4)
