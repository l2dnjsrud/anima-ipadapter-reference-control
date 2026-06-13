from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

from siglip_encoder_lora import (
    LoRALinear,
    apply_saved_siglip_lora,
    apply_siglip_lora,
    lora_parameter_names,
    save_siglip_lora,
    trainable_lora_parameters,
    verify_siglip_lora,
)


class _SelfAttention(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.q_proj = nn.Linear(4, 4, bias=False)
        self.v_proj = nn.Linear(4, 4, bias=False)
        self.out_proj = nn.Linear(4, 4, bias=False)


class _Layer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.self_attn = _SelfAttention()


class _Encoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.ModuleList([_Layer(), _Layer(), _Layer()])


class _Vision(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = _Encoder()


class _TinySigLIP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.vision_model = _Vision()


class _TinySigLIPVisionOnly(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = _Encoder()


def test_apply_siglip_lora_wraps_last_layers_as_identity() -> None:
    torch.manual_seed(1)
    model = _TinySigLIP()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    x = torch.randn(2, 3, 4)
    before = model.vision_model.encoder.layers[-1].self_attn.q_proj(x)

    spec = apply_siglip_lora(model, rank=2, alpha=2.0)
    after = model.vision_model.encoder.layers[-1].self_attn.q_proj(x)

    assert len(spec.module_names) == 6
    assert torch.allclose(after, before)
    assert isinstance(model.vision_model.encoder.layers[-1].self_attn.q_proj, LoRALinear)
    assert trainable_lora_parameters(model)
    assert all("lora_" in name for name in lora_parameter_names(model))


def test_apply_siglip_lora_matches_existing_linear_device_and_dtype() -> None:
    model = _TinySigLIP().to(dtype=torch.float16)

    apply_siglip_lora(model, rank=2, alpha=2.0)
    target = model.vision_model.encoder.layers[-1].self_attn.q_proj

    assert isinstance(target, LoRALinear)
    assert target.lora_down.weight.dtype == target.base.weight.dtype
    assert target.lora_down.weight.device == target.base.weight.device


def test_siglip_lora_save_and_load_round_trip(tmp_path: Path) -> None:
    torch.manual_seed(2)
    model = _TinySigLIP()
    spec = apply_siglip_lora(model, rank=2, alpha=2.0)
    target = model.vision_model.encoder.layers[-1].self_attn.q_proj
    assert isinstance(target, LoRALinear)
    target.lora_up.weight.data.fill_(0.25)
    output_path = tmp_path / "encoder_lora.safetensors"

    save_siglip_lora(model, output_path, spec=spec)
    loaded = _TinySigLIP()
    loaded.load_state_dict(_TinySigLIP().state_dict())
    loaded_spec = apply_saved_siglip_lora(loaded, output_path)

    assert loaded_spec.module_names == spec.module_names
    assert verify_siglip_lora(output_path).rank == 2
    assert isinstance(loaded.vision_model.encoder.layers[-1].self_attn.q_proj, LoRALinear)


def test_saved_lora_loads_into_vision_model_without_vision_model_root(tmp_path: Path) -> None:
    source = _TinySigLIP()
    spec = apply_siglip_lora(source, rank=2, alpha=2.0)
    output_path = tmp_path / "encoder_lora.safetensors"
    save_siglip_lora(source, output_path, spec=spec)

    target = _TinySigLIPVisionOnly()
    loaded_spec = apply_saved_siglip_lora(target, output_path)

    assert loaded_spec.module_names == tuple(
        name.removeprefix("vision_model.") for name in spec.module_names
    )
    assert isinstance(target.encoder.layers[-1].self_attn.q_proj, LoRALinear)
