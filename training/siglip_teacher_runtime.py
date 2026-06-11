from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

import torch

from native_pe_models import load_pe_adapter_spec
from native_pe_runtime import load_network
from siglip_model import IPAdapterSigLIP
from training.pe_space_siglip_adapter import PEInitNetwork, load_teacher_adapter
from training.pe_teacher_distillation import PETeacherNetwork
from training.siglip_real_smoke import freeze_module
from training.siglip_smoke_types import SmokeConfig


class TeacherPENetwork(PEInitNetwork, PETeacherNetwork, Protocol):
    pass


@dataclass(frozen=True, slots=True)
class TeacherRuntime:
    anima: torch.nn.Module
    vae: torch.nn.Module
    text_encoder: torch.nn.Module
    siglip: torch.nn.Module
    processor: Callable[..., dict[str, torch.Tensor]]
    pe_network: TeacherPENetwork
    pe_encoder: torch.nn.Module
    adapter: IPAdapterSigLIP
    prepare_text_inputs: Callable[..., tuple[dict[str, list[torch.Tensor]], torch.Tensor]]
    encode_pe_from_imageminus1to1: Callable[..., list[torch.Tensor]]
    frozen_params: int


def load_teacher_runtime(
    config: SmokeConfig,
    *,
    device: torch.device,
    dtype: torch.dtype,
    pe_kv_init: bool,
    pe_encoder_name: str,
) -> TeacherRuntime:
    from library.anima.weights import load_anima_model, load_qwen3_text_encoder
    from library.inference.text import prepare_text_inputs
    from library.models.qwen_vae import load_vae
    from library.vision import encode_pe_from_imageminus1to1, load_pe_encoder
    from transformers import AutoImageProcessor, SiglipVisionModel

    anima = load_anima_model(device, str(config.dit_path), "torch", device, dtype)
    anima.to(device=device, dtype=dtype)
    frozen_params = freeze_module(anima)
    vae = load_vae(str(config.vae_path), device=device, dtype=dtype, eval=True)
    frozen_params += freeze_module(vae)
    text_encoder, _ = load_qwen3_text_encoder(
        str(config.text_encoder_path), dtype=dtype, device=str(device)
    )
    frozen_params += freeze_module(text_encoder)
    siglip = SiglipVisionModel.from_pretrained(
        config.siglip_model_id, torch_dtype=dtype, trust_remote_code=True
    ).to(device)
    frozen_params += freeze_module(siglip)
    pe_spec = load_pe_adapter_spec(config.pe_checkpoint_path)
    pe_network = load_network(pe_spec, strength=1.0).to(device=device, dtype=dtype)
    adapter = load_teacher_adapter(config, pe_network, device, dtype, pe_kv_init=pe_kv_init)
    return TeacherRuntime(
        anima=anima,
        vae=vae,
        text_encoder=text_encoder,
        siglip=siglip,
        processor=AutoImageProcessor.from_pretrained(config.siglip_model_id),
        pe_network=pe_network,
        pe_encoder=load_pe_encoder(device, name=pe_encoder_name, dtype=torch.bfloat16),
        adapter=adapter,
        prepare_text_inputs=prepare_text_inputs,
        encode_pe_from_imageminus1to1=encode_pe_from_imageminus1to1,
        frozen_params=frozen_params,
    )
