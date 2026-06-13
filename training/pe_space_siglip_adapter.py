from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

import torch

from siglip_feature_calibration import CalibratedIPAdapterSigLIP
from siglip_model import IPAdapterSigLIP
from training.siglip_smoke_checkpoint import (
    load_trainable_adapter,
    set_siglip_trainable_parameters,
)
from training.siglip_smoke_types import SmokeConfig
from training.siglip_smoke_types import SmokeInputError


class WeightedProjector(Protocol):
    weight: torch.Tensor


class PEInitNetwork(Protocol):
    context_dim: int
    num_blocks: int
    to_k_ip: Sequence[WeightedProjector]
    to_v_ip: Sequence[WeightedProjector]
    ip_gate: Sequence[torch.Tensor]


@dataclass(frozen=True, slots=True)
class AdapterShape:
    siglip_dim: int
    siglip_shallow_dim: int
    dit_dim: int
    num_blocks: int
    num_queries: int
    resampler_depth: int
    resampler_heads: int
    resampler_dim: int
    resampler_dim_head: int
    intermediate_dim: int
    intermediate_layers: int
    intermediate_heads: int
    ip_heads: int
    time_embed_dim: int
    use_intermediate_encoder: bool


def build_pe_space_siglip_adapter(
    base_adapter: IPAdapterSigLIP,
    pe_network: PEInitNetwork,
) -> IPAdapterSigLIP:
    """Build a SigLIP adapter that emits PE-token-space features before K/V."""
    shape = _shape_from_adapter(base_adapter)
    adapter = _new_adapter_like(base_adapter, shape, ip_hidden_dim=pe_network.context_dim)
    _load_compatible_base_weights(adapter, base_adapter)
    _copy_pe_kv(adapter, pe_network)
    return adapter


def load_teacher_adapter(
    config: SmokeConfig,
    pe_network: PEInitNetwork,
    device: torch.device,
    dtype: torch.dtype,
    *,
    pe_kv_init: bool,
    calibrator_bottleneck_dim: int | None = None,
    train_calibrator_only: bool = False,
) -> IPAdapterSigLIP:
    base_adapter = load_trainable_adapter(
        config,
        device,
        dtype,
        calibrator_bottleneck_dim=calibrator_bottleneck_dim,
    )
    adapter = build_pe_space_siglip_adapter(base_adapter, pe_network) if pe_kv_init else base_adapter
    adapter.to(device=device, dtype=torch.float32)
    adapter.train()
    set_siglip_trainable_parameters(
        adapter,
        train_calibrator_only=train_calibrator_only,
    )
    return adapter


def _new_adapter_like(
    base_adapter: IPAdapterSigLIP,
    shape: AdapterShape,
    *,
    ip_hidden_dim: int,
) -> IPAdapterSigLIP:
    match base_adapter:
        case CalibratedIPAdapterSigLIP():
            return CalibratedIPAdapterSigLIP(
                **_shape_kwargs(shape, ip_hidden_dim),
                calibrator_bottleneck_dim=base_adapter.feature_calibrator.deep_down.weight.shape[0],
            )
        case IPAdapterSigLIP():
            return IPAdapterSigLIP(**_shape_kwargs(shape, ip_hidden_dim))


def _shape_kwargs(shape: AdapterShape, ip_hidden_dim: int) -> dict[str, int | bool]:
    return {
        "siglip_dim": shape.siglip_dim,
        "siglip_shallow_dim": shape.siglip_shallow_dim,
        "dit_dim": shape.dit_dim,
        "ip_hidden_dim": ip_hidden_dim,
        "num_blocks": shape.num_blocks,
        "num_queries": shape.num_queries,
        "resampler_depth": shape.resampler_depth,
        "resampler_heads": shape.resampler_heads,
        "resampler_dim": shape.resampler_dim,
        "resampler_dim_head": shape.resampler_dim_head,
        "intermediate_dim": shape.intermediate_dim,
        "intermediate_layers": shape.intermediate_layers,
        "intermediate_heads": shape.intermediate_heads,
        "ip_heads": shape.ip_heads,
        "time_embed_dim": shape.time_embed_dim,
        "use_intermediate_encoder": shape.use_intermediate_encoder,
    }


def _load_compatible_base_weights(adapter: IPAdapterSigLIP, base_adapter: IPAdapterSigLIP) -> None:
    target = adapter.state_dict()
    compatible = {
        key: value
        for key, value in base_adapter.state_dict().items()
        if key in target and target[key].shape == value.shape
    }
    adapter.load_state_dict(compatible, strict=False)


def _copy_pe_kv(adapter: IPAdapterSigLIP, pe_network: PEInitNetwork) -> None:
    if pe_network.num_blocks != adapter.num_blocks:
        raise SmokeInputError(
            "PE network block count must match SigLIP adapter: "
            f"pe={pe_network.num_blocks}, siglip={adapter.num_blocks}"
        )
    if pe_network.context_dim != adapter.ip_hidden_dim:
        raise SmokeInputError(
            "PE network context_dim must match SigLIP ip_hidden_dim: "
            f"pe={pe_network.context_dim}, siglip={adapter.ip_hidden_dim}"
        )
    with torch.no_grad():
        for idx in range(adapter.num_blocks):
            _copy_weight(adapter.ip_cross_attns[idx].to_k_ip.weight, pe_network.to_k_ip[idx].weight)
            _copy_weight(adapter.ip_cross_attns[idx].to_v_ip.weight, pe_network.to_v_ip[idx].weight)
            adapter.ip_scales[idx].copy_(pe_network.ip_gate[idx].reshape_as(adapter.ip_scales[idx]))


def _copy_weight(target: torch.Tensor, source: torch.Tensor) -> None:
    if target.shape != source.shape:
        raise SmokeInputError(
            f"PE K/V weight shape mismatch: target={tuple(target.shape)}, source={tuple(source.shape)}"
        )
    target.copy_(source.to(device=target.device, dtype=target.dtype))


def _shape_from_adapter(adapter: IPAdapterSigLIP) -> AdapterShape:
    return AdapterShape(
        siglip_dim=_siglip_dim(adapter),
        siglip_shallow_dim=_siglip_shallow_dim(adapter),
        dit_dim=adapter.dit_dim,
        num_blocks=adapter.num_blocks,
        num_queries=adapter.num_queries,
        resampler_depth=len(adapter.resampler.layers),
        resampler_heads=_resampler_heads(adapter),
        resampler_dim=adapter.resampler.latents.shape[2],
        resampler_dim_head=_resampler_dim_head(adapter),
        intermediate_dim=_intermediate_dim(adapter),
        intermediate_layers=_intermediate_layers(adapter),
        intermediate_heads=_intermediate_heads(adapter),
        ip_heads=_ip_heads(adapter),
        time_embed_dim=adapter.resampler.time_proj.weight.shape[0],
        use_intermediate_encoder=adapter.use_intermediate_encoder,
    )


def _siglip_dim(adapter: IPAdapterSigLIP) -> int:
    if adapter.intermediate_encoder is None:
        return adapter.resampler.proj_in.weight.shape[1]
    return adapter.intermediate_encoder.deep_proj.weight.shape[1]


def _siglip_shallow_dim(adapter: IPAdapterSigLIP) -> int:
    if adapter.intermediate_encoder is None:
        return _siglip_dim(adapter)
    return adapter.intermediate_encoder.shallow_proj.weight.shape[1]


def _intermediate_dim(adapter: IPAdapterSigLIP) -> int:
    if adapter.intermediate_encoder is None:
        return adapter.resampler.proj_in.weight.shape[1]
    return adapter.intermediate_encoder.shallow_proj.weight.shape[0]


def _intermediate_layers(adapter: IPAdapterSigLIP) -> int:
    return len(adapter.intermediate_encoder.layers) if adapter.intermediate_encoder else 1


def _intermediate_heads(adapter: IPAdapterSigLIP) -> int:
    if adapter.intermediate_encoder is None:
        return 1
    return adapter.intermediate_encoder.layers[0].cross_attn.num_heads


def _resampler_heads(adapter: IPAdapterSigLIP) -> int:
    inner_dim = adapter.resampler.layers[0][0].to_q.weight.shape[0]
    return inner_dim // _resampler_dim_head(adapter)


def _resampler_dim_head(adapter: IPAdapterSigLIP) -> int:
    inner_dim = adapter.resampler.layers[0][0].to_q.weight.shape[0]
    return 64 if inner_dim % 64 == 0 else inner_dim


def _ip_heads(adapter: IPAdapterSigLIP) -> int:
    head_dim = adapter.ip_cross_attns[0].norm_ip_q.scale.shape[0]
    return adapter.dit_dim // head_dim if head_dim > 0 and adapter.dit_dim % head_dim == 0 else 1
