from __future__ import annotations

from typing import assert_never

import torch
from torch import nn
from torch.nn import functional as F

try:
    from .siglip_model import IPAdapterSigLIP, SigLIPFeatures
except ImportError:
    from siglip_model import IPAdapterSigLIP, SigLIPFeatures


class SigLIPFeatureBridge(nn.Module):
    """Identity-initialized residual bridge over fused SigLIP tokens."""

    def __init__(self, token_dim: int, bottleneck_dim: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(token_dim)
        self.down = nn.Linear(token_dim, bottleneck_dim, bias=False)
        self.up = nn.Linear(bottleneck_dim, token_dim, bias=False)
        nn.init.zeros_(self.up.weight)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return tokens + self.up(F.gelu(self.down(self.norm(tokens))))


class BridgedIPAdapterSigLIP(IPAdapterSigLIP):
    """SigLIP adapter with a trainable bridge after shallow/deep fusion."""

    def __init__(
        self,
        siglip_dim: int = 768,
        siglip_shallow_dim: int = 768,
        dit_dim: int = 2048,
        ip_hidden_dim: int | None = None,
        num_blocks: int = 28,
        num_queries: int = 32,
        resampler_depth: int = 4,
        resampler_heads: int = 16,
        resampler_dim: int = 1024,
        resampler_dim_head: int = 64,
        intermediate_dim: int = 768,
        intermediate_layers: int = 4,
        intermediate_heads: int = 12,
        ip_heads: int = 16,
        time_embed_dim: int = 320,
        use_intermediate_encoder: bool = True,
        feature_bridge_bottleneck_dim: int = 128,
    ) -> None:
        if not use_intermediate_encoder:
            msg = "SigLIP feature bridge requires shallow/deep CrossLayerEncoder fusion."
            raise RuntimeError(msg)
        super().__init__(
            siglip_dim=siglip_dim,
            siglip_shallow_dim=siglip_shallow_dim,
            dit_dim=dit_dim,
            ip_hidden_dim=ip_hidden_dim,
            num_blocks=num_blocks,
            num_queries=num_queries,
            resampler_depth=resampler_depth,
            resampler_heads=resampler_heads,
            resampler_dim=resampler_dim,
            resampler_dim_head=resampler_dim_head,
            intermediate_dim=intermediate_dim,
            intermediate_layers=intermediate_layers,
            intermediate_heads=intermediate_heads,
            ip_heads=ip_heads,
            time_embed_dim=time_embed_dim,
            use_intermediate_encoder=use_intermediate_encoder,
        )
        self.feature_bridge = SigLIPFeatureBridge(
            token_dim=self.resampler.proj_in.weight.shape[1],
            bottleneck_dim=feature_bridge_bottleneck_dim,
        )

    def encode_ref(
        self,
        features: SigLIPFeatures | torch.Tensor,
        timestep: torch.Tensor | None = None,
    ) -> torch.Tensor:
        feature_set = _as_features(features)
        tokens = self._fused_tokens(feature_set)
        if timestep is None:
            timestep = torch.full(
                (tokens.shape[0],),
                0.5,
                device=tokens.device,
                dtype=tokens.dtype,
            )
        return self.resampler(self.feature_bridge(tokens), timestep)

    def _fused_tokens(self, features: SigLIPFeatures) -> torch.Tensor:
        if self.intermediate_encoder is None:
            return features.deep
        match features.shallow:
            case torch.Tensor() as shallow:
                return self.intermediate_encoder(shallow, features.deep)
            case None:
                msg = "SigLIP feature bridge expects shallow features before fusion."
                raise RuntimeError(msg)
            case unreachable:
                assert_never(unreachable)


def wrap_siglip_with_feature_bridge(
    adapter: IPAdapterSigLIP,
    *,
    bottleneck_dim: int,
) -> BridgedIPAdapterSigLIP:
    """Attach an identity-initialized bridge to fused SigLIP tokens."""
    if isinstance(adapter, BridgedIPAdapterSigLIP):
        return adapter
    if adapter.intermediate_encoder is None:
        msg = "feature bridge wrapper requires shallow/deep CrossLayerEncoder fusion"
        raise RuntimeError(msg)
    if hasattr(adapter, "feature_calibrator"):
        msg = "feature bridge wrapper does not combine with feature_calibrator checkpoints"
        raise RuntimeError(msg)
    wrapped = BridgedIPAdapterSigLIP(
        siglip_dim=_siglip_dim(adapter),
        siglip_shallow_dim=_siglip_shallow_dim(adapter),
        dit_dim=_dit_dim(adapter),
        ip_hidden_dim=adapter.resampler.proj_out.weight.shape[0],
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
        feature_bridge_bottleneck_dim=bottleneck_dim,
    )
    missing, unexpected = wrapped.load_state_dict(adapter.state_dict(), strict=False)
    allowed_missing = {
        "feature_bridge.norm.weight",
        "feature_bridge.norm.bias",
        "feature_bridge.down.weight",
        "feature_bridge.up.weight",
    }
    if set(missing) != allowed_missing or unexpected:
        msg = f"could not wrap SigLIP adapter with feature bridge: missing={missing}, unexpected={unexpected}"
        raise RuntimeError(msg)
    return wrapped


def _as_features(features: SigLIPFeatures | torch.Tensor) -> SigLIPFeatures:
    match features:
        case SigLIPFeatures():
            return features
        case torch.Tensor():
            return SigLIPFeatures(deep=features)
        case unreachable:
            assert_never(unreachable)


def _siglip_dim(adapter: IPAdapterSigLIP) -> int:
    if adapter.intermediate_encoder is None:
        return adapter.resampler.proj_in.weight.shape[1]
    return adapter.intermediate_encoder.deep_proj.weight.shape[1]


def _siglip_shallow_dim(adapter: IPAdapterSigLIP) -> int:
    if adapter.intermediate_encoder is None:
        return _siglip_dim(adapter)
    return adapter.intermediate_encoder.shallow_proj.weight.shape[1]


def _intermediate_dim(adapter: IPAdapterSigLIP) -> int:
    return adapter.resampler.proj_in.weight.shape[1]


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
    dit_dim = _dit_dim(adapter)
    return dit_dim // head_dim if head_dim > 0 and dit_dim % head_dim == 0 else 1


def _dit_dim(adapter: IPAdapterSigLIP) -> int:
    return adapter.ip_cross_attns[0].to_k_ip.weight.shape[0]
