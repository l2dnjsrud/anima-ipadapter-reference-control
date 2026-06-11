from __future__ import annotations

from typing import assert_never

import torch
from torch import nn
from torch.nn import functional as F

try:
    from .siglip_model import IPAdapterSigLIP, SigLIPFeatures
except ImportError:
    from siglip_model import IPAdapterSigLIP, SigLIPFeatures


class SigLIPFeatureCalibrator(nn.Module):
    """Identity-initialized residual calibration for frozen SigLIP patch tokens."""

    def __init__(self, deep_dim: int, shallow_dim: int, bottleneck_dim: int) -> None:
        super().__init__()
        self.deep_norm = nn.LayerNorm(deep_dim)
        self.deep_down = nn.Linear(deep_dim, bottleneck_dim, bias=False)
        self.deep_up = nn.Linear(bottleneck_dim, deep_dim, bias=False)
        self.shallow_norm = nn.LayerNorm(shallow_dim)
        self.shallow_down = nn.Linear(shallow_dim, bottleneck_dim, bias=False)
        self.shallow_up = nn.Linear(bottleneck_dim, shallow_dim, bias=False)
        nn.init.zeros_(self.deep_up.weight)
        nn.init.zeros_(self.shallow_up.weight)

    def forward(self, features: SigLIPFeatures) -> SigLIPFeatures:
        deep = _calibrate_stream(
            features.deep,
            self.deep_norm,
            self.deep_down,
            self.deep_up,
        )
        match features.shallow:
            case None:
                shallow = None
            case torch.Tensor() as shallow_features:
                shallow = _calibrate_stream(
                    shallow_features,
                    self.shallow_norm,
                    self.shallow_down,
                    self.shallow_up,
                )
            case unreachable:
                assert_never(unreachable)
        return SigLIPFeatures(deep=deep, shallow=shallow)


class CalibratedIPAdapterSigLIP(IPAdapterSigLIP):
    """SigLIP adapter variant with a trainable pre-resampler feature calibrator."""

    def __init__(
        self,
        siglip_dim: int = 768,
        siglip_shallow_dim: int = 768,
        dit_dim: int = 2048,
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
        calibrator_bottleneck_dim: int = 64,
    ) -> None:
        super().__init__(
            siglip_dim=siglip_dim,
            siglip_shallow_dim=siglip_shallow_dim,
            dit_dim=dit_dim,
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
        self.feature_calibrator = SigLIPFeatureCalibrator(
            deep_dim=siglip_dim,
            shallow_dim=siglip_shallow_dim,
            bottleneck_dim=calibrator_bottleneck_dim,
        )

    def encode_ref(
        self,
        features: SigLIPFeatures | torch.Tensor,
        timestep: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return super().encode_ref(self._calibrated_features(features), timestep=timestep)

    def _calibrated_features(self, features: SigLIPFeatures | torch.Tensor) -> SigLIPFeatures:
        return self.feature_calibrator(_as_features(features))


def wrap_siglip_with_calibrator(
    adapter: IPAdapterSigLIP,
    *,
    bottleneck_dim: int,
) -> CalibratedIPAdapterSigLIP:
    """Attach an identity-initialized SigLIP feature calibrator to a checkpoint."""
    if isinstance(adapter, CalibratedIPAdapterSigLIP):
        return adapter
    wrapped = CalibratedIPAdapterSigLIP(
        siglip_dim=_siglip_dim(adapter),
        siglip_shallow_dim=_siglip_shallow_dim(adapter),
        dit_dim=adapter.resampler.proj_out.weight.shape[0],
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
        calibrator_bottleneck_dim=bottleneck_dim,
    )
    missing, unexpected = wrapped.load_state_dict(adapter.state_dict(), strict=False)
    allowed_missing = {
        "feature_calibrator.deep_norm.weight",
        "feature_calibrator.deep_norm.bias",
        "feature_calibrator.deep_down.weight",
        "feature_calibrator.deep_up.weight",
        "feature_calibrator.shallow_norm.weight",
        "feature_calibrator.shallow_norm.bias",
        "feature_calibrator.shallow_down.weight",
        "feature_calibrator.shallow_up.weight",
    }
    if set(missing) != allowed_missing or unexpected:
        msg = f"could not wrap SigLIP adapter with calibrator: missing={missing}, unexpected={unexpected}"
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


def _calibrate_stream(
    x: torch.Tensor,
    norm: nn.LayerNorm,
    down: nn.Linear,
    up: nn.Linear,
) -> torch.Tensor:
    return x + up(F.gelu(down(norm(x))))


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
    dit_dim = adapter.resampler.proj_out.weight.shape[0]
    return dit_dim // head_dim if head_dim > 0 and dit_dim % head_dim == 0 else 1
