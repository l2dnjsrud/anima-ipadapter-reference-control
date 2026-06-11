from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

try:
    from .qwenvl_model import IPAdapterQwenVL
except ImportError:
    from qwenvl_model import IPAdapterQwenVL


class QwenVLFeatureCalibrator(nn.Module):
    """Identity-initialized residual calibration for QwenVL embeddings."""

    def __init__(self, embedding_dim: int, bottleneck_dim: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(embedding_dim)
        self.down = nn.Linear(embedding_dim, bottleneck_dim, bias=False)
        self.up = nn.Linear(bottleneck_dim, embedding_dim, bias=False)
        nn.init.zeros_(self.up.weight)

    def forward(self, embedding: torch.Tensor) -> torch.Tensor:
        return embedding + self.up(F.gelu(self.down(self.norm(embedding))))


class CalibratedIPAdapterQwenVL(IPAdapterQwenVL):
    """QwenVL adapter variant with a trainable pre-resampler feature calibrator."""

    def __init__(
        self,
        embedding_dim: int = 2048,
        dit_dim: int = 2048,
        num_blocks: int = 28,
        num_queries: int = 32,
        resampler_depth: int = 4,
        resampler_heads: int = 16,
        resampler_dim: int = 1024,
        resampler_dim_head: int = 64,
        ip_heads: int = 16,
        time_embed_dim: int = 320,
        calibrator_bottleneck_dim: int = 128,
    ) -> None:
        super().__init__(
            embedding_dim=embedding_dim,
            dit_dim=dit_dim,
            num_blocks=num_blocks,
            num_queries=num_queries,
            resampler_depth=resampler_depth,
            resampler_heads=resampler_heads,
            resampler_dim=resampler_dim,
            resampler_dim_head=resampler_dim_head,
            ip_heads=ip_heads,
            time_embed_dim=time_embed_dim,
        )
        self.feature_calibrator = QwenVLFeatureCalibrator(
            embedding_dim=embedding_dim,
            bottleneck_dim=calibrator_bottleneck_dim,
        )

    def encode_ref(
        self,
        embedding: torch.Tensor,
        timestep: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return super().encode_ref(self.feature_calibrator(embedding), timestep=timestep)


def wrap_qwenvl_with_calibrator(
    adapter: IPAdapterQwenVL,
    *,
    bottleneck_dim: int,
) -> CalibratedIPAdapterQwenVL:
    wrapped = CalibratedIPAdapterQwenVL(
        embedding_dim=adapter.embedding_dim,
        dit_dim=adapter.resampler.proj_out.weight.shape[0],
        num_blocks=adapter.num_blocks,
        num_queries=adapter.num_queries,
        resampler_depth=len(adapter.resampler.layers),
        resampler_heads=_resampler_heads(adapter),
        resampler_dim=adapter.resampler.latents.shape[2],
        resampler_dim_head=_resampler_dim_head(adapter),
        ip_heads=_ip_heads(adapter),
        time_embed_dim=adapter.resampler.time_proj.weight.shape[0],
        calibrator_bottleneck_dim=bottleneck_dim,
    )
    missing, unexpected = wrapped.load_state_dict(adapter.state_dict(), strict=False)
    allowed_missing = {
        "feature_calibrator.norm.weight",
        "feature_calibrator.norm.bias",
        "feature_calibrator.down.weight",
        "feature_calibrator.up.weight",
    }
    if set(missing) != allowed_missing or unexpected:
        msg = f"could not wrap QwenVL adapter with calibrator: missing={missing}, unexpected={unexpected}"
        raise RuntimeError(msg)
    return wrapped


def _resampler_heads(adapter: IPAdapterQwenVL) -> int:
    inner_dim = adapter.resampler.layers[0][0].to_q.weight.shape[0]
    return inner_dim // _resampler_dim_head(adapter)


def _resampler_dim_head(adapter: IPAdapterQwenVL) -> int:
    inner_dim = adapter.resampler.layers[0][0].to_q.weight.shape[0]
    return 64 if inner_dim % 64 == 0 else inner_dim


def _ip_heads(adapter: IPAdapterQwenVL) -> int:
    head_dim = adapter.ip_cross_attns[0].norm_ip_q.scale.shape[0]
    dit_dim = adapter.resampler.proj_out.weight.shape[0]
    return dit_dim // head_dim if head_dim > 0 and dit_dim % head_dim == 0 else 1
