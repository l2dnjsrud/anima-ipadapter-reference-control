from __future__ import annotations

import torch
from torch import nn

try:
    from .siglip_model import IPCrossAttn, TimeResampler
except ImportError:
    from siglip_model import IPCrossAttn, TimeResampler


QWENVL_FAMILY_KEY = "qwenvl_family"


class IPAdapterQwenVL(nn.Module):
    """IP-Adapter that consumes Qwen3-VL image embeddings."""

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
    ) -> None:
        super().__init__()
        self.num_blocks = num_blocks
        self.num_queries = num_queries
        self.embedding_dim = embedding_dim
        self.register_buffer(QWENVL_FAMILY_KEY, torch.ones(1), persistent=True)
        self.resampler = TimeResampler(
            embedding_dim,
            resampler_dim,
            dit_dim,
            num_queries,
            resampler_depth,
            resampler_dim_head,
            resampler_heads,
            time_embed_dim,
        )
        self.ip_cross_attns = nn.ModuleList(
            [IPCrossAttn(dit_dim, dit_dim, ip_heads) for _ in range(num_blocks)]
        )
        self.ip_scales = nn.ParameterList(
            [nn.Parameter(torch.full((1,), 0.01)) for _ in range(num_blocks)]
        )

    def encode_ref(
        self,
        embedding: torch.Tensor,
        timestep: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if embedding.ndim == 2:
            embedding = embedding.unsqueeze(1)
        if embedding.ndim != 3:
            msg = "QwenVL embedding must be rank 2 [B,D] or rank 3 [B,T,D]."
            raise RuntimeError(msg)
        if embedding.shape[-1] != self.embedding_dim:
            msg = (
                f"QwenVL embedding dim {embedding.shape[-1]} does not match "
                f"adapter dim {self.embedding_dim}."
            )
            raise RuntimeError(msg)
        if timestep is None:
            timestep = torch.full(
                (embedding.shape[0],),
                0.5,
                device=embedding.device,
                dtype=embedding.dtype,
            )
        return self.resampler(embedding, timestep)

    def forward_block(
        self,
        block_idx: int,
        query: torch.Tensor,
        image_tokens: torch.Tensor,
        weight: float = 1.0,
    ) -> torch.Tensor:
        scale = self.ip_scales[block_idx].to(device=query.device, dtype=query.dtype)
        return self.ip_cross_attns[block_idx](query, image_tokens) * scale * weight

    def project_kv(
        self,
        block_idx: int,
        image_tokens: torch.Tensor,
        weight: float = 1.0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        key, value = self.ip_cross_attns[block_idx].project_kv(image_tokens)
        scale = self.ip_scales[block_idx].to(device=value.device, dtype=value.dtype) * weight
        return key, value * scale
