from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True, slots=True)
class SigLIPFeatures:
    """Patch features emitted by SigLIP2."""

    deep: torch.Tensor
    shallow: torch.Tensor | None = None


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-8) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.ones(dim))
        self.eps = eps
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = x.norm(2, dim=-1, keepdim=True) * self.dim ** -0.5
        return self.scale * (x / (rms + self.eps))


class CrossAttentionFusionLayer(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int) -> None:
        super().__init__()
        self.q_norm = nn.LayerNorm(hidden_dim)
        self.kv_norm = nn.LayerNorm(hidden_dim)
        self.cross_attn = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=0.0, batch_first=True
        )
        self.ff_norm = nn.LayerNorm(hidden_dim)
        self.ff = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4, bias=False),
            nn.GELU(),
            nn.Linear(hidden_dim * 4, hidden_dim, bias=False),
        )

    def forward(self, shallow: torch.Tensor, deep: torch.Tensor) -> torch.Tensor:
        kv = self.kv_norm(deep)
        attn, _ = self.cross_attn(self.q_norm(shallow), kv, kv, need_weights=False)
        fused = shallow + attn
        return fused + self.ff(self.ff_norm(fused))


class CrossLayerEncoder(nn.Module):
    """Fuse shallow and deep SigLIP patch tokens without Transformer memory misuse."""

    def __init__(
        self,
        shallow_dim: int,
        deep_dim: int,
        hidden_dim: int,
        num_layers: int,
        num_heads: int,
    ) -> None:
        super().__init__()
        self.shallow_proj = nn.Linear(shallow_dim, hidden_dim, bias=False)
        self.deep_proj = nn.Linear(deep_dim, hidden_dim, bias=False)
        self.norm_shallow = RMSNorm(hidden_dim)
        self.norm_deep = RMSNorm(hidden_dim)
        self.layers = nn.ModuleList(
            [CrossAttentionFusionLayer(hidden_dim, num_heads) for _ in range(num_layers)]
        )

    def forward(self, shallow_features: torch.Tensor, deep_features: torch.Tensor) -> torch.Tensor:
        shallow = self.norm_shallow(self.shallow_proj(shallow_features))
        deep = self.norm_deep(self.deep_proj(deep_features))
        for layer in self.layers:
            shallow = layer(shallow, deep)
        return torch.cat([shallow, deep], dim=1)


class PerceiverAttention(nn.Module):
    def __init__(self, dim: int, dim_head: int, heads: int) -> None:
        super().__init__()
        self.heads = heads
        inner_dim = dim_head * heads
        self.norm_kv = nn.LayerNorm(dim)
        self.norm_q = nn.LayerNorm(dim)
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, dim, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        latents: torch.Tensor,
        shift: torch.Tensor,
        scale: torch.Tensor,
    ) -> torch.Tensor:
        x = self.norm_kv(x)
        latents = self.norm_q(latents)
        latents = latents * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
        batch, latent_len, _ = latents.shape
        query = self.to_q(latents)
        key, value = self.to_kv(torch.cat([x, latents], dim=1)).chunk(2, dim=-1)
        query = query.view(batch, latent_len, self.heads, -1).transpose(1, 2)
        key = key.view(batch, key.shape[1], self.heads, -1).transpose(1, 2)
        value = value.view(batch, value.shape[1], self.heads, -1).transpose(1, 2)
        out = F.scaled_dot_product_attention(query, key, value)
        return self.to_out(out.transpose(1, 2).reshape(batch, latent_len, -1))


class TimeResampler(nn.Module):
    def __init__(
        self,
        input_dim: int,
        dim: int,
        output_dim: int,
        num_queries: int,
        depth: int,
        dim_head: int,
        heads: int,
        time_embed_dim: int,
    ) -> None:
        super().__init__()
        self.latents = nn.Parameter(torch.randn(1, num_queries, dim) / dim**0.5)
        self.proj_in = nn.Linear(input_dim, dim, bias=False)
        self.proj_out = nn.Linear(dim, output_dim, bias=False)
        self.norm_out = nn.LayerNorm(output_dim)
        self.time_proj = nn.Linear(1, time_embed_dim)
        self.time_mlp = nn.Sequential(nn.Linear(time_embed_dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.layers = nn.ModuleList(
            [
                nn.ModuleList(
                    [
                        PerceiverAttention(dim, dim_head, heads),
                        nn.Sequential(
                            nn.LayerNorm(dim),
                            nn.Linear(dim, dim * 4, bias=False),
                            nn.GELU(),
                            nn.Linear(dim * 4, dim, bias=False),
                        ),
                        nn.Sequential(nn.SiLU(), nn.Linear(dim, 4 * dim)),
                    ]
                )
                for _ in range(depth)
            ]
        )

    def _embed_timestep(self, timestep: torch.Tensor) -> torch.Tensor:
        width = self.time_proj.out_features
        half = width // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(half, device=timestep.device) / max(half, 1)
        )
        args = timestep.reshape(-1, 1).float() * freqs.reshape(1, -1)
        embedding = torch.cat([args.cos(), args.sin()], dim=-1)
        if embedding.shape[-1] < width:
            embedding = F.pad(embedding, (0, width - embedding.shape[-1]))
        first = self.time_mlp[0]
        return self.time_mlp(embedding.to(device=timestep.device, dtype=first.weight.dtype))

    def forward(self, x: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        batch = x.shape[0]
        x = self.proj_in(x)
        latents = self.latents.expand(batch, -1, -1)
        timestep_embedding = self._embed_timestep(timestep)
        for attn, ff, ada_ln in self.layers:
            shift_msa, scale_msa, shift_ff, scale_ff = ada_ln(timestep_embedding).chunk(4, dim=1)
            latents = latents + attn(x, latents, shift_msa, scale_msa)
            residual = latents
            latents = ff[0](latents) * (1 + scale_ff.unsqueeze(1)) + shift_ff.unsqueeze(1)
            latents = ff[3](ff[2](ff[1](latents))) + residual
        return self.norm_out(self.proj_out(latents))


class IPCrossAttn(nn.Module):
    def __init__(self, hidden_size: int, ip_hidden_dim: int, num_heads: int) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.norm_ip_q = RMSNorm(self.head_dim, eps=1e-6)
        self.norm_ip_k = RMSNorm(self.head_dim, eps=1e-6)
        self.to_k_ip = nn.Linear(ip_hidden_dim, hidden_size, bias=False)
        self.to_v_ip = nn.Linear(ip_hidden_dim, hidden_size, bias=False)

    def project_kv(self, ip_hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.to_k_ip(ip_hidden_states), self.to_v_ip(ip_hidden_states)

    def forward(self, query: torch.Tensor, ip_hidden_states: torch.Tensor) -> torch.Tensor:
        batch, seq_len, hidden = query.shape
        token_len = ip_hidden_states.shape[1]
        q = query.view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        q = self.norm_ip_q(q)
        k_ip, v_ip = self.project_kv(ip_hidden_states)
        k_ip = k_ip.view(batch, token_len, self.num_heads, self.head_dim).transpose(1, 2)
        v_ip = v_ip.view(batch, token_len, self.num_heads, self.head_dim).transpose(1, 2)
        out = F.scaled_dot_product_attention(q, self.norm_ip_k(k_ip), v_ip)
        return out.transpose(1, 2).reshape(batch, seq_len, hidden)


class IPAdapterSigLIP(nn.Module):
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
    ) -> None:
        super().__init__()
        self.num_blocks = num_blocks
        self.num_queries = num_queries
        self.use_intermediate_encoder = use_intermediate_encoder
        self.intermediate_encoder = (
            CrossLayerEncoder(
                siglip_shallow_dim, siglip_dim, intermediate_dim, intermediate_layers, intermediate_heads
            )
            if use_intermediate_encoder
            else None
        )
        input_dim = intermediate_dim if use_intermediate_encoder else siglip_dim
        self.resampler = TimeResampler(
            input_dim, resampler_dim, dit_dim, num_queries, resampler_depth,
            resampler_dim_head, resampler_heads, time_embed_dim
        )
        self.ip_cross_attns = nn.ModuleList(
            [IPCrossAttn(dit_dim, dit_dim, ip_heads) for _ in range(num_blocks)]
        )
        self.ip_scales = nn.ParameterList(
            [nn.Parameter(torch.full((1,), 0.01)) for _ in range(num_blocks)]
        )

    def encode_ref(
        self,
        features: SigLIPFeatures | torch.Tensor,
        timestep: torch.Tensor | None = None,
    ) -> torch.Tensor:
        deep = features.deep if isinstance(features, SigLIPFeatures) else features
        shallow = features.shallow if isinstance(features, SigLIPFeatures) else None
        if timestep is None:
            timestep = torch.full((deep.shape[0],), 0.5, device=deep.device, dtype=deep.dtype)
        if self.intermediate_encoder is not None:
            if shallow is None:
                msg = "SigLIP checkpoint expects shallow features; encoder returned deep features only."
                raise RuntimeError(msg)
            deep = self.intermediate_encoder(shallow, deep)
        return self.resampler(deep, timestep)

    def forward_block(
        self, block_idx: int, query: torch.Tensor, image_tokens: torch.Tensor, weight: float = 1.0
    ) -> torch.Tensor:
        scale = self.ip_scales[block_idx].to(device=query.device, dtype=query.dtype)
        return self.ip_cross_attns[block_idx](query, image_tokens) * scale * weight

    def project_kv(
        self, block_idx: int, image_tokens: torch.Tensor, weight: float = 1.0
    ) -> tuple[torch.Tensor, torch.Tensor]:
        key, value = self.ip_cross_attns[block_idx].project_kv(image_tokens)
        scale = self.ip_scales[block_idx].to(device=value.device, dtype=value.dtype) * weight
        return key, value * scale
