from __future__ import annotations

from typing import Any

import torch
from torch.nn import functional as F


def ip_attention_contribution(
    *,
    attention: Any,
    adapter: Any,
    block_idx: int,
    x: torch.Tensor,
    image_tokens: torch.Tensor,
    weight: float,
    result: torch.Tensor,
    context: torch.Tensor | None,
    rope_emb: Any | None,
    transformer_options: dict[str, Any],
) -> torch.Tensor:
    """Return the native IP contribution, preferring the PE-style query path."""

    if not hasattr(attention, "compute_qkv"):
        return adapter.forward_block(block_idx, x, image_tokens, weight)
    q = _compute_query(attention, x, context, rope_emb)
    if q is None:
        return adapter.forward_block(block_idx, x, image_tokens, weight)
    del transformer_options
    return _sdpa_contribution(adapter, block_idx, q, image_tokens, weight, result)


def _compute_query(
    attention: Any,
    x: torch.Tensor,
    context: torch.Tensor | None,
    rope_emb: Any | None,
) -> torch.Tensor | None:
    try:
        q, _k, _v = attention.compute_qkv(x, context, rope_emb=rope_emb)
    except TypeError:
        try:
            q, _k, _v = attention.compute_qkv(x, context, rope_cos_sin=rope_emb)
        except TypeError:
            return None
    return q


def _sdpa_contribution(
    adapter: Any,
    block_idx: int,
    q: torch.Tensor,
    image_tokens: torch.Tensor,
    weight: float,
    result: torch.Tensor,
) -> torch.Tensor:
    attention = adapter.ip_cross_attns[block_idx]
    tokens = _match_batch(image_tokens, q.shape[0])
    key, value = adapter.project_kv(block_idx, tokens, weight)
    key = _to_sdpa_layout(key, attention.num_heads, attention.head_dim)
    value = _to_sdpa_layout(value, attention.num_heads, attention.head_dim)
    q_sdpa = _query_to_sdpa_layout(q).to(dtype=key.dtype)
    key = attention.norm_ip_k(key)
    out = F.scaled_dot_product_attention(q_sdpa, key, value)
    return out.transpose(1, 2).reshape(result.shape).to(dtype=result.dtype)


def _to_sdpa_layout(
    tensor: torch.Tensor,
    num_heads: int,
    head_dim: int,
) -> torch.Tensor:
    return tensor.unflatten(-1, (num_heads, head_dim)).transpose(1, 2).contiguous()


def _query_to_sdpa_layout(q: torch.Tensor) -> torch.Tensor:
    if q.ndim == 4:
        return q.transpose(1, 2)
    if q.ndim != 3:
        msg = f"cross-attention query must be rank 3 or 4, got {tuple(q.shape)}"
        raise RuntimeError(msg)
    return q.unsqueeze(1)


def _match_batch(tokens: torch.Tensor, batch: int) -> torch.Tensor:
    if tokens.shape[0] == batch:
        return tokens
    repeats = -(-batch // tokens.shape[0])
    return tokens.repeat(repeats, 1, 1)[:batch]
