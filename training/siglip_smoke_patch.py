from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Protocol

import torch

from native_ip_attention import ip_attention_contribution
from siglip_model import IPAdapterSigLIP


type CrossAttnForward = Callable[
    [torch.Tensor, AttentionParamsLike, torch.Tensor, tuple[torch.Tensor, torch.Tensor] | None],
    torch.Tensor,
]


class AttentionParamsLike(Protocol):
    pass


class CrossAttentionLike(Protocol):
    forward: CrossAttnForward


class BlockLike(Protocol):
    cross_attn: CrossAttentionLike


class AnimaLike(Protocol):
    blocks: list[BlockLike]


@contextmanager
def patched_cross_attention(
    anima: AnimaLike,
    adapter: IPAdapterSigLIP,
    image_tokens: torch.Tensor,
    *,
    weight: float = 1.0,
) -> Iterator[None]:
    originals: list[tuple[CrossAttentionLike, CrossAttnForward]] = []
    for block_idx, block in enumerate(anima.blocks):
        cross_attn = block.cross_attn
        original = cross_attn.forward
        originals.append((cross_attn, original))
        cross_attn.forward = _patched_forward(
            cross_attn,
            original,
            adapter,
            image_tokens,
            block_idx,
            weight,
        )
    try:
        yield
    finally:
        for cross_attn, original in originals:
            cross_attn.forward = original


def _patched_forward(
    attention: CrossAttentionLike,
    original: CrossAttnForward,
    adapter: IPAdapterSigLIP,
    image_tokens: torch.Tensor,
    block_idx: int,
    weight: float,
) -> CrossAttnForward:
    def forward(
        x: torch.Tensor,
        attn_params: AttentionParamsLike,
        context: torch.Tensor,
        rope_cos_sin: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        base = original(x, attn_params, context, rope_cos_sin)
        contribution = ip_attention_contribution(
            attention=attention,
            adapter=adapter,
            block_idx=block_idx,
            x=x,
            image_tokens=image_tokens,
            weight=weight,
            result=base,
            context=context,
            rope_emb=rope_cos_sin,
            transformer_options={},
        )
        return base + contribution

    return forward
