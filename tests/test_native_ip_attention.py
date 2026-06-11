from __future__ import annotations

import torch

from native_ip_attention import ip_attention_contribution
from siglip_model import IPAdapterSigLIP


def test_ip_attention_contribution_uses_cross_attention_query_when_available() -> None:
    adapter = _tiny_adapter()
    for scale in adapter.ip_scales:
        scale.data.fill_(1.0)
    attention = _FakeAttention()
    x = torch.randn(1, 4, 16)
    tokens = torch.randn(1, 3, 16)
    result = torch.zeros_like(x)

    contribution = ip_attention_contribution(
        attention=attention,
        adapter=adapter,
        block_idx=0,
        x=x,
        image_tokens=tokens,
        weight=1.0,
        result=result,
        context=x,
        rope_emb=None,
        transformer_options={},
    )

    assert attention.compute_qkv_called is True
    assert contribution.shape == result.shape
    assert not torch.allclose(contribution, torch.zeros_like(contribution))


def test_ip_attention_contribution_falls_back_without_compute_qkv() -> None:
    adapter = _tiny_adapter()
    for scale in adapter.ip_scales:
        scale.data.fill_(1.0)
    x = torch.randn(1, 4, 16)
    tokens = torch.randn(1, 3, 16)

    contribution = ip_attention_contribution(
        attention=object(),
        adapter=adapter,
        block_idx=0,
        x=x,
        image_tokens=tokens,
        weight=1.0,
        result=torch.zeros_like(x),
        context=None,
        rope_emb=None,
        transformer_options={},
    )

    assert contribution.shape == x.shape
    assert not torch.allclose(contribution, torch.zeros_like(contribution))


def test_ip_attention_contribution_supports_anima_rope_cos_sin_name() -> None:
    adapter = _tiny_adapter()
    for scale in adapter.ip_scales:
        scale.data.fill_(1.0)
    attention = _FakeAnimaAttention()
    x = torch.randn(1, 4, 16)

    contribution = ip_attention_contribution(
        attention=attention,
        adapter=adapter,
        block_idx=0,
        x=x,
        image_tokens=torch.randn(1, 3, 16),
        weight=1.0,
        result=torch.zeros_like(x),
        context=x,
        rope_emb=None,
        transformer_options={},
    )

    assert attention.compute_qkv_called is True
    assert contribution.shape == x.shape


class _FakeAttention:
    def __init__(self) -> None:
        self.compute_qkv_called = False

    def compute_qkv(
        self,
        x: torch.Tensor,
        context: torch.Tensor | None = None,
        *,
        rope_emb: object | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        del context, rope_emb
        self.compute_qkv_called = True
        batch, seq_len, hidden = x.shape
        q = x.reshape(batch, seq_len, 4, hidden // 4)
        return q, q, q


class _FakeAnimaAttention:
    def __init__(self) -> None:
        self.compute_qkv_called = False

    def compute_qkv(
        self,
        x: torch.Tensor,
        context: torch.Tensor | None,
        rope_cos_sin: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        del context, rope_cos_sin
        self.compute_qkv_called = True
        batch, seq_len, hidden = x.shape
        q = x.reshape(batch, seq_len, 4, hidden // 4)
        return q, q, q


def _tiny_adapter() -> IPAdapterSigLIP:
    return IPAdapterSigLIP(
        siglip_dim=8,
        siglip_shallow_dim=8,
        dit_dim=16,
        num_blocks=2,
        num_queries=3,
        resampler_depth=1,
        resampler_heads=2,
        resampler_dim=16,
        resampler_dim_head=8,
        intermediate_dim=8,
        intermediate_layers=1,
        intermediate_heads=2,
        ip_heads=4,
        time_embed_dim=10,
        use_intermediate_encoder=True,
    )
