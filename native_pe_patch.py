from __future__ import annotations

import logging
from typing import Any

import torch
import torch.nn.functional as F

LOGGER = logging.getLogger(__name__)


def patch_network_to_comfy_attention(network: Any, unet: Any) -> None:
    """Patch ComfyUI Anima cross-attention modules for a loaded PE IP-Adapter."""
    if getattr(network, "_patched", False):
        LOGGER.warning("PE IP-Adapter Comfy patch called twice; skipping")
        return
    if unet is None or not hasattr(unet, "blocks"):
        raise ValueError("Comfy PE IP-Adapter patch requires an Anima DiT with .blocks")
    if len(unet.blocks) != network.num_blocks:
        raise ValueError(
            f"DiT has {len(unet.blocks)} blocks, IP-Adapter expects {network.num_blocks}."
        )

    for idx, block in enumerate(unet.blocks):
        cross_attn = block.cross_attn
        _validate_cross_attention(network, cross_attn, idx)
        network._cross_attn_modules.append(cross_attn)
        network._original_forwards.append(cross_attn.forward)
        cross_attn._ip_k_cached = None
        cross_attn._ip_v_cached = None
        cross_attn._ip_gate_scale_cached = None
        cross_attn._ip_gate = network.ip_gate[idx]
        cross_attn._ip_diag_ratio_sum = None
        cross_attn._ip_diag_count = None
        cross_attn.forward = _make_comfy_patched_forward(cross_attn, network)

    network._patched = True
    LOGGER.info("PE IP-Adapter: patched %s ComfyUI cross-attn blocks", len(unet.blocks))


def _validate_cross_attention(network: Any, cross_attn: Any, idx: int) -> None:
    if cross_attn.is_selfattn:
        raise RuntimeError(f"block[{idx}].cross_attn unexpectedly self-attention")
    if cross_attn.context_dim != network.context_dim:
        raise ValueError(
            f"block[{idx}].cross_attn context_dim {cross_attn.context_dim} "
            f"!= IP context_dim {network.context_dim}"
        )
    if cross_attn.n_heads != network.num_heads or cross_attn.head_dim != network.head_dim:
        raise ValueError(
            f"block[{idx}].cross_attn heads/head_dim mismatch: "
            f"({cross_attn.n_heads}, {cross_attn.head_dim}) vs "
            f"({network.num_heads}, {network.head_dim})"
        )


def _make_comfy_patched_forward(orig_attn: Any, ip_net: Any):
    def patched_forward(
        x: torch.Tensor,
        context: torch.Tensor | None = None,
        rope_emb: torch.Tensor | None = None,
        transformer_options: dict | None = None,
    ) -> torch.Tensor:
        options = {} if transformer_options is None else transformer_options
        q, k, v = orig_attn.compute_qkv(x, context, rope_emb=rope_emb)
        text_result = orig_attn.attn_op(q, k, v, transformer_options=options)
        text_result = _flatten_attention_result(text_result)

        ip_k = getattr(orig_attn, "_ip_k_cached", None)
        ip_v = getattr(orig_attn, "_ip_v_cached", None)
        if ip_k is not None and ip_v is not None:
            text_result = text_result + _ip_attention_contribution(
                orig_attn,
                ip_net,
                q,
                ip_k,
                ip_v,
                text_result,
            )

        return orig_attn.output_dropout(orig_attn.output_proj(text_result))

    return patched_forward


def _flatten_attention_result(text_result: torch.Tensor) -> torch.Tensor:
    if text_result.ndim == 4:
        return text_result.reshape(text_result.shape[0], text_result.shape[1], -1)
    return text_result


def _ip_attention_contribution(
    orig_attn: Any,
    ip_net: Any,
    q: torch.Tensor,
    ip_k: torch.Tensor,
    ip_v: torch.Tensor,
    text_result: torch.Tensor,
) -> torch.Tensor:
    batch = q.shape[0]
    if ip_k.shape[0] == 1 and batch > 1:
        ip_k = ip_k.expand(batch, -1, -1, -1)
        ip_v = ip_v.expand(batch, -1, -1, -1)
    elif ip_k.shape[0] != batch:
        raise RuntimeError(f"IP-Adapter K/V batch {ip_k.shape[0]} does not match q batch {batch}")

    q_sdpa = q.transpose(1, 2).to(dtype=ip_k.dtype)
    ip_out = F.scaled_dot_product_attention(q_sdpa, ip_k, ip_v)
    ip_out = ip_out.transpose(1, 2).reshape(text_result.shape).to(dtype=text_result.dtype)
    gate_scale = getattr(orig_attn, "_ip_gate_scale_cached", None)
    if gate_scale is None:
        raise RuntimeError("IP-Adapter K/V cache is present but gate scale cache is missing")
    contribution = gate_scale.to(device=ip_out.device, dtype=ip_out.dtype) * ip_out
    _record_diagnostics(orig_attn, ip_net, contribution, text_result)
    return contribution


def _record_diagnostics(
    orig_attn: Any,
    ip_net: Any,
    contribution: torch.Tensor,
    text_result: torch.Tensor,
) -> None:
    diag_sum = getattr(orig_attn, "_ip_diag_ratio_sum", None)
    if not getattr(ip_net, "_diag_enabled", False) or diag_sum is None:
        return
    with torch.no_grad():
        ip_norm = contribution.float().norm()
        txt_norm = text_result.float().norm().clamp_min(1e-12)
        diag_sum.add_(ip_norm / txt_norm)
        orig_attn._ip_diag_count.add_(1)
