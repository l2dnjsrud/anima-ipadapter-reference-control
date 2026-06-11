from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import torch

try:
    from .qwenvl_model import IPAdapterQwenVL, QWENVL_FAMILY_KEY
except ImportError:
    from qwenvl_model import IPAdapterQwenVL, QWENVL_FAMILY_KEY


TensorState = Mapping[str, torch.Tensor]


@dataclass(frozen=True, slots=True)
class QwenVLCheckpointError(RuntimeError):
    reason: str

    def __str__(self) -> str:
        return self.reason


@dataclass(frozen=True, slots=True)
class QwenVLCheckpointSpec:
    embedding_dim: int
    dit_dim: int
    num_blocks: int
    num_queries: int
    resampler_depth: int
    resampler_heads: int
    resampler_dim: int
    resampler_dim_head: int
    ip_heads: int
    time_embed_dim: int


def detect_qwenvl_checkpoint(state: TensorState) -> QwenVLCheckpointSpec:
    """Detect an Anima Qwen3-VL embedding IP-Adapter checkpoint."""

    keys = set(state)
    if _is_pe_core_checkpoint(keys):
        raise QwenVLCheckpointError(
            "This is the PE-Core Anima IP-Adapter checkpoint, not a QwenVL checkpoint."
        )
    if QWENVL_FAMILY_KEY not in keys:
        raise QwenVLCheckpointError("Malformed QwenVL checkpoint: missing QwenVL family marker.")
    if _has_siglip_only_keys(keys):
        raise QwenVLCheckpointError(
            "This checkpoint contains SigLIP-only tensors; use the SigLIP loader."
        )
    missing = [
        key
        for key in (
            "resampler.latents",
            "resampler.proj_in.weight",
            "resampler.proj_out.weight",
            "resampler.time_proj.weight",
        )
        if key not in keys
    ]
    if missing:
        raise QwenVLCheckpointError(
            "Malformed QwenVL checkpoint: missing " + ", ".join(sorted(missing))
        )
    block_indices = _numbered_children(keys, "ip_cross_attns")
    layer_indices = _numbered_children(keys, "resampler.layers")
    if not block_indices:
        raise QwenVLCheckpointError("Malformed QwenVL checkpoint: no IP blocks found.")
    if not layer_indices:
        raise QwenVLCheckpointError(
            "Malformed QwenVL checkpoint: no TimeResampler layers found."
        )
    resampler_inner = _first_tensor_shape(state, "resampler.layers.0.0.to_q.weight")[0]
    resampler_heads, resampler_dim_head = _infer_attention_split(resampler_inner)
    head_dim = _optional_first_dim(state, "ip_cross_attns.0.norm_ip_q.scale")
    dit_dim = state["resampler.proj_out.weight"].shape[0]
    return QwenVLCheckpointSpec(
        embedding_dim=state["resampler.proj_in.weight"].shape[1],
        dit_dim=dit_dim,
        num_blocks=max(block_indices) + 1,
        num_queries=state["resampler.latents"].shape[1],
        resampler_depth=max(layer_indices) + 1,
        resampler_heads=resampler_heads,
        resampler_dim=state["resampler.latents"].shape[2],
        resampler_dim_head=resampler_dim_head,
        ip_heads=dit_dim // head_dim if head_dim > 0 and dit_dim % head_dim == 0 else _default_heads(dit_dim),
        time_embed_dim=state["resampler.time_proj.weight"].shape[0],
    )


def build_qwenvl_adapter_from_state(state: TensorState) -> IPAdapterQwenVL:
    spec = detect_qwenvl_checkpoint(state)
    adapter = IPAdapterQwenVL(
        embedding_dim=spec.embedding_dim,
        dit_dim=spec.dit_dim,
        num_blocks=spec.num_blocks,
        num_queries=spec.num_queries,
        resampler_depth=spec.resampler_depth,
        resampler_heads=spec.resampler_heads,
        resampler_dim=spec.resampler_dim,
        resampler_dim_head=spec.resampler_dim_head,
        ip_heads=spec.ip_heads,
        time_embed_dim=spec.time_embed_dim,
    )
    adapter.load_state_dict(dict(state), strict=True)
    adapter.eval()
    for parameter in adapter.parameters():
        parameter.requires_grad_(False)
    return adapter


def load_qwenvl_adapter(path: Path) -> IPAdapterQwenVL:
    try:
        from safetensors.torch import load_file
    except ModuleNotFoundError as exc:
        raise QwenVLCheckpointError("safetensors is required to load QwenVL checkpoints.") from exc
    state = load_file(str(path), device="cpu")
    return build_qwenvl_adapter_from_state(state)


def _has_siglip_only_keys(keys: set[str]) -> bool:
    return "intermediate_encoder.shallow_proj.weight" in keys or any(
        key.startswith("feature_calibrator.") for key in keys
    )


def _is_pe_core_checkpoint(keys: set[str]) -> bool:
    return "ip_centroid" in keys or any(key.startswith("ip_gate.") for key in keys)


def _numbered_children(keys: set[str], prefix: str) -> list[int]:
    indices: list[int] = []
    head = prefix + "."
    for key in keys:
        if key.startswith(head):
            piece = key[len(head) :].split(".", maxsplit=1)[0]
            if piece.isdigit():
                indices.append(int(piece))
    return indices


def _first_tensor_shape(state: TensorState, key: str) -> torch.Size:
    if key not in state:
        raise QwenVLCheckpointError(f"Malformed QwenVL checkpoint: missing {key}")
    return state[key].shape


def _optional_first_dim(state: TensorState, key: str) -> int:
    return state[key].shape[0] if key in state else 0


def _infer_attention_split(inner_dim: int) -> tuple[int, int]:
    if inner_dim % 64 == 0:
        return inner_dim // 64, 64
    return 1, inner_dim


def _default_heads(hidden_dim: int) -> int:
    for candidate in (16, 12, 8, 4, 2):
        if hidden_dim % candidate == 0:
            return candidate
    return 1
