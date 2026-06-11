from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import torch

try:
    from .siglip_feature_calibration import CalibratedIPAdapterSigLIP
    from .siglip_model import IPAdapterSigLIP
except ImportError:
    from siglip_feature_calibration import CalibratedIPAdapterSigLIP
    from siglip_model import IPAdapterSigLIP


TensorState = Mapping[str, torch.Tensor]


@dataclass(frozen=True, slots=True)
class SigLIPCheckpointError(RuntimeError):
    reason: str

    def __str__(self) -> str:
        return self.reason


@dataclass(frozen=True, slots=True)
class SigLIPCheckpointSpec:
    siglip_dim: int
    siglip_shallow_dim: int
    dit_dim: int
    num_blocks: int
    num_queries: int
    resampler_depth: int
    resampler_heads: int
    resampler_dim: int
    resampler_dim_head: int
    intermediate_dim: int
    intermediate_layers: int
    intermediate_heads: int
    ip_heads: int
    time_embed_dim: int
    use_intermediate_encoder: bool
    calibrator_bottleneck_dim: int | None


def detect_siglip_checkpoint(state: TensorState) -> SigLIPCheckpointSpec:
    """Detect a Wenaka/InstantCharacter-style SigLIP checkpoint."""

    keys = set(state)
    if _is_pe_core_checkpoint(keys):
        raise SigLIPCheckpointError(
            "This is the PE-Core Anima IP-Adapter checkpoint, not a SigLIP2 "
            "TimeResampler/IPCrossAttn checkpoint. Use the PE loader for it."
        )
    if "qwenvl_family" in keys:
        raise SigLIPCheckpointError(
            "This is a QwenVL embedding IP-Adapter checkpoint, not a SigLIP2 checkpoint."
        )
    if "resampler.latents" in keys and "resampler.time_proj.weight" not in keys:
        raise SigLIPCheckpointError(
            "Found an older Perceiver-style checkpoint without resampler.time_proj; "
            "it is not compatible with the SigLIP2 TimeResampler loader."
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
        raise SigLIPCheckpointError(
            "Malformed SigLIP checkpoint: missing " + ", ".join(sorted(missing))
        )
    block_indices = _numbered_children(keys, "ip_cross_attns")
    layer_indices = _numbered_children(keys, "resampler.layers")
    if not block_indices:
        raise SigLIPCheckpointError("Malformed SigLIP checkpoint: no ip_cross_attns blocks found.")
    if not layer_indices:
        raise SigLIPCheckpointError("Malformed SigLIP checkpoint: no TimeResampler layers found.")

    resampler_dim = state["resampler.latents"].shape[2]
    dit_dim = state["resampler.proj_out.weight"].shape[0]
    inner_dim = _first_tensor_shape(state, "resampler.layers.0.0.to_q.weight")[0]
    resampler_heads, resampler_dim_head = _infer_attention_split(inner_dim)
    use_intermediate = "intermediate_encoder.shallow_proj.weight" in keys
    intermediate_dim = (
        state["intermediate_encoder.shallow_proj.weight"].shape[0]
        if use_intermediate
        else state["resampler.proj_in.weight"].shape[1]
    )
    if use_intermediate and state["resampler.proj_in.weight"].shape[1] != intermediate_dim:
        raise SigLIPCheckpointError(
            "SigLIP checkpoint has the Wenaka resampler input_dim mismatch: "
            f"proj_in expects {state['resampler.proj_in.weight'].shape[1]}, "
            f"but CrossLayerEncoder emits {intermediate_dim}."
        )
    siglip_dim = (
        state["intermediate_encoder.deep_proj.weight"].shape[1]
        if use_intermediate
        else state["resampler.proj_in.weight"].shape[1]
    )
    siglip_shallow_dim = (
        state["intermediate_encoder.shallow_proj.weight"].shape[1] if use_intermediate else siglip_dim
    )
    calibrator_bottleneck_dim = _detect_calibrator_bottleneck(
        state, siglip_dim, siglip_shallow_dim
    )
    head_dim = _optional_first_dim(state, "ip_cross_attns.0.norm_ip_q.scale")
    return SigLIPCheckpointSpec(
        siglip_dim=siglip_dim,
        siglip_shallow_dim=siglip_shallow_dim,
        dit_dim=dit_dim,
        num_blocks=max(block_indices) + 1,
        num_queries=state["resampler.latents"].shape[1],
        resampler_depth=max(layer_indices) + 1,
        resampler_heads=resampler_heads,
        resampler_dim=resampler_dim,
        resampler_dim_head=resampler_dim_head,
        intermediate_dim=intermediate_dim,
        intermediate_layers=max(_numbered_children(keys, "intermediate_encoder.layers")) + 1
        if use_intermediate
        else 0,
        intermediate_heads=_default_heads(intermediate_dim),
        ip_heads=dit_dim // head_dim if head_dim > 0 and dit_dim % head_dim == 0 else _default_heads(dit_dim),
        time_embed_dim=state["resampler.time_proj.weight"].shape[0],
        use_intermediate_encoder=use_intermediate,
        calibrator_bottleneck_dim=calibrator_bottleneck_dim,
    )


def build_siglip_adapter_from_state(state: TensorState) -> IPAdapterSigLIP:
    spec = detect_siglip_checkpoint(state)
    adapter_cls = (
        IPAdapterSigLIP
        if spec.calibrator_bottleneck_dim is None
        else CalibratedIPAdapterSigLIP
    )
    kwargs = {
        "siglip_dim": spec.siglip_dim,
        "siglip_shallow_dim": spec.siglip_shallow_dim,
        "dit_dim": spec.dit_dim,
        "num_blocks": spec.num_blocks,
        "num_queries": spec.num_queries,
        "resampler_depth": spec.resampler_depth,
        "resampler_heads": spec.resampler_heads,
        "resampler_dim": spec.resampler_dim,
        "resampler_dim_head": spec.resampler_dim_head,
        "intermediate_dim": spec.intermediate_dim,
        "intermediate_layers": max(spec.intermediate_layers, 1),
        "intermediate_heads": spec.intermediate_heads,
        "ip_heads": spec.ip_heads,
        "time_embed_dim": spec.time_embed_dim,
        "use_intermediate_encoder": spec.use_intermediate_encoder,
    }
    if spec.calibrator_bottleneck_dim is None:
        adapter = adapter_cls(**kwargs)
    else:
        adapter = adapter_cls(
            **kwargs,
            calibrator_bottleneck_dim=spec.calibrator_bottleneck_dim,
        )
    adapter.load_state_dict(dict(state), strict=True)
    adapter.eval()
    for parameter in adapter.parameters():
        parameter.requires_grad_(False)
    return adapter


def load_siglip_adapter(path: Path) -> IPAdapterSigLIP:
    try:
        from safetensors.torch import load_file
    except ModuleNotFoundError as exc:
        raise SigLIPCheckpointError("safetensors is required to load SigLIP checkpoints.") from exc
    state = load_file(str(path), device="cpu")
    return build_siglip_adapter_from_state(state)


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
        raise SigLIPCheckpointError(f"Malformed SigLIP checkpoint: missing {key}")
    return state[key].shape


def _optional_first_dim(state: TensorState, key: str) -> int:
    return state[key].shape[0] if key in state else 0


def _detect_calibrator_bottleneck(
    state: TensorState, siglip_dim: int, siglip_shallow_dim: int
) -> int | None:
    keys = set(state)
    if not any(key.startswith("feature_calibrator.") for key in keys):
        return None
    required = (
        "feature_calibrator.deep_norm.weight",
        "feature_calibrator.deep_norm.bias",
        "feature_calibrator.deep_down.weight",
        "feature_calibrator.deep_up.weight",
        "feature_calibrator.shallow_norm.weight",
        "feature_calibrator.shallow_norm.bias",
        "feature_calibrator.shallow_down.weight",
        "feature_calibrator.shallow_up.weight",
    )
    missing = [key for key in required if key not in keys]
    if missing:
        raise SigLIPCheckpointError(
            "Malformed SigLIP calibration checkpoint: missing "
            + ", ".join(sorted(missing))
        )
    bottleneck_dim = _calibrator_weight_shape(
        state, "feature_calibrator.deep_down.weight"
    )[0]
    expected_shapes = {
        "feature_calibrator.deep_norm.weight": (siglip_dim,),
        "feature_calibrator.deep_norm.bias": (siglip_dim,),
        "feature_calibrator.deep_down.weight": (bottleneck_dim, siglip_dim),
        "feature_calibrator.deep_up.weight": (siglip_dim, bottleneck_dim),
        "feature_calibrator.shallow_norm.weight": (siglip_shallow_dim,),
        "feature_calibrator.shallow_norm.bias": (siglip_shallow_dim,),
        "feature_calibrator.shallow_down.weight": (
            bottleneck_dim,
            siglip_shallow_dim,
        ),
        "feature_calibrator.shallow_up.weight": (
            siglip_shallow_dim,
            bottleneck_dim,
        ),
    }
    for key, expected_shape in expected_shapes.items():
        actual_shape = tuple(state[key].shape)
        if actual_shape != expected_shape:
            raise SigLIPCheckpointError(
                f"Malformed SigLIP calibration checkpoint: {key} shape "
                f"{actual_shape} != {expected_shape}"
            )
    return bottleneck_dim


def _calibrator_weight_shape(state: TensorState, key: str) -> tuple[int, ...]:
    shape = tuple(state[key].shape)
    if len(shape) != 2:
        raise SigLIPCheckpointError(
            f"Malformed SigLIP calibration checkpoint: {key} must be rank 2"
        )
    return shape


def _infer_attention_split(inner_dim: int) -> tuple[int, int]:
    if inner_dim % 64 == 0:
        return inner_dim // 64, 64
    return 1, inner_dim


def _default_heads(hidden_dim: int) -> int:
    for candidate in (16, 12, 8, 4, 2):
        if hidden_dim % candidate == 0:
            return candidate
    return 1
