from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from typing import Callable
from typing import Protocol

import torch

try:
    from .native_ip_attention import ip_attention_contribution
    from .native_pe_runtime import find_anima_diffusion_model, runtime_dtype
    from .siglip_model import IPAdapterSigLIP, SigLIPFeatures
except ImportError:
    from native_ip_attention import ip_attention_contribution
    from native_pe_runtime import find_anima_diffusion_model, runtime_dtype
    from siglip_model import IPAdapterSigLIP, SigLIPFeatures


class ModelSampling(Protocol):
    def percent_to_sigma(self, percent: float) -> float: ...


class ModelPatcherLike(Protocol):
    model_options: dict[str, Any]

    def clone(self) -> ModelPatcherLike: ...

    def get_model_object(self, name: str) -> ModelSampling: ...

    def set_model_unet_function_wrapper(
        self,
        wrapper: Callable[[Callable[..., torch.Tensor], dict[str, Any]], torch.Tensor],
    ) -> None: ...


@dataclass(slots=True)
class _RuntimeState:
    loaded_to: tuple[torch.device, torch.dtype] | None = None


@dataclass(frozen=True, slots=True)
class _PatchHandle:
    forwards: tuple[tuple[Any, Callable[..., torch.Tensor]], ...]


def apply_siglip_adapter(
    model: ModelPatcherLike,
    adapter: IPAdapterSigLIP,
    features: SigLIPFeatures,
    weight: float,
    start_percent: float,
    end_percent: float,
) -> ModelPatcherLike:
    dit = find_anima_diffusion_model(model)
    model_sampling = model.get_model_object("model_sampling")
    sigma_start = float(model_sampling.percent_to_sigma(float(start_percent)))
    sigma_end = float(model_sampling.percent_to_sigma(float(end_percent)))
    old_wrapper = model.model_options.get("model_function_wrapper")
    source_features = _clone_features(features)
    runtime = _RuntimeState()

    def call_next(
        apply_model: Callable[..., torch.Tensor],
        args: dict[str, Any],
    ) -> torch.Tensor:
        if old_wrapper is not None:
            return old_wrapper(apply_model, args)
        return apply_model(args["input"], args["timestep"], **args["c"])

    def wrapper(
        apply_model: Callable[..., torch.Tensor],
        args: dict[str, Any],
    ) -> torch.Tensor:
        input_x = args["input"]
        timestep = args["timestep"]
        sigma = (
            float(timestep.max().item())
            if torch.is_tensor(timestep)
            else float(timestep)
        )
        if weight == 0.0 or not (sigma_end <= sigma <= sigma_start):
            return call_next(apply_model, args)

        device = input_x.device
        dtype = (
            torch.bfloat16 if device.type == "cuda" else runtime_dtype(input_x.dtype)
        )
        target = (device, dtype)
        if runtime.loaded_to != target:
            adapter.to(device=device, dtype=dtype)
            runtime.loaded_to = target

        tokens = adapter.encode_ref(
            _features_to(source_features, device, dtype),
            timestep=_timestep_batch(
                timestep, source_features.deep.shape[0], device, dtype
            ),
        )
        handle = patch_siglip_to_comfy_attention(adapter, dit, tokens, weight)
        try:
            return call_next(apply_model, args)
        finally:
            remove_siglip_patches(handle)

    patched = model.clone()
    patched.set_model_unet_function_wrapper(wrapper)
    return patched


def patch_siglip_to_comfy_attention(
    adapter: IPAdapterSigLIP,
    dit: Any,
    image_tokens: torch.Tensor,
    weight: float,
) -> _PatchHandle:
    if not hasattr(dit, "blocks"):
        raise ValueError("SigLIP patch requires an Anima DiT with .blocks")
    if len(dit.blocks) != adapter.num_blocks:
        raise ValueError(
            f"DiT has {len(dit.blocks)} blocks, adapter expects {adapter.num_blocks}."
        )

    patched: list[tuple[Any, Callable[..., torch.Tensor]]] = []
    for idx, block in enumerate(dit.blocks):
        cross_attn = block.cross_attn
        original = cross_attn.forward
        patched.append((cross_attn, original))
        cross_attn.forward = _make_siglip_forward(
            cross_attn, original, adapter, idx, image_tokens, weight
        )
    return _PatchHandle(forwards=tuple(patched))


def remove_siglip_patches(handle: _PatchHandle) -> None:
    for cross_attn, original in handle.forwards:
        cross_attn.forward = original


def _make_siglip_forward(
    attention: Any,
    original: Callable[..., torch.Tensor],
    adapter: IPAdapterSigLIP,
    block_idx: int,
    image_tokens: torch.Tensor,
    weight: float,
) -> Callable[..., torch.Tensor]:
    def patched_forward(*args: Any, **kwargs: Any) -> torch.Tensor:
        if not args or not isinstance(args[0], torch.Tensor):
            raise RuntimeError("SigLIP cross-attention patch expected tensor input x.")
        x = args[0]
        result = original(*args, **kwargs)
        module_dtype = adapter.ip_cross_attns[block_idx].to_k_ip.weight.dtype
        query = x.to(dtype=module_dtype)
        tokens = _match_batch(
            image_tokens.to(device=x.device, dtype=module_dtype), x.shape[0]
        )
        context, rope_emb, transformer_options = _attention_inputs(args, kwargs)
        contribution = ip_attention_contribution(
            attention=attention,
            adapter=adapter,
            block_idx=block_idx,
            x=query,
            image_tokens=tokens,
            weight=weight,
            result=result,
            context=context,
            rope_emb=rope_emb,
            transformer_options=transformer_options,
        )
        return result + contribution.to(result)

    return patched_forward


def _attention_inputs(
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> tuple[torch.Tensor | None, Any | None, dict[str, Any]]:
    context = kwargs.get("context")
    rope_emb = kwargs.get("rope_emb", kwargs.get("rope_cos_sin"))
    transformer_options = kwargs.get("transformer_options", {})
    if len(args) >= 3 and torch.is_tensor(args[2]):
        context = args[2]
    elif len(args) >= 2 and (torch.is_tensor(args[1]) or args[1] is None):
        context = args[1]
    if len(args) >= 4:
        rope_emb = args[3]
    if isinstance(context, torch.Tensor) or context is None:
        return context, rope_emb, transformer_options
    return None, rope_emb, transformer_options


def _clone_features(features: SigLIPFeatures) -> SigLIPFeatures:
    shallow = (
        features.shallow.detach().clone() if features.shallow is not None else None
    )
    return SigLIPFeatures(deep=features.deep.detach().clone(), shallow=shallow)


def _features_to(
    features: SigLIPFeatures, device: torch.device, dtype: torch.dtype
) -> SigLIPFeatures:
    shallow = (
        features.shallow.to(device=device, dtype=dtype)
        if features.shallow is not None
        else None
    )
    return SigLIPFeatures(
        deep=features.deep.to(device=device, dtype=dtype), shallow=shallow
    )


def _match_batch(tokens: torch.Tensor, batch: int) -> torch.Tensor:
    if tokens.shape[0] == batch:
        return tokens
    repeats = -(-batch // tokens.shape[0])
    return tokens.repeat(repeats, 1, 1)[:batch]


def _timestep_batch(
    timestep: torch.Tensor | float,
    batch: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    if torch.is_tensor(timestep):
        values = timestep.flatten().to(device=device, dtype=dtype)
    else:
        values = torch.full((batch,), float(timestep), device=device, dtype=dtype)
    return values[:1].expand(batch) if values.numel() == 1 else values[:batch]
