from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch
from safetensors.torch import load_file, safe_open, save_file
from torch import nn


LORA_DOWN_SUFFIX = ".lora_down.weight"
LORA_UP_SUFFIX = ".lora_up.weight"


@dataclass(frozen=True, slots=True)
class SigLIPLoRASpec:
    module_names: tuple[str, ...]
    rank: int
    alpha: float


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, *, rank: int, alpha: float) -> None:
        super().__init__()
        if rank < 1:
            raise ValueError("rank must be >= 1")
        self.base = base
        self.rank = rank
        self.alpha = alpha
        kwargs = {"device": base.weight.device, "dtype": base.weight.dtype}
        self.lora_down = nn.Linear(base.in_features, rank, bias=False, **kwargs)
        self.lora_up = nn.Linear(rank, base.out_features, bias=False, **kwargs)
        nn.init.kaiming_uniform_(self.lora_down.weight, a=5**0.5)
        nn.init.zeros_(self.lora_up.weight)
        for parameter in self.base.parameters():
            parameter.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = self.alpha / float(self.rank)
        return self.base(x) + self.lora_up(self.lora_down(x)) * scale


def default_lora_module_names(
    model: nn.Module,
    *,
    layer_count: int = 2,
    projections: tuple[str, ...] = ("q_proj", "v_proj", "out_proj"),
) -> tuple[str, ...]:
    prefix = "vision_model." if hasattr(model, "vision_model") else ""
    vision = model.vision_model if prefix else model
    layers = vision.encoder.layers
    start = max(0, len(layers) - layer_count)
    return tuple(
        f"{prefix}encoder.layers.{idx}.self_attn.{projection}"
        for idx in range(start, len(layers))
        for projection in projections
    )


def apply_siglip_lora(
    model: nn.Module,
    *,
    module_names: Iterable[str] | None = None,
    rank: int = 8,
    alpha: float = 8.0,
) -> SigLIPLoRASpec:
    targets = tuple(module_names or default_lora_module_names(model))
    for name in targets:
        parent, child_name = _parent_and_child(model, name)
        child = _get_child(parent, child_name)
        if isinstance(child, LoRALinear):
            continue
        if not isinstance(child, nn.Linear):
            raise TypeError(f"LoRA target is not Linear: {name}")
        _set_child(parent, child_name, LoRALinear(child, rank=rank, alpha=alpha))
    return SigLIPLoRASpec(module_names=targets, rank=rank, alpha=alpha)


def trainable_lora_parameters(model: nn.Module) -> list[nn.Parameter]:
    parameters = [
        parameter
        for module in model.modules()
        if isinstance(module, LoRALinear)
        for parameter in (*module.lora_down.parameters(), *module.lora_up.parameters())
    ]
    if not parameters:
        raise ValueError("SigLIP model has no LoRA parameters")
    return parameters


def lora_parameter_names(model: nn.Module) -> tuple[str, ...]:
    return tuple(
        name
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    )


def save_siglip_lora(
    model: nn.Module,
    output_path: Path,
    *,
    spec: SigLIPLoRASpec,
    metadata: dict[str, str] | None = None,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    state = _lora_state(model)
    if not state:
        raise ValueError("no LoRA state to save")
    meta = {
        "format": "pt",
        "ss_encoder": "siglip2",
        "ss_adapter": "SigLIPVisionLoRA",
        "rank": str(spec.rank),
        "alpha": str(spec.alpha),
        "module_names": json.dumps(list(spec.module_names)),
    }
    meta.update(metadata or {})
    save_file(state, str(output_path), metadata=meta)


def apply_saved_siglip_lora(model: nn.Module, path: Path) -> SigLIPLoRASpec:
    state = load_file(str(path), device="cpu")
    metadata = _metadata(path)
    module_names = _module_names_from_state(state, metadata)
    runtime_names = _resolve_module_names(model, module_names)
    rank = int(metadata.get("rank") or _infer_rank(state, module_names[0]))
    alpha = float(metadata.get("alpha") or rank)
    spec = apply_siglip_lora(model, module_names=runtime_names, rank=rank, alpha=alpha)
    missing: list[str] = []
    for state_name, runtime_name in zip(module_names, spec.module_names, strict=True):
        module = _module_by_name(model, runtime_name)
        if not isinstance(module, LoRALinear):
            raise TypeError(f"LoRA target was not wrapped: {runtime_name}")
        down_key = f"{state_name}{LORA_DOWN_SUFFIX}"
        up_key = f"{state_name}{LORA_UP_SUFFIX}"
        if down_key not in state or up_key not in state:
            missing.extend([key for key in (down_key, up_key) if key not in state])
            continue
        module.lora_down.weight.data.copy_(
            state[down_key].to(device=module.lora_down.weight.device, dtype=module.lora_down.weight.dtype)
        )
        module.lora_up.weight.data.copy_(
            state[up_key].to(device=module.lora_up.weight.device, dtype=module.lora_up.weight.dtype)
        )
    if missing:
        raise ValueError("missing LoRA keys: " + ", ".join(sorted(missing)))
    return spec


def verify_siglip_lora(path: Path) -> SigLIPLoRASpec:
    state = load_file(str(path), device="cpu")
    metadata = _metadata(path)
    module_names = _module_names_from_state(state, metadata)
    return SigLIPLoRASpec(
        module_names=module_names,
        rank=int(metadata.get("rank") or _infer_rank(state, module_names[0])),
        alpha=float(metadata.get("alpha") or _infer_rank(state, module_names[0])),
    )


def _lora_state(model: nn.Module) -> dict[str, torch.Tensor]:
    state: dict[str, torch.Tensor] = {}
    for name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            state[f"{name}{LORA_DOWN_SUFFIX}"] = module.lora_down.weight.detach().cpu()
            state[f"{name}{LORA_UP_SUFFIX}"] = module.lora_up.weight.detach().cpu()
    return state


def _metadata(path: Path) -> dict[str, str]:
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        return dict(handle.metadata() or {})


def _module_names_from_state(
    state: dict[str, torch.Tensor],
    metadata: dict[str, str],
) -> tuple[str, ...]:
    if raw := metadata.get("module_names"):
        decoded = json.loads(raw)
        return tuple(str(item) for item in decoded)
    return tuple(
        sorted(key[: -len(LORA_DOWN_SUFFIX)] for key in state if key.endswith(LORA_DOWN_SUFFIX))
    )


def _infer_rank(state: dict[str, torch.Tensor], module_name: str) -> int:
    return int(state[f"{module_name}{LORA_DOWN_SUFFIX}"].shape[0])


def _resolve_module_names(
    model: nn.Module,
    module_names: tuple[str, ...],
) -> tuple[str, ...]:
    return tuple(_resolve_module_name(model, name) for name in module_names)


def _resolve_module_name(model: nn.Module, name: str) -> str:
    if _has_module(model, name):
        return name
    if name.startswith("vision_model."):
        candidate = name.removeprefix("vision_model.")
        if _has_module(model, candidate):
            return candidate
    candidate = f"vision_model.{name}"
    if _has_module(model, candidate):
        return candidate
    return name


def _has_module(root: nn.Module, name: str) -> bool:
    try:
        _module_by_name(root, name)
    except (AttributeError, IndexError):
        return False
    return True


def _module_by_name(root: nn.Module, name: str) -> nn.Module:
    module: nn.Module = root
    for part in name.split("."):
        module = _get_child(module, part)
    return module


def _parent_and_child(root: nn.Module, name: str) -> tuple[nn.Module, str]:
    parts = name.split(".")
    parent = _module_by_name(root, ".".join(parts[:-1])) if len(parts) > 1 else root
    return parent, parts[-1]


def _get_child(module: nn.Module, name: str) -> nn.Module:
    if name.isdigit() and isinstance(module, nn.ModuleList | nn.Sequential):
        return module[int(name)]
    return getattr(module, name)


def _set_child(module: nn.Module, name: str, child: nn.Module) -> None:
    if name.isdigit() and isinstance(module, nn.ModuleList | nn.Sequential):
        module[int(name)] = child
    else:
        setattr(module, name, child)
