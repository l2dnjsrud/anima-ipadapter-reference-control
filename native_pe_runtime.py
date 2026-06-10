from __future__ import annotations

import sys
import types
from dataclasses import dataclass
from typing import Any, Final

import torch

try:
    from .native_pe_models import DEFAULT_ANIMA_ROOT, AnimaPEIPAdapterSpec
except ImportError:
    from native_pe_models import DEFAULT_ANIMA_ROOT, AnimaPEIPAdapterSpec


DIFFUSION_ATTR_PATHS: Final[tuple[str, ...]] = (
    "diffusion_model",
    "model.diffusion_model",
    "model.model.diffusion_model",
    "inner_model.diffusion_model",
    "model.inner_model.diffusion_model",
)
SEARCH_CHILD_ATTRS: Final[tuple[str, ...]] = (
    "model",
    "inner_model",
    "diffusion_model",
    "unet",
    "wrapped",
)


def ensure_anima_root() -> None:
    root = str(DEFAULT_ANIMA_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)


def install_anima_training_stubs() -> None:
    if "library.training.method_adapter" in sys.modules:
        return

    training_pkg = types.ModuleType("library.training")
    training_pkg.__path__ = []

    method_adapter = types.ModuleType("library.training.method_adapter")

    class MethodAdapter:
        pass

    @dataclass(frozen=True)
    class SetupCtx:
        args: Any
        accelerator: Any
        network: Any
        unet: Any
        text_encoders: list
        weight_dtype: torch.dtype

    @dataclass(frozen=True)
    class StepCtx:
        args: Any
        accelerator: Any
        network: Any
        weight_dtype: torch.dtype

    @dataclass(frozen=True)
    class ValidationBaseline:
        name: str
        enter: Any
        exit: Any

    method_adapter.MethodAdapter = MethodAdapter
    method_adapter.SetupCtx = SetupCtx
    method_adapter.StepCtx = StepCtx
    method_adapter.ValidationBaseline = ValidationBaseline

    hashing = types.ModuleType("library.training.hashing")

    def precalculate_safetensors_hashes(*_args, **_kwargs) -> tuple[str, str]:
        return "", ""

    hashing.precalculate_safetensors_hashes = precalculate_safetensors_hashes

    sys.modules.setdefault("library.training", training_pkg)
    sys.modules["library.training.method_adapter"] = method_adapter
    sys.modules.setdefault("library.training.hashing", hashing)


def looks_like_anima_dit(candidate: Any) -> bool:
    return (
        candidate is not None
        and hasattr(candidate, "blocks")
        and hasattr(candidate, "prepare_embedded_sequence")
        and hasattr(candidate, "unpatchify")
        and hasattr(candidate, "patch_spatial")
        and hasattr(candidate, "patch_temporal")
    )


def get_attr_path(root: Any, attr_path: str) -> tuple[Any, bool]:
    candidate = root
    for part in attr_path.split("."):
        if not hasattr(candidate, part):
            return None, False
        candidate = getattr(candidate, part)
    return candidate, True


def find_anima_diffusion_model(model_patcher: Any) -> Any:
    roots: list[Any] = []
    if hasattr(model_patcher, "model"):
        roots.append(model_patcher.model)
    roots.append(model_patcher)

    for root in roots:
        for path in DIFFUSION_ATTR_PATHS:
            candidate, ok = get_attr_path(root, path)
            if ok and looks_like_anima_dit(candidate):
                return candidate

    seen: set[int] = set()
    stack = roots[:]
    while stack and len(seen) < 256:
        candidate = stack.pop()
        if id(candidate) in seen:
            continue
        seen.add(id(candidate))
        if looks_like_anima_dit(candidate):
            return candidate
        for name in SEARCH_CHILD_ATTRS:
            if hasattr(candidate, name):
                try:
                    stack.append(getattr(candidate, name))
                except AttributeError:
                    pass
    raise RuntimeError("Could not find an Anima/MiniTrainDIT diffusion model in MODEL.")


def dtype_from_name(name: str) -> torch.dtype:
    if name == "float16":
        return torch.float16
    if name == "float32":
        return torch.float32
    return torch.bfloat16


def runtime_dtype(dtype: torch.dtype) -> torch.dtype:
    if dtype in (torch.float16, torch.bfloat16, torch.float32):
        return dtype
    return torch.bfloat16


def device_from_name(name: str) -> torch.device:
    if name == "cpu":
        return torch.device("cpu")
    if name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested for PE encoding but is not available.")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def comfy_image_to_minus1to1(image: torch.Tensor) -> tuple[torch.Tensor, tuple[int, int]]:
    if image.ndim != 4 or image.shape[-1] < 3:
        raise ValueError(f"Expected ComfyUI IMAGE [B,H,W,C>=3], got {tuple(image.shape)}")
    height, width = int(image.shape[1]), int(image.shape[2])
    nchw = image[:1, :, :, :3].detach().permute(0, 3, 1, 2).contiguous()
    return nchw.clamp(0.0, 1.0) * 2.0 - 1.0, (height, width)


def load_network(spec: AnimaPEIPAdapterSpec, strength: float):
    ensure_anima_root()
    install_anima_training_stubs()
    from networks.methods.ip_adapter import create_network_from_weights

    network, state = create_network_from_weights(
        strength,
        str(spec.path),
        ae=None,
        text_encoders=[],
        unet=None,
        for_inference=True,
    )
    missing, unexpected = network.load_state_dict(state, strict=False)
    real_missing = [key for key in missing if not key.startswith("_pe_inner.")]
    if real_missing or unexpected:
        raise RuntimeError(
            f"Failed to load PE IP-Adapter {spec.name}: "
            f"missing={real_missing}, unexpected={unexpected}"
        )
    network._centroid_active = None
    network.eval().requires_grad_(False)
    return network
