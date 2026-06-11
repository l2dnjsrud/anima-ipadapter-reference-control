from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Final

import numpy as np
import torch
from PIL import Image

try:
    from .native_pe_runtime import find_anima_diffusion_model, runtime_dtype
    from .native_siglip_runtime import (
        ModelPatcherLike,
        patch_siglip_to_comfy_attention,
        remove_siglip_patches,
    )
    from .qwenvl_checkpoint import QwenVLCheckpointError, load_qwenvl_adapter
    from .qwenvl_model import IPAdapterQwenVL
except ImportError:
    from native_pe_runtime import find_anima_diffusion_model, runtime_dtype
    from native_siglip_runtime import (
        ModelPatcherLike,
        patch_siglip_to_comfy_attention,
        remove_siglip_patches,
    )
    from qwenvl_checkpoint import QwenVLCheckpointError, load_qwenvl_adapter
    from qwenvl_model import IPAdapterQwenVL


DEFAULT_QWENVL_MODEL_ID: Final[str] = "Qwen/Qwen3-VL-Embedding-2B"
DEFAULT_QWENVL_ADAPTER_NAME: Final[str] = "anima_qwenvl_ip_adapter.safetensors"
DEFAULT_QWENVL_INSTRUCTION: Final[str] = (
    "Represent this manhwa/anime reference image for visual style, color palette, "
    "composition, character identity, and panel layout."
)
DEFAULT_COMFY_MODELS_ROOT: Final[Path] = Path(
    os.environ.get("ANIMA_COMFY_MODELS_ROOT", "/data/ai/models")
)

try:
    import folder_paths
except ModuleNotFoundError:
    folder_paths = None


@dataclass(slots=True)
class _RuntimeState:
    loaded_to: tuple[torch.device, torch.dtype] | None = None


def _ensure_model_folders() -> None:
    if folder_paths is None:
        return
    if "ipadapter" not in folder_paths.folder_names_and_paths:
        folder_paths.folder_names_and_paths["ipadapter"] = (
            [str(Path(folder_paths.models_dir) / "ipadapter")],
            folder_paths.supported_pt_extensions,
        )
    folder = DEFAULT_COMFY_MODELS_ROOT / "ipadapter"
    if folder.exists():
        folder_paths.add_model_folder_path("ipadapter", str(folder), is_default=True)


def _model_names(folder_name: str, preferred_name: str) -> list[str]:
    if folder_paths is None:
        return [preferred_name]
    _ensure_model_folders()
    names = folder_paths.get_filename_list(folder_name)
    if preferred_name in names:
        return [preferred_name, *[name for name in names if name != preferred_name]]
    return names or [preferred_name]


def _model_path(folder_name: str, model_name: str) -> Path:
    if folder_paths is None:
        return DEFAULT_COMFY_MODELS_ROOT / folder_name / model_name
    _ensure_model_folders()
    return Path(folder_paths.get_full_path_or_raise(folder_name, model_name))


class AnimaQwenVLIPAdapterLoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "ipadapter_name": (
                    _model_names("ipadapter", DEFAULT_QWENVL_ADAPTER_NAME),
                )
            }
        }

    RETURN_TYPES = ("ANIMA_QWENVL_IPADAPTER",)
    RETURN_NAMES = ("ipadapter",)
    FUNCTION = "load"
    CATEGORY = "anima/ip-adapter"

    def load(self, ipadapter_name: str) -> tuple[IPAdapterQwenVL]:
        path = _model_path("ipadapter", ipadapter_name)
        if not path.is_file():
            raise QwenVLCheckpointError(f"QwenVL checkpoint not found: {path}")
        return (load_qwenvl_adapter(path),)


class AnimaQwenVLEncodeImage:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "qwenvl_model_id": ("STRING", {"default": DEFAULT_QWENVL_MODEL_ID}),
                "instruction": (
                    "STRING",
                    {"default": DEFAULT_QWENVL_INSTRUCTION, "multiline": True},
                ),
            }
        }

    RETURN_TYPES = ("QWENVL_EMBEDDING",)
    RETURN_NAMES = ("embedding",)
    FUNCTION = "encode"
    CATEGORY = "anima/ip-adapter"

    def encode(
        self,
        image: torch.Tensor,
        qwenvl_model_id: str,
        instruction: str,
    ) -> tuple[torch.Tensor]:
        model = self._embedding_stack(qwenvl_model_id)
        inputs = [{"image": pil_image} for pil_image in _comfy_images_to_pil(image)]
        with torch.no_grad():
            embedding = model.encode(
                inputs,
                normalize_embeddings=True,
                convert_to_tensor=True,
                prompt=instruction,
            )
        return (torch.as_tensor(embedding).float().cpu(),)

    def _embedding_stack(self, model_id: str):
        cache_key = "_qwenvl_" + model_id.replace("/", "_")
        if not hasattr(self, cache_key):
            from sentence_transformers import SentenceTransformer

            device = "cuda" if torch.cuda.is_available() else "cpu"
            model = SentenceTransformer(
                model_id,
                device=device,
                model_kwargs={"torch_dtype": "bfloat16"} if device == "cuda" else {},
            )
            setattr(self, cache_key, model)
        return getattr(self, cache_key)


class AnimaQwenVLIPAdapterApply:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "ipadapter": ("ANIMA_QWENVL_IPADAPTER",),
                "embedding": ("QWENVL_EMBEDDING",),
                "weight": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": 5.0, "step": 0.05},
                ),
                "start_at": (
                    "FLOAT",
                    {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01},
                ),
                "end_at": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01},
                ),
            }
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "apply"
    CATEGORY = "anima/ip-adapter"

    def apply(
        self,
        model: ModelPatcherLike,
        ipadapter: IPAdapterQwenVL,
        embedding: torch.Tensor,
        weight: float,
        start_at: float,
        end_at: float,
    ) -> tuple[ModelPatcherLike]:
        return (apply_qwenvl_adapter(model, ipadapter, embedding, weight, start_at, end_at),)


def apply_qwenvl_adapter(
    model: ModelPatcherLike,
    adapter: IPAdapterQwenVL,
    embedding: torch.Tensor,
    weight: float,
    start_percent: float,
    end_percent: float,
) -> ModelPatcherLike:
    dit = find_anima_diffusion_model(model)
    model_sampling = model.get_model_object("model_sampling")
    sigma_start = float(model_sampling.percent_to_sigma(float(start_percent)))
    sigma_end = float(model_sampling.percent_to_sigma(float(end_percent)))
    old_wrapper = model.model_options.get("model_function_wrapper")
    source_embedding = embedding.detach().clone()
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
        sigma = float(timestep.max().item()) if torch.is_tensor(timestep) else float(timestep)
        if weight == 0.0 or not (sigma_end <= sigma <= sigma_start):
            return call_next(apply_model, args)
        device = input_x.device
        dtype = torch.bfloat16 if device.type == "cuda" else runtime_dtype(input_x.dtype)
        target = (device, dtype)
        if runtime.loaded_to != target:
            adapter.to(device=device, dtype=dtype)
            runtime.loaded_to = target
        prepared = source_embedding.to(device=device, dtype=dtype)
        tokens = adapter.encode_ref(
            prepared,
            timestep=_timestep_batch(timestep, prepared.shape[0], device, dtype),
        )
        handle = patch_siglip_to_comfy_attention(adapter, dit, tokens, weight)
        try:
            return call_next(apply_model, args)
        finally:
            remove_siglip_patches(handle)

    patched = model.clone()
    patched.set_model_unet_function_wrapper(wrapper)
    return patched


def _comfy_images_to_pil(image: torch.Tensor) -> list[Image.Image]:
    batch = image if image.ndim == 4 else image.unsqueeze(0)
    images: list[Image.Image] = []
    for idx in range(batch.shape[0]):
        array = (
            batch[idx, :, :, :3].detach().cpu().clamp(0, 1).numpy() * 255.0
        ).round()
        images.append(Image.fromarray(array.astype(np.uint8), mode="RGB"))
    return images


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


QWENVL_NODE_CLASS_MAPPINGS = {
    "AnimaQwenVLIPAdapterLoader": AnimaQwenVLIPAdapterLoader,
    "AnimaQwenVLEncodeImage": AnimaQwenVLEncodeImage,
    "AnimaQwenVLIPAdapterApply": AnimaQwenVLIPAdapterApply,
}

QWENVL_NODE_DISPLAY_NAME_MAPPINGS = {
    "AnimaQwenVLIPAdapterLoader": "Anima QwenVL IP-Adapter Loader",
    "AnimaQwenVLEncodeImage": "Anima Qwen3-VL Encode Image",
    "AnimaQwenVLIPAdapterApply": "Anima QwenVL IP-Adapter Apply",
}

NODE_CLASS_MAPPINGS = QWENVL_NODE_CLASS_MAPPINGS
NODE_DISPLAY_NAME_MAPPINGS = QWENVL_NODE_DISPLAY_NAME_MAPPINGS
