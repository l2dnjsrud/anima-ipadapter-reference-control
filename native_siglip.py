from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Final
from typing import Mapping, Protocol

import numpy as np
import torch
from PIL import Image
from torch import nn

try:
    from .siglip_checkpoint import SigLIPCheckpointError, load_siglip_adapter
    from .siglip_model import IPAdapterSigLIP, SigLIPFeatures
except ImportError:
    from siglip_checkpoint import SigLIPCheckpointError, load_siglip_adapter
    from siglip_model import IPAdapterSigLIP, SigLIPFeatures


DEFAULT_SIGLIP_MODEL_ID = "google/siglip2-base-patch16-512"
DEFAULT_COMFY_MODELS_ROOT: Final[Path] = Path(
    os.environ.get("ANIMA_COMFY_MODELS_ROOT", "/data/ai/models")
)
DEFAULT_SIGLIP_ADAPTER_NAME: Final[str] = "anima_siglip_ip_adapter.safetensors"

try:
    import folder_paths
except ModuleNotFoundError:
    folder_paths = None

ExtraOptionValue = torch.Tensor | int | float | str | tuple[str, int] | tuple[str, str, int]
ExtraOptions = Mapping[str, ExtraOptionValue]


class ModelPatchTarget(Protocol):
    def clone(self) -> ModelPatchTarget: ...

    def set_model_attn2_patch(self, patch: nn.Module) -> None: ...


class SigLIPKVAttn2Patch(nn.Module):
    """ComfyUI attn2 patch that appends IP-Adapter key/value tokens."""

    def __init__(
        self,
        adapter: IPAdapterSigLIP,
        features: SigLIPFeatures,
        weight: float,
        start_at: float,
        end_at: float,
    ) -> None:
        super().__init__()
        self.adapter = adapter
        self.features = features
        self.weight = weight
        self.start_at = start_at
        self.end_at = end_at

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        extra_options: ExtraOptions,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        timestep = _timestep_from_options(extra_options, q.shape[0], q.device, q.dtype)
        if not _in_timestep_window(timestep, self.start_at, self.end_at):
            return q, k, v
        adapter = self.adapter.to(device=q.device, dtype=q.dtype)
        features = _features_to(self.features, q.device, q.dtype)
        token_timestep = _match_timestep_batch(timestep, features.deep.shape[0])
        tokens = adapter.encode_ref(features, timestep=token_timestep)
        block_idx = _block_index(extra_options, adapter.num_blocks)
        ip_k, ip_v = adapter.project_kv(block_idx, tokens, self.weight)
        if ip_k.shape[0] != k.shape[0]:
            ip_k = _match_batch(ip_k, k.shape[0])
            ip_v = _match_batch(ip_v, v.shape[0])
        return q, torch.cat([k, ip_k.to(k)], dim=1), torch.cat([v, ip_v.to(v)], dim=1)


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


class AnimaSigLIPIPAdapterLoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "ipadapter_name": (_model_names("ipadapter", DEFAULT_SIGLIP_ADAPTER_NAME),)
            }
        }

    RETURN_TYPES = ("ANIMA_SIGLIP_IPADAPTER",)
    RETURN_NAMES = ("ipadapter",)
    FUNCTION = "load"
    CATEGORY = "anima/ip-adapter"

    def load(self, ipadapter_name: str) -> tuple[IPAdapterSigLIP]:
        path = _model_path("ipadapter", ipadapter_name)
        if not path.is_file():
            raise SigLIPCheckpointError(f"SigLIP checkpoint not found: {path}")
        return (load_siglip_adapter(path),)


class AnimaSigLIPEncodeImage:
    @classmethod
    def INPUT_TYPES(cls) -> dict[str, dict[str, tuple[str, dict[str, str | bool]]]]:
        return {
            "required": {
                "image": ("IMAGE",),
                "siglip_model_id": ("STRING", {"default": DEFAULT_SIGLIP_MODEL_ID}),
                "include_shallow": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("SIGLIP_FEATURES",)
    RETURN_NAMES = ("siglip_features",)
    FUNCTION = "encode"
    CATEGORY = "anima/ip-adapter"

    def encode(
        self, image: torch.Tensor, siglip_model_id: str, include_shallow: bool
    ) -> tuple[SigLIPFeatures]:
        model, processor, device, dtype = self._vision_stack(siglip_model_id)
        pil_images = [_pad_to_square(img) for img in _comfy_images_to_pil(image)]
        inputs = processor(images=pil_images, return_tensors="pt", do_resize=False)
        inputs = {key: value.to(device=device, dtype=dtype) for key, value in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=include_shallow)
        deep = outputs.last_hidden_state.float().cpu()
        shallow = None
        if include_shallow:
            hidden_states = outputs.hidden_states
            shallow = torch.cat(
                [hidden_states[len(hidden_states) // 4], hidden_states[len(hidden_states) // 2]],
                dim=1,
            ).float().cpu()
        return (SigLIPFeatures(deep=deep, shallow=shallow),)

    def _vision_stack(self, model_id: str):
        cache_key = "_siglip_" + model_id.replace("/", "_")
        if not hasattr(self, cache_key):
            from transformers import AutoImageProcessor, SiglipVisionModel

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
            model = SiglipVisionModel.from_pretrained(
                model_id, torch_dtype=dtype, trust_remote_code=True
            ).to(device)
            model.eval()
            setattr(self, cache_key, (model, AutoImageProcessor.from_pretrained(model_id), device, dtype))
        return getattr(self, cache_key)


class AnimaSigLIPIPAdapterApply:
    @classmethod
    def INPUT_TYPES(cls) -> dict[str, dict[str, tuple[str, dict[str, float]]]]:
        return {
            "required": {
                "model": ("MODEL",),
                "ipadapter": ("ANIMA_SIGLIP_IPADAPTER",),
                "siglip_features": ("SIGLIP_FEATURES",),
                "weight": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 5.0, "step": 0.05}),
                "start_at": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "end_at": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "apply"
    CATEGORY = "anima/ip-adapter"

    def apply(
        self,
        model: ModelPatchTarget,
        ipadapter: IPAdapterSigLIP,
        siglip_features: SigLIPFeatures,
        weight: float,
        start_at: float,
        end_at: float,
    ) -> tuple[ModelPatchTarget]:
        patched = model.clone()
        patch = SigLIPKVAttn2Patch(ipadapter, siglip_features, weight, start_at, end_at)
        patched.set_model_attn2_patch(patch)
        return (patched,)


def _features_to(features: SigLIPFeatures, device: torch.device, dtype: torch.dtype) -> SigLIPFeatures:
    shallow = features.shallow.to(device=device, dtype=dtype) if features.shallow is not None else None
    return SigLIPFeatures(deep=features.deep.to(device=device, dtype=dtype), shallow=shallow)


def _timestep_from_options(
    extra_options: ExtraOptions, batch: int, device: torch.device, dtype: torch.dtype
) -> torch.Tensor:
    value = extra_options.get("timestep", extra_options.get("sigma", 0.5))
    if isinstance(value, torch.Tensor):
        timestep = value.flatten().to(device=device, dtype=dtype)
    elif isinstance(value, int | float):
        timestep = torch.full((batch,), float(value), device=device, dtype=dtype)
    else:
        timestep = torch.full((batch,), 0.5, device=device, dtype=dtype)
    return timestep[:1].expand(batch) if timestep.numel() == 1 else timestep[:batch]


def _in_timestep_window(timestep: torch.Tensor, start_at: float, end_at: float) -> bool:
    current = float(timestep.flatten()[0].detach().cpu())
    return start_at <= current <= end_at


def _block_index(extra_options: ExtraOptions, num_blocks: int) -> int:
    block = extra_options.get("block", 0)
    if isinstance(block, tuple):
        candidate = block[-1]
    else:
        candidate = block
    index = int(candidate) if isinstance(candidate, int) else 0
    return max(0, min(index, num_blocks - 1))


def _match_batch(tokens: torch.Tensor, batch: int) -> torch.Tensor:
    if tokens.shape[0] == batch:
        return tokens
    repeats = math.ceil(batch / tokens.shape[0])
    return tokens.repeat(repeats, 1, 1)[:batch]


def _match_timestep_batch(timestep: torch.Tensor, batch: int) -> torch.Tensor:
    if timestep.numel() == batch:
        return timestep
    return timestep[:1].expand(batch)


def _comfy_images_to_pil(image: torch.Tensor) -> list[Image.Image]:
    batch = image if image.ndim == 4 else image.unsqueeze(0)
    images: list[Image.Image] = []
    for idx in range(batch.shape[0]):
        array = (batch[idx, :, :, :3].detach().cpu().clamp(0, 1).numpy() * 255.0).round()
        images.append(Image.fromarray(array.astype(np.uint8), mode="RGB"))
    return images


def _pad_to_square(image: Image.Image, size: int = 512) -> Image.Image:
    width, height = image.size
    side = max(width, height)
    canvas = Image.new("RGB", (side, side), (255, 255, 255))
    canvas.paste(image, ((side - width) // 2, (side - height) // 2))
    return canvas.resize((size, size))


SIGLIP_NODE_CLASS_MAPPINGS = {
    "AnimaSigLIPIPAdapterLoader": AnimaSigLIPIPAdapterLoader,
    "AnimaSigLIPEncodeImage": AnimaSigLIPEncodeImage,
    "AnimaSigLIPIPAdapterApply": AnimaSigLIPIPAdapterApply,
}

SIGLIP_NODE_DISPLAY_NAME_MAPPINGS = {
    "AnimaSigLIPIPAdapterLoader": "Anima SigLIP IP-Adapter Loader",
    "AnimaSigLIPEncodeImage": "Anima SigLIP2 Encode Image",
    "AnimaSigLIPIPAdapterApply": "Anima SigLIP IP-Adapter Apply",
}

NODE_CLASS_MAPPINGS = SIGLIP_NODE_CLASS_MAPPINGS
NODE_DISPLAY_NAME_MAPPINGS = SIGLIP_NODE_DISPLAY_NAME_MAPPINGS
