from __future__ import annotations

import os
from pathlib import Path
from typing import Final

import numpy as np
import torch
from PIL import Image

try:
    from .native_siglip_runtime import ModelPatcherLike, apply_siglip_adapter
    from .siglip_encoder_lora import apply_saved_siglip_lora
    from .siglip_checkpoint import SigLIPCheckpointError, load_siglip_adapter
    from .siglip_model import IPAdapterSigLIP, SigLIPFeatures
except ImportError:
    from native_siglip_runtime import ModelPatcherLike, apply_siglip_adapter
    from siglip_encoder_lora import apply_saved_siglip_lora
    from siglip_checkpoint import SigLIPCheckpointError, load_siglip_adapter
    from siglip_model import IPAdapterSigLIP, SigLIPFeatures


DEFAULT_SIGLIP_MODEL_ID = "google/siglip2-base-patch16-512"
DEFAULT_COMFY_MODELS_ROOT: Final[Path] = Path(
    os.environ.get("ANIMA_COMFY_MODELS_ROOT", "/data/ai/models")
)
DEFAULT_SIGLIP_ADAPTER_NAME: Final[str] = (
    "anima_siglip_ip_adapter_c089_shape_pe_teacher_0032_20260613.safetensors"
)
NO_SIGLIP_LORA_NAME: Final[str] = "none"

try:
    import folder_paths
except ModuleNotFoundError:
    folder_paths = None


def _ensure_model_folders() -> None:
    if folder_paths is None:
        return
    if "ipadapter" not in folder_paths.folder_names_and_paths:
        folder_paths.folder_names_and_paths["ipadapter"] = (
            [str(Path(folder_paths.models_dir) / "ipadapter")],
            folder_paths.supported_pt_extensions,
        )
    if "siglip_lora" not in folder_paths.folder_names_and_paths:
        folder_paths.folder_names_and_paths["siglip_lora"] = (
            [str(Path(folder_paths.models_dir) / "siglip_lora")],
            folder_paths.supported_pt_extensions,
        )
    folder = DEFAULT_COMFY_MODELS_ROOT / "ipadapter"
    if folder.exists():
        folder_paths.add_model_folder_path("ipadapter", str(folder), is_default=True)
    lora_folder = DEFAULT_COMFY_MODELS_ROOT / "siglip_lora"
    if lora_folder.exists():
        folder_paths.add_model_folder_path("siglip_lora", str(lora_folder), is_default=True)


def _model_names(folder_name: str, preferred_name: str) -> list[str]:
    if folder_paths is None:
        return [preferred_name]
    _ensure_model_folders()
    names = folder_paths.get_filename_list(folder_name)
    if preferred_name in names:
        return [preferred_name, *[name for name in names if name != preferred_name]]
    return names or [preferred_name]


def _optional_model_names(folder_name: str) -> list[str]:
    names = _model_names(folder_name, NO_SIGLIP_LORA_NAME)
    return [NO_SIGLIP_LORA_NAME, *[name for name in names if name != NO_SIGLIP_LORA_NAME]]


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
                "ipadapter_name": (
                    _model_names("ipadapter", DEFAULT_SIGLIP_ADAPTER_NAME),
                )
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
                "encoder_lora_name": (_optional_model_names("siglip_lora"),),
                "include_shallow": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("SIGLIP_FEATURES",)
    RETURN_NAMES = ("siglip_features",)
    FUNCTION = "encode"
    CATEGORY = "anima/ip-adapter"

    def encode(
        self,
        image: torch.Tensor,
        siglip_model_id: str,
        encoder_lora_name: str,
        include_shallow: bool,
    ) -> tuple[SigLIPFeatures]:
        model, processor, device, dtype = self._vision_stack(siglip_model_id, encoder_lora_name)
        pil_images = [_pad_to_square(img) for img in _comfy_images_to_pil(image)]
        inputs = processor(images=pil_images, return_tensors="pt", do_resize=False)
        inputs = {
            key: value.to(device=device, dtype=dtype) for key, value in inputs.items()
        }
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=include_shallow)
        deep = outputs.last_hidden_state.float().cpu()
        shallow = None
        if include_shallow:
            hidden_states = outputs.hidden_states
            shallow = (
                torch.cat(
                    [
                        hidden_states[len(hidden_states) // 4],
                        hidden_states[len(hidden_states) // 2],
                    ],
                    dim=1,
                )
                .float()
                .cpu()
            )
        return (SigLIPFeatures(deep=deep, shallow=shallow),)

    def _vision_stack(self, model_id: str, encoder_lora_name: str):
        lora_key = encoder_lora_name.replace("/", "_").replace(".", "_")
        cache_key = "_siglip_" + model_id.replace("/", "_") + "_" + lora_key
        if not hasattr(self, cache_key):
            from transformers import AutoImageProcessor, SiglipVisionModel

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
            model = SiglipVisionModel.from_pretrained(
                model_id, torch_dtype=dtype, trust_remote_code=True
            ).to(device)
            if encoder_lora_name != NO_SIGLIP_LORA_NAME:
                apply_saved_siglip_lora(model, _model_path("siglip_lora", encoder_lora_name))
            model.eval()
            processor = AutoImageProcessor.from_pretrained(model_id)
            setattr(self, cache_key, (model, processor, device, dtype))
        return getattr(self, cache_key)


class AnimaSigLIPIPAdapterApply:
    @classmethod
    def INPUT_TYPES(cls) -> dict[str, dict[str, tuple[str, dict[str, float]]]]:
        return {
            "required": {
                "model": ("MODEL",),
                "ipadapter": ("ANIMA_SIGLIP_IPADAPTER",),
                "siglip_features": ("SIGLIP_FEATURES",),
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
        ipadapter: IPAdapterSigLIP,
        siglip_features: SigLIPFeatures,
        weight: float,
        start_at: float,
        end_at: float,
    ) -> tuple[ModelPatcherLike]:
        return (
            apply_siglip_adapter(
                model, ipadapter, siglip_features, weight, start_at, end_at
            ),
        )


def _comfy_images_to_pil(image: torch.Tensor) -> list[Image.Image]:
    batch = image if image.ndim == 4 else image.unsqueeze(0)
    images: list[Image.Image] = []
    for idx in range(batch.shape[0]):
        array = (
            batch[idx, :, :, :3].detach().cpu().clamp(0, 1).numpy() * 255.0
        ).round()
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
