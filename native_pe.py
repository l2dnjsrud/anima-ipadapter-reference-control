from __future__ import annotations

from typing import Any

import torch

try:
    from .native_pe_models import (
        DEFAULT_IPADAPTER_NAME,
        DEVICE_CHOICES,
        DTYPE_CHOICES,
        SUPPORTED_ENCODERS,
        AnimaPEFeatures,
        AnimaPEIPAdapterSpec,
        load_pe_adapter_spec,
        model_names,
        model_path,
    )
    from .native_pe_patch import patch_network_to_comfy_attention
    from .native_pe_runtime import (
        comfy_image_to_minus1to1,
        device_from_name,
        dtype_from_name,
        ensure_anima_root,
        find_anima_diffusion_model,
        load_network,
        runtime_dtype,
    )
except ImportError:
    from native_pe_models import (
        DEFAULT_IPADAPTER_NAME,
        DEVICE_CHOICES,
        DTYPE_CHOICES,
        SUPPORTED_ENCODERS,
        AnimaPEFeatures,
        AnimaPEIPAdapterSpec,
        load_pe_adapter_spec,
        model_names,
        model_path,
    )
    from native_pe_patch import patch_network_to_comfy_attention
    from native_pe_runtime import (
        comfy_image_to_minus1to1,
        device_from_name,
        dtype_from_name,
        ensure_anima_root,
        find_anima_diffusion_model,
        load_network,
        runtime_dtype,
    )


_PE_ENCODER_CACHE: dict[tuple[str, str, str], Any] = {}

_model_names = model_names
_model_path = model_path
_ensure_anima_root = ensure_anima_root
_dtype_from_name = dtype_from_name
_runtime_dtype = runtime_dtype
_device_from_name = device_from_name
_comfy_image_to_minus1to1 = comfy_image_to_minus1to1
_load_network = load_network


class AnimaPEIPAdapterLoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "ipadapter_name": (_model_names("ipadapter", DEFAULT_IPADAPTER_NAME),),
            }
        }

    RETURN_TYPES = ("ANIMA_PE_IPADAPTER",)
    RETURN_NAMES = ("adapter",)
    FUNCTION = "load"
    CATEGORY = "anima/ip-adapter"
    DESCRIPTION = "Load a PE-Core Anima IP-Adapter checkpoint for native MODEL patching."

    def load(self, ipadapter_name: str):
        path = _model_path("ipadapter", ipadapter_name)
        return (load_pe_adapter_spec(path, ipadapter_name),)


class AnimaPEEncodeImage:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "encoder_name": (list(SUPPORTED_ENCODERS),),
                "device": (list(DEVICE_CHOICES),),
                "dtype": (list(DTYPE_CHOICES),),
                "cache_encoder": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("ANIMA_PE_FEATURES",)
    RETURN_NAMES = ("features",)
    FUNCTION = "encode"
    CATEGORY = "anima/ip-adapter"
    DESCRIPTION = "Encode a reference image with PE-Core for Anima IP-Adapter Apply."

    def encode(
        self,
        image: torch.Tensor,
        encoder_name: str,
        device: str,
        dtype: str,
        cache_encoder: bool,
    ):
        _ensure_anima_root()
        from library.vision import encode_pe_from_imageminus1to1, load_pe_encoder

        resolved_device = _device_from_name(device)
        resolved_dtype = _dtype_from_name(dtype)
        cache_key = (encoder_name, str(resolved_device), str(resolved_dtype))
        bundle = _PE_ENCODER_CACHE.get(cache_key) if cache_encoder else None
        if bundle is None:
            bundle = load_pe_encoder(
                resolved_device,
                name=encoder_name,
                dtype=resolved_dtype,
            )
            if cache_encoder:
                _PE_ENCODER_CACHE[cache_key] = bundle

        image_pm1, source_size = _comfy_image_to_minus1to1(image)
        with torch.no_grad():
            feats = encode_pe_from_imageminus1to1(
                bundle,
                image_pm1.to(device=resolved_device, dtype=resolved_dtype),
                same_bucket=True,
            )
        features = torch.stack(feats, dim=0).detach().cpu().float()
        return (AnimaPEFeatures(features=features, encoder_name=encoder_name, source_size=source_size),)


class AnimaPEIPAdapterApply:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "adapter": ("ANIMA_PE_IPADAPTER",),
                "features": ("ANIMA_PE_FEATURES",),
                "strength": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.05}),
                "start_percent": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.001}),
                "end_percent": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.001}),
                "preserve_wrapper": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "apply"
    CATEGORY = "anima/ip-adapter"
    DESCRIPTION = "Patch a ComfyUI MODEL with PE-Core Anima IP-Adapter reference features."

    def apply(
        self,
        model: Any,
        adapter: AnimaPEIPAdapterSpec,
        features: AnimaPEFeatures,
        strength: float,
        start_percent: float,
        end_percent: float,
        preserve_wrapper: bool,
    ):
        expected_dim = int(adapter.metadata["ss_encoder_dim"])
        if features.encoder_name != adapter.metadata.get("ss_encoder"):
            raise ValueError(
                f"Feature encoder {features.encoder_name!r} does not match adapter "
                f"encoder {adapter.metadata.get('ss_encoder')!r}."
            )
        if features.features.ndim != 3 or int(features.features.shape[-1]) != expected_dim:
            raise ValueError(
                f"PE features must be [B,T,{expected_dim}], got {tuple(features.features.shape)}"
            )

        dit = find_anima_diffusion_model(model)
        network = _load_network(adapter, float(strength))
        model_sampling = model.get_model_object("model_sampling")
        sigma_start = float(model_sampling.percent_to_sigma(float(start_percent)))
        sigma_end = float(model_sampling.percent_to_sigma(float(end_percent)))
        old_wrapper = model.model_options.get("model_function_wrapper")
        source_features = features.features.detach().clone()
        runtime: dict[str, Any] = {"loaded_to": None}

        def call_next(apply_model, args):
            if preserve_wrapper and old_wrapper is not None:
                return old_wrapper(apply_model, args)
            return apply_model(args["input"], args["timestep"], **args["c"])

        def wrapper(apply_model, args):
            input_x = args["input"]
            timestep = args["timestep"]
            sigma = float(timestep.max().item()) if torch.is_tensor(timestep) else float(timestep)
            if strength == 0.0 or not (sigma_end <= sigma <= sigma_start):
                return call_next(apply_model, args)

            device = input_x.device
            dtype = _runtime_dtype(input_x.dtype)
            tag = (device, dtype)
            if runtime["loaded_to"] != tag:
                network.to(device=device, dtype=dtype)
                runtime["loaded_to"] = tag

            patch_network_to_comfy_attention(network, dit)
            try:
                ip_features = source_features.to(device=device, dtype=dtype)
                ip_tokens = network.encode_ip_tokens(ip_features)
                network.set_ip_tokens(ip_tokens)
                return call_next(apply_model, args)
            finally:
                network.clear_ip_tokens()
                network.remove_from()

        patched = model.clone()
        patched.set_model_unet_function_wrapper(wrapper)
        return (patched,)


NODE_CLASS_MAPPINGS = {
    "AnimaPEIPAdapterLoader": AnimaPEIPAdapterLoader,
    "AnimaPEEncodeImage": AnimaPEEncodeImage,
    "AnimaPEIPAdapterApply": AnimaPEIPAdapterApply,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AnimaPEIPAdapterLoader": "Anima PE IP-Adapter Loader",
    "AnimaPEEncodeImage": "Anima PE Encode Image",
    "AnimaPEIPAdapterApply": "Anima PE IP-Adapter Apply",
}
