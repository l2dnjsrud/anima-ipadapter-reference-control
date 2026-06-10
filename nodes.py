from __future__ import annotations

import os
import tempfile
import time
from pathlib import Path
from typing import Final

import folder_paths
import numpy as np
import torch
from PIL import Image

try:
    from .runner import (
        GenerationOptions,
        RunnerPaths,
        SUPPORTED_ATTN_MODES,
        build_command,
        build_subprocess_env,
        newest_png,
        require_runner_inputs,
        resolved_paths,
        run_command,
    )
except ImportError:
    from runner import (
        GenerationOptions,
        RunnerPaths,
        SUPPORTED_ATTN_MODES,
        build_command,
        build_subprocess_env,
        newest_png,
        require_runner_inputs,
        resolved_paths,
        run_command,
    )


DEFAULT_ANIMA_ROOT: Final[Path] = Path(
    os.environ.get("ANIMA_LORA_ROOT", "/home/wktwin/anima-lora-training-bundle/anima_lora")
)
DEFAULT_PYTHON: Final[Path] = Path(
    os.environ.get("ANIMA_LORA_PYTHON", str(DEFAULT_ANIMA_ROOT / ".venv/bin/python"))
)
DEFAULT_COMFY_MODELS_ROOT: Final[Path] = Path(
    os.environ.get("ANIMA_COMFY_MODELS_ROOT", "/data/ai/models")
)
DEFAULT_IPADAPTER_NAME: Final[str] = "anima_ip_adapter_quality_20260610.safetensors"
DEFAULT_DIT_NAME: Final[str] = "anima-base-v1.0.safetensors"
DEFAULT_TEXT_ENCODER_NAME: Final[str] = "qwen_3_06b_base.safetensors"
DEFAULT_VAE_NAME: Final[str] = "qwen/qwen_image_vae.safetensors"
DEFAULT_OUTPUT_SUBDIR: Final[str] = "comfy_ipadapter"
DEFAULT_PROMPT: Final[str] = (
    "masterpiece, best quality, score_7, safe. manga panel layout, clean line art, "
    "expressive character, cinematic composition."
)
DEFAULT_NEGATIVE_PROMPT: Final[str] = "low quality, blurry, bad anatomy, text, watermark"
SAMPLER_CHOICES: Final[tuple[str, ...]] = ("er_sde", "euler", "lcm")


def _ensure_model_folders() -> None:
    if "ipadapter" not in folder_paths.folder_names_and_paths:
        current_paths = [str(Path(folder_paths.models_dir) / "ipadapter")]
        folder_paths.folder_names_and_paths["ipadapter"] = (
            current_paths,
            folder_paths.supported_pt_extensions,
        )
    for folder_name in ("ipadapter", "diffusion_models", "text_encoders", "vae"):
        folder = DEFAULT_COMFY_MODELS_ROOT / folder_name
        if folder.exists():
            folder_paths.add_model_folder_path(folder_name, str(folder), is_default=True)


def _model_names(folder_name: str, preferred_name: str) -> list[str]:
    _ensure_model_folders()
    names = folder_paths.get_filename_list(folder_name)
    if preferred_name in names:
        return [preferred_name, *[name for name in names if name != preferred_name]]
    return names or [preferred_name]


def _model_path(folder_name: str, model_name: str) -> Path:
    _ensure_model_folders()
    return Path(folder_paths.get_full_path_or_raise(folder_name, model_name))


def _output_dir(output_subdir: str) -> Path:
    candidate = Path(output_subdir or DEFAULT_OUTPUT_SUBDIR).expanduser()
    if candidate.is_absolute():
        return candidate
    return DEFAULT_ANIMA_ROOT / "output" / candidate


class AnimaIPAdapterGenerate:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "reference_image": ("IMAGE",),
                "prompt": ("STRING", {"multiline": True, "default": DEFAULT_PROMPT}),
                "negative_prompt": (
                    "STRING",
                    {"multiline": True, "default": DEFAULT_NEGATIVE_PROMPT},
                ),
                "seed": ("INT", {"default": 20260610, "min": 0, "max": 2**63 - 1}),
                "height": ("INT", {"default": 1120, "min": 256, "max": 2048, "step": 16}),
                "width": ("INT", {"default": 960, "min": 256, "max": 2048, "step": 16}),
                "steps": ("INT", {"default": 20, "min": 1, "max": 100}),
                "guidance_scale": (
                    "FLOAT",
                    {"default": 3.5, "min": 1.0, "max": 20.0, "step": 0.1},
                ),
                "flow_shift": (
                    "FLOAT",
                    {"default": 3.0, "min": 0.0, "max": 10.0, "step": 0.1},
                ),
                "ip_scale": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": 3.0, "step": 0.05},
                ),
                "attn_mode": (list(SUPPORTED_ATTN_MODES),),
                "sampler": (list(SAMPLER_CHOICES),),
                "match_reference_size": ("BOOLEAN", {"default": False}),
                "ipadapter_name": (_model_names("ipadapter", DEFAULT_IPADAPTER_NAME),),
                "dit_name": (_model_names("diffusion_models", DEFAULT_DIT_NAME),),
                "text_encoder_name": (
                    _model_names("text_encoders", DEFAULT_TEXT_ENCODER_NAME),
                ),
                "vae_name": (_model_names("vae", DEFAULT_VAE_NAME),),
                "output_subdir": ("STRING", {"default": DEFAULT_OUTPUT_SUBDIR}),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("image", "output_path")
    FUNCTION = "generate"
    CATEGORY = "anima/ip-adapter"
    DESCRIPTION = (
        "Run the trained Anima PE-Core IP-Adapter reference-control checkpoint "
        "through the verified anima_lora inference path."
    )

    def generate(
        self,
        reference_image: torch.Tensor,
        prompt: str,
        negative_prompt: str,
        seed: int,
        height: int,
        width: int,
        steps: int,
        guidance_scale: float,
        flow_shift: float,
        ip_scale: float,
        attn_mode: str,
        sampler: str,
        match_reference_size: bool,
        ipadapter_name: str,
        dit_name: str,
        text_encoder_name: str,
        vae_name: str,
        output_subdir: str,
    ) -> tuple[torch.Tensor, str]:
        paths = resolved_paths(
            RunnerPaths(
                python_executable=DEFAULT_PYTHON,
                anima_root=DEFAULT_ANIMA_ROOT,
                dit=_model_path("diffusion_models", dit_name),
                text_encoder=_model_path("text_encoders", text_encoder_name),
                vae=_model_path("vae", vae_name),
                checkpoint=_model_path("ipadapter", ipadapter_name),
                output_dir=_output_dir(output_subdir),
            )
        )
        options = GenerationOptions(
            prompt=prompt,
            negative_prompt=negative_prompt,
            seed=int(seed),
            height=int(height),
            width=int(width),
            steps=int(steps),
            guidance_scale=float(guidance_scale),
            flow_shift=float(flow_shift),
            ip_scale=float(ip_scale),
            attn_mode=attn_mode,
            sampler=sampler,
            match_reference_size=bool(match_reference_size),
        )
        paths.output_dir.mkdir(parents=True, exist_ok=True)

        with tempfile.TemporaryDirectory(prefix="anima_ipadapter_ref_") as tmp_dir:
            reference_path = Path(tmp_dir) / "reference.png"
            _comfy_image_to_pil(reference_image).save(reference_path)
            require_runner_inputs(paths, reference_path)
            before_ns = time.time_ns()
            command = build_command(paths, options, reference_path)
            run_command(command, cwd=paths.anima_root, env=build_subprocess_env(paths.anima_root))
            output_path = newest_png(paths.output_dir, after_ns=before_ns)

        return (_pil_to_comfy_image(output_path), str(output_path))


def _comfy_image_to_pil(image: torch.Tensor) -> Image.Image:
    if image.ndim != 4 or image.shape[-1] < 3:
        raise RuntimeError(
            f"Expected ComfyUI IMAGE tensor [B,H,W,C>=3], got shape {tuple(image.shape)}"
        )
    pixels = image[:1, :, :, :3].detach().cpu().clamp(0.0, 1.0).squeeze(0)
    array = (pixels.numpy() * 255.0).round().astype(np.uint8)
    return Image.fromarray(array)


def _pil_to_comfy_image(path: Path) -> torch.Tensor:
    with Image.open(path) as image:
        rgb = image.convert("RGB")
        array = np.asarray(rgb, dtype=np.float32) / 255.0
    return torch.from_numpy(array).unsqueeze(0)


NODE_CLASS_MAPPINGS = {
    "AnimaIPAdapterGenerate": AnimaIPAdapterGenerate,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AnimaIPAdapterGenerate": "Anima IP-Adapter Generate",
}
