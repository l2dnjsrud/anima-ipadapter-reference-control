from __future__ import annotations

import tempfile
import time
from pathlib import Path
from typing import Final

import numpy as np
import torch
from PIL import Image

try:
    from .runner import (
        GenerationOptions,
        RunnerPaths,
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
        build_command,
        build_subprocess_env,
        newest_png,
        require_runner_inputs,
        resolved_paths,
        run_command,
    )


REPO_ROOT: Final[Path] = Path(__file__).resolve().parent
DEFAULT_ANIMA_ROOT: Final[Path] = Path("/home/wktwin/anima-lora-training-bundle/anima_lora")
DEFAULT_PYTHON: Final[Path] = DEFAULT_ANIMA_ROOT / ".venv/bin/python"
DEFAULT_DIT: Final[str] = "models/diffusion_models/anima-base-v1.0.safetensors"
DEFAULT_TEXT_ENCODER: Final[str] = "models/text_encoders/qwen_3_06b_base.safetensors"
DEFAULT_VAE: Final[str] = "models/vae/qwen_image_vae.safetensors"
DEFAULT_CHECKPOINT: Final[Path] = (
    REPO_ROOT / "checkpoints/anima_ip_adapter_quality_20260610.safetensors"
)
DEFAULT_OUTPUT_DIR: Final[Path] = DEFAULT_ANIMA_ROOT / "output/comfy_ipadapter"
DEFAULT_PROMPT: Final[str] = (
    "masterpiece, best quality, score_7, safe. manga panel layout, clean line art, "
    "expressive character, cinematic composition."
)
DEFAULT_NEGATIVE_PROMPT: Final[str] = "low quality, blurry, bad anatomy, text, watermark"


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
                "attn_mode": (["flash", "torch", "sageattn", "flex", "xformers"],),
                "sampler": (["er_sde", "euler", "lcm"],),
                "match_reference_size": ("BOOLEAN", {"default": False}),
                "anima_root": ("STRING", {"default": str(DEFAULT_ANIMA_ROOT)}),
                "python_executable": ("STRING", {"default": str(DEFAULT_PYTHON)}),
                "dit_path": ("STRING", {"default": DEFAULT_DIT}),
                "text_encoder_path": ("STRING", {"default": DEFAULT_TEXT_ENCODER}),
                "vae_path": ("STRING", {"default": DEFAULT_VAE}),
                "checkpoint_path": ("STRING", {"default": str(DEFAULT_CHECKPOINT)}),
                "output_dir": ("STRING", {"default": str(DEFAULT_OUTPUT_DIR)}),
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
        anima_root: str,
        python_executable: str,
        dit_path: str,
        text_encoder_path: str,
        vae_path: str,
        checkpoint_path: str,
        output_dir: str,
    ) -> tuple[torch.Tensor, str]:
        paths = resolved_paths(
            RunnerPaths(
                python_executable=Path(python_executable),
                anima_root=Path(anima_root),
                dit=Path(dit_path),
                text_encoder=Path(text_encoder_path),
                vae=Path(vae_path),
                checkpoint=Path(checkpoint_path),
                output_dir=Path(output_dir),
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
