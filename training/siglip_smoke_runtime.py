from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import torch

from siglip_model import SigLIPFeatures
from training.siglip_smoke_data import load_anima_pixels, load_siglip_image
from training.siglip_smoke_types import (
    MAX_PILOT_ROWS,
    MAX_PILOT_STEPS,
    SmokeConfig,
    SmokeInputError,
)


def encode_target_latents(
    vae,
    path: Path,
    resolution: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    pixels = load_anima_pixels(path, resolution=resolution).to(
        device=device, dtype=dtype
    )
    with torch.no_grad():
        return vae.encode_pixels_to_latents(pixels).to(device=device, dtype=dtype)


def encode_prompt(
    prompt: str,
    text_encoder_path: Path,
    text_encoder: torch.nn.Module,
    anima: torch.nn.Module,
    prepare_text_inputs,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    args = SimpleNamespace(
        prompt=prompt,
        negative_prompt="",
        text_encoder=str(text_encoder_path),
        text_encoder_cpu=False,
    )
    cond, _null = prepare_text_inputs(
        args, device, anima, shared_models={"text_encoder": text_encoder}
    )
    return cond["embed"][0].to(device=device, dtype=dtype)


def encode_siglip_features(
    model: torch.nn.Module,
    processor,
    image_path: Path,
    device: torch.device,
    dtype: torch.dtype,
) -> SigLIPFeatures:
    image = load_siglip_image(image_path)
    inputs = processor(images=[image], return_tensors="pt", do_resize=False)
    inputs = {
        key: value.to(device=device, dtype=dtype) for key, value in inputs.items()
    }
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    hidden_states = outputs.hidden_states
    shallow = torch.cat(
        [
            hidden_states[len(hidden_states) // 4],
            hidden_states[len(hidden_states) // 2],
        ],
        dim=1,
    )
    return SigLIPFeatures(deep=outputs.last_hidden_state, shallow=shallow)


def noise_args() -> SimpleNamespace:
    return SimpleNamespace(
        timestep_sampling="uniform",
        sigmoid_bias=0.0,
        sigmoid_scale=1.0,
        discrete_flow_shift=1.0,
        weighting_scheme="none",
        logit_mean=0.0,
        logit_std=1.0,
        mode_scale=1.29,
        t_min=None,
        t_max=None,
        ip_noise_gamma=0.0,
        ip_noise_gamma_random_strength=False,
    )


def validate_config(config: SmokeConfig) -> None:
    if config.steps < 1:
        raise SmokeInputError("steps must be >= 1")
    if config.steps > MAX_PILOT_STEPS:
        raise SmokeInputError(
            f"steps must be <= {MAX_PILOT_STEPS} for bounded pilot runs"
        )
    if config.max_rows < 1:
        raise SmokeInputError("max_rows must be >= 1")
    if config.max_rows > MAX_PILOT_ROWS:
        raise SmokeInputError(
            f"max_rows must be <= {MAX_PILOT_ROWS} for bounded pilot runs"
        )
    required_paths = (
        config.manifest_path,
        config.image_root,
        config.dit_path,
        config.text_encoder_path,
        config.vae_path,
    )
    for path in required_paths:
        if not path.exists():
            raise SmokeInputError(f"required path not found: {path}")
    if (
        config.init_checkpoint_path is not None
        and not config.init_checkpoint_path.is_file()
    ):
        raise SmokeInputError(
            f"init checkpoint not found: {config.init_checkpoint_path}"
        )


def seed_everything(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
