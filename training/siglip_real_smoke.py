# /// script
# dependencies = [
#   "numpy",
#   "pillow",
#   "safetensors",
#   "torch",
#   "transformers",
#   "typer",
#   "rich",
# ]
# ///
# --- How to run -----------------------------------------------------
# /home/wktwin/anima-lora-training-bundle/anima_lora/.venv/bin/python training/siglip_real_smoke_cli.py \
#   --manifest-path training/manifests/local_color_pairs_pilot_20260610.jsonl \
#   --image-root /home/wktwin/anima-lora-training-bundle/image_dataset_color_panel_style_v5_best \
#   --steps 1 --resolution 256 --device cuda:0

from __future__ import annotations

import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
ANIMA_ROOT = Path("/home/wktwin/anima-lora-training-bundle/anima_lora")
for candidate in (ROOT, ANIMA_ROOT):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from siglip_model import IPAdapterSigLIP, SigLIPFeatures  # noqa: E402
from training.siglip_smoke_checkpoint import (  # noqa: E402
    load_trainable_adapter,
    save_adapter_checkpoint,
    trainable_adapter_parameters,
    verify_checkpoint,
)
from training.siglip_smoke_data import load_pair_rows, resolve_pair_paths  # noqa: E402
from training.siglip_smoke_patch import patched_cross_attention  # noqa: E402
from training.siglip_smoke_runtime import (  # noqa: E402
    encode_prompt,
    encode_siglip_features,
    encode_target_latents,
    noise_args,
    seed_everything,
    validate_config,
)
from training.siglip_smoke_types import (  # noqa: E402
    SmokeConfig,
    SmokeInputError,
    SmokeSummary,
    PairRow,
)


DEFAULT_DIT = ANIMA_ROOT / "models/diffusion_models/anima-base-v1.0.safetensors"
DEFAULT_TEXT = ANIMA_ROOT / "models/text_encoders/qwen_3_06b_base.safetensors"
DEFAULT_VAE = ANIMA_ROOT / "models/vae/qwen_image_vae.safetensors"
DEFAULT_OUTPUT = Path(
    "/data/ai/models/ipadapter/anima_siglip_ip_adapter_smoke_20260610.safetensors"
)
DEFAULT_PE = Path(
    "/data/ai/models/ipadapter/anima_ip_adapter_quality_20260610.safetensors"
)
DEFAULT_SIGLIP = "google/siglip2-base-patch16-512"
PREPARED_ROW_CACHE_LIMIT = 128


@dataclass(slots=True)
class PreparedTrainingRow:
    latents: torch.Tensor
    crossattn_emb: torch.Tensor
    features: SigLIPFeatures


def freeze_module(module: torch.nn.Module) -> int:
    module.eval()
    total = 0
    for parameter in module.parameters():
        total += parameter.numel()
        parameter.requires_grad_(False)
    return total


def trainable_parameter_count(module: torch.nn.Module) -> int:
    return sum(
        parameter.numel()
        for parameter in module.parameters()
        if parameter.requires_grad
    )


def prepare_training_row(
    row: PairRow,
    config: SmokeConfig,
    vae,
    text_encoder: torch.nn.Module,
    anima: torch.nn.Module,
    siglip: torch.nn.Module,
    processor,
    prepare_text_inputs,
    device: torch.device,
    dtype: torch.dtype,
) -> PreparedTrainingRow:
    paths = resolve_pair_paths(row, config.image_root)
    latents = encode_target_latents(
        vae, paths.target_image, config.resolution, device, dtype
    ).detach()
    crossattn_emb = encode_prompt(
        row.prompt,
        config.text_encoder_path,
        text_encoder,
        anima,
        prepare_text_inputs,
        device,
        dtype,
    ).detach()
    features = encode_siglip_features(
        siglip,
        processor,
        paths.ref_image,
        device,
        dtype,
    )
    return PreparedTrainingRow(
        latents=latents,
        crossattn_emb=crossattn_emb,
        features=_detach_features(features),
    )


def _detach_features(features: SigLIPFeatures) -> SigLIPFeatures:
    shallow = features.shallow.detach() if features.shallow is not None else None
    return SigLIPFeatures(deep=features.deep.detach(), shallow=shallow)


def run_real_smoke(config: SmokeConfig) -> SmokeSummary:
    validate_config(config)
    seed_everything(config.seed)
    device = torch.device(config.device)
    dtype = torch.float32
    rows = load_pair_rows(config.manifest_path, limit=config.max_rows)
    random.Random(config.seed).shuffle(rows)

    from library.anima.weights import load_anima_model, load_qwen3_text_encoder
    from library.inference.text import prepare_text_inputs
    from library.models.qwen_vae import load_vae
    from library.runtime.noise import (
        FlowMatchEulerDiscreteScheduler,
        get_noisy_model_input_and_timesteps,
    )
    from transformers import AutoImageProcessor, SiglipVisionModel

    anima = load_anima_model(device, str(config.dit_path), "torch", device, dtype)
    anima.to(device=device, dtype=dtype)
    frozen_params = freeze_module(anima)
    vae = load_vae(str(config.vae_path), device=device, dtype=dtype, eval=True)
    frozen_params += freeze_module(vae)
    text_encoder, _ = load_qwen3_text_encoder(
        str(config.text_encoder_path), dtype=dtype, device=str(device)
    )
    frozen_params += freeze_module(text_encoder)
    siglip = SiglipVisionModel.from_pretrained(
        config.siglip_model_id,
        torch_dtype=dtype,
        trust_remote_code=True,
    ).to(device)
    processor = AutoImageProcessor.from_pretrained(config.siglip_model_id)
    frozen_params += freeze_module(siglip)

    adapter = load_trainable_adapter(config, device, dtype)
    optimizer = torch.optim.AdamW(trainable_adapter_parameters(adapter), lr=config.lr)
    scheduler = FlowMatchEulerDiscreteScheduler(num_train_timesteps=1000, shift=1.0)
    losses: list[float] = []
    prepared_cache = (
        [
            prepare_training_row(
                row,
                config,
                vae,
                text_encoder,
                anima,
                siglip,
                processor,
                prepare_text_inputs,
                device,
                dtype,
            )
            for row in rows
        ]
        if len(rows) <= PREPARED_ROW_CACHE_LIMIT
        else None
    )

    for step in range(config.steps):
        row_index = step % len(rows)
        prepared = (
            prepared_cache[row_index]
            if prepared_cache is not None
            else prepare_training_row(
                rows[row_index],
                config,
                vae,
                text_encoder,
                anima,
                siglip,
                processor,
                prepare_text_inputs,
                device,
                dtype,
            )
        )
        latents = prepared.latents
        noise = torch.randn_like(latents)
        noisy, timesteps, _sigmas = get_noisy_model_input_and_timesteps(
            noise_args(),
            scheduler,
            latents,
            noise,
            device,
            dtype,
        )
        image_tokens = adapter.encode_ref(prepared.features, timestep=timesteps)
        padding_mask = torch.zeros(
            latents.shape[0],
            1,
            latents.shape[-2],
            latents.shape[-1],
            device=device,
            dtype=dtype,
        )

        optimizer.zero_grad(set_to_none=True)
        with patched_cross_attention(anima, adapter, image_tokens):
            model_pred = anima(
                noisy.unsqueeze(2),
                timesteps,
                prepared.crossattn_emb,
                padding_mask=padding_mask,
            )
        target = noise - latents
        loss = torch.nn.functional.mse_loss(
            model_pred.squeeze(2).float(), target.float()
        )
        if not torch.isfinite(loss):
            raise SmokeInputError(
                f"non-finite loss at step {step}: {float(loss.detach().cpu())}"
            )
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu()))

    save_adapter_checkpoint(adapter, config.output_path)
    checkpoint = verify_checkpoint(config.output_path, config.pe_checkpoint_path)
    return SmokeSummary(
        steps=config.steps,
        rows_loaded=len(rows),
        first_loss=losses[0],
        final_loss=losses[-1],
        mean_loss=sum(losses) / len(losses),
        finite_loss=all(math.isfinite(loss) for loss in losses),
        loss_history=tuple(losses),
        trainable_parameters=trainable_parameter_count(adapter),
        frozen_base_parameters=frozen_params,
        checkpoint=checkpoint,
        init_checkpoint_path=str(config.init_checkpoint_path)
        if config.init_checkpoint_path
        else None,
    )


if __name__ == "__main__":
    from training.siglip_real_smoke_cli import app

    app()
