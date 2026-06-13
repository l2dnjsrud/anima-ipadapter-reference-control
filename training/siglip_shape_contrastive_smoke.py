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

from training.latent_shape_loss import (  # noqa: E402
    denoised_latents_from_velocity,
    latent_edge_projection_loss,
)
from training.siglip_contrastive_smoke import _predict  # noqa: E402
from training.siglip_prepared_cache import get_prepared, get_wrong_prepared, prepare_cache  # noqa: E402
from training.siglip_real_smoke import (  # noqa: E402
    freeze_module,
    load_trainable_adapter,
    save_adapter_checkpoint,
    trainable_parameter_count,
    verify_checkpoint,
)
from training.siglip_reference_loss import reference_margin_loss  # noqa: E402
from training.siglip_shape_cache import (  # noqa: E402
    get_reference_latents,
    prepare_reference_latent_cache,
)
from training.siglip_smoke_data import load_pair_rows  # noqa: E402
from training.siglip_smoke_runtime import noise_args, seed_everything, validate_config  # noqa: E402
from training.siglip_smoke_types import (  # noqa: E402
    CheckpointVerification,
    SmokeConfig,
    SmokeInputError,
)


@dataclass(frozen=True, slots=True)
class ShapeContrastiveSmokeSummary:
    steps: int
    rows_loaded: int
    first_loss: float
    final_loss: float
    mean_loss: float
    mean_base_loss: float
    mean_contrastive_loss: float
    mean_shape_loss: float
    finite_loss: bool
    explicit_negative_rows: int
    trainable_parameters: int
    frozen_base_parameters: int
    checkpoint: CheckpointVerification
    init_checkpoint_path: str | None
    contrastive_weight: float
    contrastive_margin: float
    shape_weight: float
    reference_shape_weight: float


def run_shape_contrastive_smoke(
    config: SmokeConfig,
    *,
    contrastive_weight: float,
    contrastive_margin: float,
    shape_weight: float,
    reference_shape_weight: float,
) -> ShapeContrastiveSmokeSummary:
    validate_config(config)
    _validate_weights(shape_weight, reference_shape_weight)
    if config.max_rows < 2:
        raise SmokeInputError("shape contrastive smoke requires at least two rows")
    seed_everything(config.seed)
    device = torch.device(config.device)
    dtype = torch.float32
    rows = load_pair_rows(config.manifest_path, limit=config.max_rows)
    if len(rows) < 2:
        raise SmokeInputError("shape contrastive smoke requires at least two loaded rows")
    explicit_negative_rows = sum(1 for row in rows if row.neg_id is not None)
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
        config.siglip_model_id, torch_dtype=dtype, trust_remote_code=True
    ).to(device)
    processor = AutoImageProcessor.from_pretrained(config.siglip_model_id)
    frozen_params += freeze_module(siglip)

    adapter = load_trainable_adapter(config, device, dtype)
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=config.lr)
    scheduler = FlowMatchEulerDiscreteScheduler(num_train_timesteps=1000, shift=1.0)
    prepared_cache = prepare_cache(
        rows, config, vae, text_encoder, anima, siglip, processor, prepare_text_inputs, device, dtype
    )
    ref_cache = prepare_reference_latent_cache(rows, config, vae, device, dtype)
    losses: list[float] = []
    base_losses: list[float] = []
    contrastive_losses: list[float] = []
    shape_losses: list[float] = []

    for step in range(config.steps):
        prepared = get_prepared(
            prepared_cache, rows, step % len(rows), config, vae, text_encoder,
            anima, siglip, processor, prepare_text_inputs, device, dtype,
        )
        wrong = get_wrong_prepared(
            prepared_cache, rows, step % len(rows), config, vae, text_encoder,
            anima, siglip, processor, prepare_text_inputs, device, dtype,
        )
        ref_latents = get_reference_latents(ref_cache, rows, step % len(rows), config, vae, device, dtype)
        loss, parts = _shape_contrastive_step(
            anima, adapter, prepared, wrong, ref_latents, scheduler, device, dtype,
            contrastive_weight, contrastive_margin, shape_weight, reference_shape_weight,
        )
        if not torch.isfinite(loss):
            raise SmokeInputError(
                f"non-finite loss at step {step}: {float(loss.detach().cpu())}"
            )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
        base_losses.append(parts[0])
        contrastive_losses.append(parts[1])
        shape_losses.append(parts[2])

    save_adapter_checkpoint(adapter, config.output_path)
    checkpoint = verify_checkpoint(config.output_path, config.pe_checkpoint_path)
    return _summary(
        config, len(rows), losses, base_losses, contrastive_losses, shape_losses,
        explicit_negative_rows, trainable_parameter_count(adapter), frozen_params,
        checkpoint, contrastive_weight, contrastive_margin, shape_weight,
        reference_shape_weight,
    )


def _shape_contrastive_step(
    anima: torch.nn.Module,
    adapter,
    prepared,
    wrong,
    ref_latents: torch.Tensor,
    scheduler,
    device: torch.device,
    dtype: torch.dtype,
    contrastive_weight: float,
    contrastive_margin: float,
    shape_weight: float,
    reference_shape_weight: float,
) -> tuple[torch.Tensor, tuple[float, float, float]]:
    latents = prepared.latents
    noise = torch.randn_like(latents)
    from library.runtime.noise import get_noisy_model_input_and_timesteps

    noisy, timesteps, sigmas = get_noisy_model_input_and_timesteps(
        noise_args(), scheduler, latents, noise, device, dtype
    )
    padding_mask = torch.zeros(latents.shape[0], 1, latents.shape[-2], latents.shape[-1], device=device, dtype=dtype)
    target = noise - latents
    correct_pred = _predict(anima, adapter, prepared, noisy, timesteps, padding_mask)
    wrong_pred = _predict(anima, adapter, wrong, noisy, timesteps, padding_mask)
    base_loss = torch.nn.functional.mse_loss(correct_pred.float(), target.float())
    contrastive_loss = reference_margin_loss(
        correct_pred, wrong_pred, target, margin=contrastive_margin
    )
    denoised = denoised_latents_from_velocity(noisy, correct_pred, sigmas)
    shape_loss = latent_edge_projection_loss(
        denoised, latents, reference_latents=ref_latents, reference_weight=reference_shape_weight
    )
    loss = base_loss + contrastive_weight * contrastive_loss + shape_weight * shape_loss
    return loss, (
        float(base_loss.detach().cpu()),
        float(contrastive_loss.detach().cpu()),
        float(shape_loss.detach().cpu()),
    )


def _validate_weights(shape_weight: float, reference_shape_weight: float) -> None:
    if shape_weight < 0.0:
        raise SmokeInputError("shape_weight must be >= 0")
    if reference_shape_weight < 0.0:
        raise SmokeInputError("reference_shape_weight must be >= 0")


def _summary(
    config: SmokeConfig,
    rows_loaded: int,
    losses: list[float],
    base_losses: list[float],
    contrastive_losses: list[float],
    shape_losses: list[float],
    explicit_negative_rows: int,
    trainable_parameters: int,
    frozen_params: int,
    checkpoint: CheckpointVerification,
    contrastive_weight: float,
    contrastive_margin: float,
    shape_weight: float,
    reference_shape_weight: float,
) -> ShapeContrastiveSmokeSummary:
    return ShapeContrastiveSmokeSummary(
        steps=config.steps,
        rows_loaded=rows_loaded,
        first_loss=losses[0],
        final_loss=losses[-1],
        mean_loss=sum(losses) / len(losses),
        mean_base_loss=sum(base_losses) / len(base_losses),
        mean_contrastive_loss=sum(contrastive_losses) / len(contrastive_losses),
        mean_shape_loss=sum(shape_losses) / len(shape_losses),
        finite_loss=all(math.isfinite(loss) for loss in losses),
        explicit_negative_rows=explicit_negative_rows,
        trainable_parameters=trainable_parameters,
        frozen_base_parameters=frozen_params,
        checkpoint=checkpoint,
        init_checkpoint_path=str(config.init_checkpoint_path) if config.init_checkpoint_path else None,
        contrastive_weight=contrastive_weight,
        contrastive_margin=contrastive_margin,
        shape_weight=shape_weight,
        reference_shape_weight=reference_shape_weight,
    )
