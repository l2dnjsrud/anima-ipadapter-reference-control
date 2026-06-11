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

from training.siglip_real_smoke import (  # noqa: E402
    PreparedTrainingRow,
    freeze_module,
    load_trainable_adapter,
    save_adapter_checkpoint,
    trainable_parameter_count,
    verify_checkpoint,
)
from training.siglip_prepared_cache import get_prepared, prepare_cache  # noqa: E402
from training.siglip_reference_loss import (  # noqa: E402
    reference_margin_loss,
    wrong_reference_index,
)
from training.siglip_smoke_data import load_pair_rows  # noqa: E402
from training.siglip_smoke_patch import patched_cross_attention  # noqa: E402
from training.siglip_smoke_runtime import noise_args, seed_everything, validate_config  # noqa: E402
from training.siglip_smoke_types import (  # noqa: E402
    CheckpointVerification,
    SmokeConfig,
    SmokeInputError,
)


@dataclass(frozen=True, slots=True)
class ContrastiveSmokeSummary:
    steps: int
    rows_loaded: int
    first_loss: float
    final_loss: float
    mean_loss: float
    mean_base_loss: float
    mean_contrastive_loss: float
    finite_loss: bool
    trainable_parameters: int
    frozen_base_parameters: int
    checkpoint: CheckpointVerification
    init_checkpoint_path: str | None
    contrastive_weight: float
    contrastive_margin: float


def run_contrastive_smoke(
    config: SmokeConfig,
    *,
    contrastive_weight: float,
    contrastive_margin: float,
) -> ContrastiveSmokeSummary:
    validate_config(config)
    if config.max_rows < 2:
        raise SmokeInputError("contrastive smoke requires at least two rows")
    seed_everything(config.seed)
    device = torch.device(config.device)
    dtype = torch.float32
    rows = load_pair_rows(config.manifest_path, limit=config.max_rows)
    if len(rows) < 2:
        raise SmokeInputError("contrastive smoke requires at least two loaded rows")
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
    cache = prepare_cache(
        rows,
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
    losses: list[float] = []
    base_losses: list[float] = []
    contrastive_losses: list[float] = []

    for step in range(config.steps):
        row_index = step % len(rows)
        prepared = get_prepared(
            cache,
            rows,
            row_index,
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
        wrong_prepared = get_prepared(
            cache,
            rows,
            wrong_reference_index(row_index, len(rows)),
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
        latents = prepared.latents
        noise = torch.randn_like(latents)
        noisy, timesteps, _sigmas = get_noisy_model_input_and_timesteps(
            noise_args(), scheduler, latents, noise, device, dtype
        )
        padding_mask = torch.zeros(
            latents.shape[0],
            1,
            latents.shape[-2],
            latents.shape[-1],
            device=device,
            dtype=dtype,
        )
        target = noise - latents

        optimizer.zero_grad(set_to_none=True)
        correct_pred = _predict(
            anima, adapter, prepared, noisy, timesteps, padding_mask
        )
        wrong_pred = _predict(
            anima, adapter, wrong_prepared, noisy, timesteps, padding_mask
        )
        base_loss = torch.nn.functional.mse_loss(correct_pred.float(), target.float())
        contrastive_loss = reference_margin_loss(
            correct_pred, wrong_pred, target, margin=contrastive_margin
        )
        loss = base_loss + contrastive_weight * contrastive_loss
        if not torch.isfinite(loss):
            raise SmokeInputError(
                f"non-finite loss at step {step}: {float(loss.detach().cpu())}"
            )
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
        base_losses.append(float(base_loss.detach().cpu()))
        contrastive_losses.append(float(contrastive_loss.detach().cpu()))

    save_adapter_checkpoint(adapter, config.output_path)
    checkpoint = verify_checkpoint(config.output_path, config.pe_checkpoint_path)
    return ContrastiveSmokeSummary(
        steps=config.steps,
        rows_loaded=len(rows),
        first_loss=losses[0],
        final_loss=losses[-1],
        mean_loss=sum(losses) / len(losses),
        mean_base_loss=sum(base_losses) / len(base_losses),
        mean_contrastive_loss=sum(contrastive_losses) / len(contrastive_losses),
        finite_loss=all(math.isfinite(loss) for loss in losses),
        trainable_parameters=trainable_parameter_count(adapter),
        frozen_base_parameters=frozen_params,
        checkpoint=checkpoint,
        init_checkpoint_path=str(config.init_checkpoint_path)
        if config.init_checkpoint_path
        else None,
        contrastive_weight=contrastive_weight,
        contrastive_margin=contrastive_margin,
    )


def _predict(
    anima: torch.nn.Module,
    adapter,
    prepared: PreparedTrainingRow,
    noisy: torch.Tensor,
    timesteps: torch.Tensor,
    padding_mask: torch.Tensor,
) -> torch.Tensor:
    image_tokens = adapter.encode_ref(prepared.features, timestep=timesteps)
    with patched_cross_attention(anima, adapter, image_tokens):
        pred = anima(
            noisy.unsqueeze(2),
            timesteps,
            prepared.crossattn_emb,
            padding_mask=padding_mask,
        )
    return pred.squeeze(2)
