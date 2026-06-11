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

from native_pe_models import load_pe_adapter_spec  # noqa: E402
from native_pe_runtime import load_network  # noqa: E402
from training.pe_teacher_distillation import (  # noqa: E402
    predict_with_pe_teacher,
    teacher_distillation_loss,
)
from training.pe_teacher_features import get_pe_features, prepare_pe_cache  # noqa: E402
from training.siglip_contrastive_smoke import _predict  # noqa: E402
from training.siglip_real_smoke import (  # noqa: E402
    freeze_module,
    load_trainable_adapter,
    save_adapter_checkpoint,
    trainable_parameter_count,
    verify_checkpoint,
)
from training.siglip_prepared_cache import get_prepared, prepare_cache  # noqa: E402
from training.siglip_reference_loss import (  # noqa: E402
    reference_margin_loss,
    reference_token_separation_loss,
    wrong_reference_index,
)
from training.siglip_smoke_data import load_pair_rows  # noqa: E402
from training.siglip_smoke_runtime import noise_args, seed_everything, validate_config  # noqa: E402
from training.siglip_smoke_types import (  # noqa: E402
    CheckpointVerification,
    SmokeConfig,
    SmokeInputError,
)


@dataclass(frozen=True, slots=True)
class TeacherSmokeSummary:
    steps: int
    rows_loaded: int
    first_loss: float
    final_loss: float
    mean_loss: float
    mean_base_loss: float
    mean_contrastive_loss: float
    mean_teacher_loss: float
    mean_token_loss: float
    finite_loss: bool
    trainable_parameters: int
    frozen_base_parameters: int
    checkpoint: CheckpointVerification
    init_checkpoint_path: str | None
    contrastive_weight: float
    contrastive_margin: float
    teacher_weight: float


def run_teacher_smoke(
    config: SmokeConfig,
    *,
    contrastive_weight: float,
    contrastive_margin: float,
    teacher_weight: float,
    token_weight: float = 0.0,
    token_max_similarity: float = 0.2,
    pe_encoder_name: str = "pe",
) -> TeacherSmokeSummary:
    validate_config(config)
    if config.max_rows < 2:
        raise SmokeInputError("teacher smoke requires at least two rows")
    seed_everything(config.seed)
    device = torch.device(config.device)
    dtype = torch.float32
    rows = load_pair_rows(config.manifest_path, limit=config.max_rows)
    if len(rows) < 2:
        raise SmokeInputError("teacher smoke requires at least two loaded rows")
    random.Random(config.seed).shuffle(rows)

    from library.anima.weights import load_anima_model, load_qwen3_text_encoder
    from library.inference.text import prepare_text_inputs
    from library.models.qwen_vae import load_vae
    from library.runtime.noise import (
        FlowMatchEulerDiscreteScheduler,
        get_noisy_model_input_and_timesteps,
    )
    from library.vision import encode_pe_from_imageminus1to1, load_pe_encoder
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

    pe_spec = load_pe_adapter_spec(config.pe_checkpoint_path)
    pe_network = load_network(pe_spec, strength=1.0).to(device=device, dtype=dtype)
    pe_encoder = load_pe_encoder(device, name=pe_encoder_name, dtype=torch.bfloat16)
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
    pe_cache = prepare_pe_cache(
        rows,
        config,
        pe_encoder,
        encode_pe_from_imageminus1to1,
        device,
        dtype,
    )
    losses: list[float] = []
    base_losses: list[float] = []
    contrastive_losses: list[float] = []
    teacher_losses: list[float] = []
    token_losses: list[float] = []

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
            latents.shape[0], 1, latents.shape[-2], latents.shape[-1], device=device, dtype=dtype
        )
        target = noise - latents
        pe_features = get_pe_features(
            pe_cache,
            rows,
            row_index,
            config,
            pe_encoder,
            encode_pe_from_imageminus1to1,
            device,
            dtype,
        )

        optimizer.zero_grad(set_to_none=True)
        teacher_pred = predict_with_pe_teacher(
            anima=anima,
            network=pe_network,
            pe_features=pe_features,
            noisy=noisy,
            timesteps=timesteps,
            crossattn_emb=prepared.crossattn_emb,
            padding_mask=padding_mask,
        )
        correct_pred = _predict(
            anima, adapter, prepared, noisy, timesteps, padding_mask
        )
        wrong_pred = _predict(
            anima, adapter, wrong_prepared, noisy, timesteps, padding_mask
        )
        correct_tokens = adapter.encode_ref(prepared.features, timestep=timesteps)
        wrong_tokens = adapter.encode_ref(wrong_prepared.features, timestep=timesteps)
        base_loss = torch.nn.functional.mse_loss(correct_pred.float(), target.float())
        contrastive_loss = reference_margin_loss(
            correct_pred, wrong_pred, target, margin=contrastive_margin
        )
        teacher_loss = teacher_distillation_loss(correct_pred, teacher_pred)
        token_loss = reference_token_separation_loss(
            correct_tokens, wrong_tokens, max_similarity=token_max_similarity
        )
        loss = (
            base_loss
            + contrastive_weight * contrastive_loss
            + teacher_weight * teacher_loss
            + token_weight * token_loss
        )
        if not torch.isfinite(loss):
            raise SmokeInputError(
                f"non-finite loss at step {step}: {float(loss.detach().cpu())}"
            )
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
        base_losses.append(float(base_loss.detach().cpu()))
        contrastive_losses.append(float(contrastive_loss.detach().cpu()))
        teacher_losses.append(float(teacher_loss.detach().cpu()))
        token_losses.append(float(token_loss.detach().cpu()))

    save_adapter_checkpoint(adapter, config.output_path)
    checkpoint = verify_checkpoint(config.output_path, config.pe_checkpoint_path)
    return TeacherSmokeSummary(
        steps=config.steps,
        rows_loaded=len(rows),
        first_loss=losses[0],
        final_loss=losses[-1],
        mean_loss=sum(losses) / len(losses),
        mean_base_loss=sum(base_losses) / len(base_losses),
        mean_contrastive_loss=sum(contrastive_losses) / len(contrastive_losses),
        mean_teacher_loss=sum(teacher_losses) / len(teacher_losses),
        mean_token_loss=sum(token_losses) / len(token_losses),
        finite_loss=all(math.isfinite(loss) for loss in losses),
        trainable_parameters=trainable_parameter_count(adapter),
        frozen_base_parameters=frozen_params,
        checkpoint=checkpoint,
        init_checkpoint_path=str(config.init_checkpoint_path)
        if config.init_checkpoint_path
        else None,
        contrastive_weight=contrastive_weight,
        contrastive_margin=contrastive_margin,
        teacher_weight=teacher_weight,
    )
