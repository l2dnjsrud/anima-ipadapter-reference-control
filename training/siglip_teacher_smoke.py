from __future__ import annotations

import random
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
ANIMA_ROOT = Path("/home/wktwin/anima-lora-training-bundle/anima_lora")
for candidate in (ROOT, ANIMA_ROOT):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from training.pe_teacher_distillation import predict_with_pe_teacher  # noqa: E402
from training.pe_teacher_features import get_pe_features, prepare_pe_cache  # noqa: E402
from training.siglip_contrastive_smoke import _predict  # noqa: E402
from training.siglip_real_smoke import (  # noqa: E402
    save_adapter_checkpoint,
    trainable_parameter_count,
    verify_checkpoint,
)
from training.siglip_prepared_cache import get_prepared, prepare_cache  # noqa: E402
from training.siglip_reference_loss import wrong_reference_index  # noqa: E402
from training.siglip_smoke_checkpoint import trainable_adapter_parameters  # noqa: E402
from training.siglip_smoke_data import load_pair_rows  # noqa: E402
from training.siglip_smoke_runtime import noise_args, seed_everything, validate_config  # noqa: E402
from training.siglip_smoke_types import SmokeConfig, SmokeInputError  # noqa: E402
from training.siglip_teacher_summary import TeacherSmokeSummary, build_teacher_smoke_summary  # noqa: E402
from training.siglip_teacher_step import TeacherLossWeights, compute_teacher_step_losses  # noqa: E402
from training.siglip_teacher_runtime import load_teacher_runtime  # noqa: E402


def run_teacher_smoke(
    config: SmokeConfig,
    *,
    contrastive_weight: float,
    contrastive_margin: float,
    teacher_weight: float,
    token_weight: float = 0.0,
    token_max_similarity: float = 0.2,
    pe_token_weight: float = 0.0,
    pe_token_block_stride: int = 4,
    pe_retrieval_weight: float = 0.0,
    pe_retrieval_margin: float = 0.2,
    pe_kv_init: bool = False,
    pe_encoder_name: str = "pe",
    calibrator_bottleneck_dim: int | None = None,
    train_calibrator_only: bool = False,
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

    from library.runtime.noise import (
        FlowMatchEulerDiscreteScheduler,
        get_noisy_model_input_and_timesteps,
    )

    runtime = load_teacher_runtime(
        config,
        device=device,
        dtype=dtype,
        pe_kv_init=pe_kv_init,
        pe_encoder_name=pe_encoder_name,
        calibrator_bottleneck_dim=calibrator_bottleneck_dim,
        train_calibrator_only=train_calibrator_only,
    )
    optimizer = torch.optim.AdamW(trainable_adapter_parameters(runtime.adapter), lr=config.lr)
    scheduler = FlowMatchEulerDiscreteScheduler(num_train_timesteps=1000, shift=1.0)
    cache = prepare_cache(
        rows,
        config,
        runtime.vae,
        runtime.text_encoder,
        runtime.anima,
        runtime.siglip,
        runtime.processor,
        runtime.prepare_text_inputs,
        device,
        dtype,
    )
    pe_cache = prepare_pe_cache(
        rows,
        config,
        runtime.pe_encoder,
        runtime.encode_pe_from_imageminus1to1,
        device,
        dtype,
    )
    losses: list[float] = []
    base_losses: list[float] = []
    contrastive_losses: list[float] = []
    teacher_losses: list[float] = []
    token_losses: list[float] = []
    pe_token_losses: list[float] = []
    pe_retrieval_losses: list[float] = []
    weights = TeacherLossWeights(
        contrastive=contrastive_weight,
        teacher=teacher_weight,
        token=token_weight,
        pe_token=pe_token_weight,
        pe_retrieval=pe_retrieval_weight,
    )

    for step in range(config.steps):
        row_index = step % len(rows)
        wrong_index = wrong_reference_index(row_index, len(rows))
        prepared = get_prepared(
            cache,
            rows,
            row_index,
            config,
            runtime.vae,
            runtime.text_encoder,
            runtime.anima,
            runtime.siglip,
            runtime.processor,
            runtime.prepare_text_inputs,
            device,
            dtype,
        )
        wrong_prepared = get_prepared(
            cache,
            rows,
            wrong_index,
            config,
            runtime.vae,
            runtime.text_encoder,
            runtime.anima,
            runtime.siglip,
            runtime.processor,
            runtime.prepare_text_inputs,
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
            runtime.pe_encoder,
            runtime.encode_pe_from_imageminus1to1,
            device,
            dtype,
        )
        wrong_pe_features = get_pe_features(
            pe_cache,
            rows,
            wrong_index,
            config,
            runtime.pe_encoder,
            runtime.encode_pe_from_imageminus1to1,
            device,
            dtype,
        )
        with torch.no_grad():
            pe_tokens = runtime.pe_network.encode_ip_tokens(pe_features).detach()
            wrong_pe_tokens = runtime.pe_network.encode_ip_tokens(wrong_pe_features).detach()

        optimizer.zero_grad(set_to_none=True)
        teacher_pred = predict_with_pe_teacher(
            anima=runtime.anima,
            network=runtime.pe_network,
            pe_features=pe_features,
            noisy=noisy,
            timesteps=timesteps,
            crossattn_emb=prepared.crossattn_emb,
            padding_mask=padding_mask,
        )
        correct_pred = _predict(
            runtime.anima, runtime.adapter, prepared, noisy, timesteps, padding_mask
        )
        wrong_pred = _predict(
            runtime.anima, runtime.adapter, wrong_prepared, noisy, timesteps, padding_mask
        )
        correct_tokens = runtime.adapter.encode_ref(prepared.features, timestep=timesteps)
        wrong_tokens = runtime.adapter.encode_ref(wrong_prepared.features, timestep=timesteps)
        step_losses = compute_teacher_step_losses(
            adapter=runtime.adapter,
            pe_network=runtime.pe_network,
            correct_pred=correct_pred,
            wrong_pred=wrong_pred,
            target=target,
            teacher_pred=teacher_pred,
            correct_tokens=correct_tokens,
            wrong_tokens=wrong_tokens,
            pe_tokens=pe_tokens,
            wrong_pe_tokens=wrong_pe_tokens,
            weights=weights,
            contrastive_margin=contrastive_margin,
            token_max_similarity=token_max_similarity,
            pe_token_block_stride=pe_token_block_stride,
            pe_retrieval_margin=pe_retrieval_margin,
        )
        loss = step_losses.total
        if not torch.isfinite(loss):
            raise SmokeInputError(
                f"non-finite loss at step {step}: {float(loss.detach().cpu())}"
            )
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
        base_losses.append(float(step_losses.base.detach().cpu()))
        contrastive_losses.append(float(step_losses.contrastive.detach().cpu()))
        teacher_losses.append(float(step_losses.teacher.detach().cpu()))
        token_losses.append(float(step_losses.token.detach().cpu()))
        pe_token_losses.append(float(step_losses.pe_token.detach().cpu()))
        pe_retrieval_losses.append(float(step_losses.pe_retrieval.detach().cpu()))

    save_adapter_checkpoint(runtime.adapter, config.output_path)
    checkpoint = verify_checkpoint(config.output_path, config.pe_checkpoint_path)
    return build_teacher_smoke_summary(
        steps=config.steps,
        rows_loaded=len(rows),
        losses=losses,
        base_losses=base_losses,
        contrastive_losses=contrastive_losses,
        teacher_losses=teacher_losses,
        token_losses=token_losses,
        pe_token_losses=pe_token_losses,
        pe_retrieval_losses=pe_retrieval_losses,
        trainable_parameters=trainable_parameter_count(runtime.adapter),
        frozen_base_parameters=runtime.frozen_params,
        checkpoint=checkpoint,
        init_checkpoint_path=config.init_checkpoint_path,
        contrastive_weight=contrastive_weight,
        contrastive_margin=contrastive_margin,
        teacher_weight=teacher_weight,
        token_weight=token_weight,
        token_max_similarity=token_max_similarity,
        pe_token_weight=pe_token_weight,
        pe_token_block_stride=pe_token_block_stride,
        pe_retrieval_weight=pe_retrieval_weight,
        pe_retrieval_margin=pe_retrieval_margin,
        calibrator_bottleneck_dim=calibrator_bottleneck_dim,
        train_calibrator_only=train_calibrator_only,
    )
