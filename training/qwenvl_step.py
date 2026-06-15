from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.nn import functional as F

from qwenvl_model import IPAdapterQwenVL
from training.qwenvl_real_smoke import PreparedQwenVLRow
from training.qwenvl_token_retrieval import qwenvl_token_retrieval_loss
from training.siglip_reference_loss import reference_margin_loss
from training.siglip_smoke_patch import patched_cross_attention
from training.siglip_smoke_runtime import noise_args


@dataclass(frozen=True, slots=True)
class QwenVLStepWeights:
    contrastive: float
    retrieval: float


@dataclass(frozen=True, slots=True)
class QwenVLStepLosses:
    total: torch.Tensor
    base: torch.Tensor
    contrastive: torch.Tensor
    retrieval: torch.Tensor


def compute_qwenvl_loss_total(
    *,
    base: torch.Tensor,
    contrastive: torch.Tensor,
    retrieval: torch.Tensor,
    weights: QwenVLStepWeights,
) -> QwenVLStepLosses:
    total = base + weights.contrastive * contrastive + weights.retrieval * retrieval
    return QwenVLStepLosses(
        total=total,
        base=base,
        contrastive=contrastive,
        retrieval=retrieval,
    )


def run_qwenvl_step(
    *,
    anima: torch.nn.Module,
    adapter: IPAdapterQwenVL,
    prepared: PreparedQwenVLRow,
    wrong_prepared: PreparedQwenVLRow,
    scheduler,
    device: torch.device,
    dtype: torch.dtype,
    weights: QwenVLStepWeights,
    contrastive_margin: float,
    retrieval_margin: float,
) -> QwenVLStepLosses:
    latents = prepared.latents
    noise = torch.randn_like(latents)
    from library.runtime.noise import get_noisy_model_input_and_timesteps

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
    correct_tokens = adapter.encode_ref(prepared.embedding, timestep=timesteps)
    wrong_tokens = adapter.encode_ref(wrong_prepared.embedding, timestep=timesteps)
    correct_pred = _predict_with_tokens(
        anima, adapter, prepared, correct_tokens, noisy, timesteps, padding_mask
    )
    wrong_pred = _predict_with_tokens(
        anima, adapter, wrong_prepared, wrong_tokens, noisy, timesteps, padding_mask
    )
    target = noise - latents
    base = F.mse_loss(correct_pred.float(), target.float())
    contrastive = reference_margin_loss(
        correct_pred, wrong_pred, target, margin=contrastive_margin
    )
    retrieval = (
        qwenvl_token_retrieval_loss(
            student_tokens=correct_tokens,
            embedding=prepared.embedding,
            wrong_embedding=wrong_prepared.embedding,
            margin=retrieval_margin,
        )
        if weights.retrieval > 0.0
        else base.detach().new_zeros(())
    )
    return compute_qwenvl_loss_total(
        base=base,
        contrastive=contrastive,
        retrieval=retrieval,
        weights=weights,
    )


def _predict_with_tokens(
    anima: torch.nn.Module,
    adapter: IPAdapterQwenVL,
    prepared: PreparedQwenVLRow,
    image_tokens: torch.Tensor,
    noisy: torch.Tensor,
    timesteps: torch.Tensor,
    padding_mask: torch.Tensor,
) -> torch.Tensor:
    with patched_cross_attention(anima, adapter, image_tokens):
        pred = anima(
            noisy.unsqueeze(2),
            timesteps,
            prepared.crossattn_emb,
            padding_mask=padding_mask,
        )
    return pred.squeeze(2)
