from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

import torch
from torch.nn import functional as F

from training.siglip_smoke_types import SmokeInputError


class AnimaTeacherLike(Protocol):
    def __call__(
        self,
        noisy: torch.Tensor,
        timesteps: torch.Tensor,
        crossattn_emb: torch.Tensor,
        *,
        padding_mask: torch.Tensor,
    ) -> torch.Tensor: ...


class PETeacherNetwork(Protocol):
    def apply_to(
        self,
        text_encoders: Sequence[torch.nn.Module],
        unet: AnimaTeacherLike,
    ) -> None: ...

    def encode_ip_tokens(self, pe_features: torch.Tensor) -> torch.Tensor: ...

    def set_ip_tokens(self, ip_tokens: torch.Tensor) -> None: ...

    def clear_ip_tokens(self) -> None: ...

    def remove_from(self) -> None: ...


def teacher_distillation_loss(
    student_prediction: torch.Tensor,
    teacher_prediction: torch.Tensor,
) -> torch.Tensor:
    """Return the denoiser-output loss that makes SigLIP imitate PE control."""
    if student_prediction.shape != teacher_prediction.shape:
        raise SmokeInputError(
            "teacher prediction must match student prediction shape: "
            f"student={tuple(student_prediction.shape)}, "
            f"teacher={tuple(teacher_prediction.shape)}"
        )
    if student_prediction.ndim < 1:
        raise SmokeInputError("teacher prediction must include a batch dimension")
    return F.mse_loss(student_prediction.float(), teacher_prediction.float())


def predict_with_pe_teacher(
    *,
    anima: AnimaTeacherLike,
    network: PETeacherNetwork,
    pe_features: torch.Tensor,
    noisy: torch.Tensor,
    timesteps: torch.Tensor,
    crossattn_emb: torch.Tensor,
    padding_mask: torch.Tensor,
) -> torch.Tensor:
    """Run a frozen PE adapter teacher on the same noisy latent training input."""
    network.apply_to([], anima)
    try:
        with torch.no_grad():
            ip_tokens = network.encode_ip_tokens(pe_features)
            network.set_ip_tokens(ip_tokens)
            prediction = anima(
                noisy.unsqueeze(2),
                timesteps,
                crossattn_emb,
                padding_mask=padding_mask,
            )
    finally:
        try:
            network.clear_ip_tokens()
        finally:
            network.remove_from()
    return prediction.squeeze(2).detach()
