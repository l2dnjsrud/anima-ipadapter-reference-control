from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.nn import functional as F

from training.pe_teacher_distillation import teacher_distillation_loss
from training.pe_teacher_token_alignment import (
    PEKVProjector,
    SigLIPKVProjector,
    pe_token_alignment_loss,
)
from training.pe_token_retrieval import pe_token_retrieval_loss
from training.siglip_reference_loss import (
    reference_margin_loss,
    reference_token_separation_loss,
)


@dataclass(frozen=True, slots=True)
class TeacherLossWeights:
    contrastive: float
    teacher: float
    token: float
    pe_token: float
    pe_retrieval: float


@dataclass(frozen=True, slots=True)
class TeacherStepLosses:
    total: torch.Tensor
    base: torch.Tensor
    contrastive: torch.Tensor
    teacher: torch.Tensor
    token: torch.Tensor
    pe_token: torch.Tensor
    pe_retrieval: torch.Tensor


def compute_teacher_step_losses(
    *,
    adapter: SigLIPKVProjector,
    pe_network: PEKVProjector,
    correct_pred: torch.Tensor,
    wrong_pred: torch.Tensor,
    target: torch.Tensor,
    teacher_pred: torch.Tensor,
    correct_tokens: torch.Tensor,
    wrong_tokens: torch.Tensor,
    pe_tokens: torch.Tensor,
    wrong_pe_tokens: torch.Tensor,
    weights: TeacherLossWeights,
    contrastive_margin: float,
    token_max_similarity: float,
    pe_token_block_stride: int,
    pe_retrieval_margin: float,
) -> TeacherStepLosses:
    """Compute all teacher-smoke losses for one training step."""
    base = F.mse_loss(correct_pred.float(), target.float())
    contrastive = reference_margin_loss(
        correct_pred, wrong_pred, target, margin=contrastive_margin
    )
    teacher = teacher_distillation_loss(correct_pred, teacher_pred)
    token = reference_token_separation_loss(
        correct_tokens, wrong_tokens, max_similarity=token_max_similarity
    )
    pe_token = (
        pe_token_alignment_loss(
            adapter,
            pe_network,
            student_tokens=correct_tokens,
            pe_tokens=pe_tokens,
            block_stride=pe_token_block_stride,
        )
        if weights.pe_token > 0.0
        else _zero_like(base)
    )
    pe_retrieval = (
        pe_token_retrieval_loss(
            student_tokens=correct_tokens,
            pe_tokens=pe_tokens,
            wrong_pe_tokens=wrong_pe_tokens,
            margin=pe_retrieval_margin,
        )
        if weights.pe_retrieval > 0.0
        else _zero_like(base)
    )
    total = (
        base
        + weights.contrastive * contrastive
        + weights.teacher * teacher
        + weights.token * token
        + weights.pe_token * pe_token
        + weights.pe_retrieval * pe_retrieval
    )
    return TeacherStepLosses(
        total=total,
        base=base,
        contrastive=contrastive,
        teacher=teacher,
        token=token,
        pe_token=pe_token,
        pe_retrieval=pe_retrieval,
    )


def _zero_like(loss: torch.Tensor) -> torch.Tensor:
    return loss.detach().new_zeros(())
