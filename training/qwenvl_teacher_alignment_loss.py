from __future__ import annotations

import torch
from torch.nn import functional as F

from training.siglip_smoke_types import SmokeInputError


def qwenvl_teacher_alignment_loss(
    *,
    student_tokens: torch.Tensor,
    teacher_embedding: torch.Tensor,
) -> torch.Tensor:
    """Align pooled QwenVL adapter tokens to a teacher image embedding."""
    _validate_shapes(
        student_tokens=student_tokens,
        teacher_embedding=teacher_embedding,
    )
    student = _descriptor(student_tokens)
    teacher = _descriptor(teacher_embedding).to(device=student.device, dtype=student.dtype)
    cosine = (student * teacher).sum(dim=-1)
    return (1.0 - cosine).mean()


def _descriptor(values: torch.Tensor) -> torch.Tensor:
    if values.ndim == 3:
        values = values.mean(dim=1)
    return F.normalize(values.float(), dim=-1)


def _validate_shapes(
    *,
    student_tokens: torch.Tensor,
    teacher_embedding: torch.Tensor,
) -> None:
    if student_tokens.ndim != 3:
        raise SmokeInputError("student tokens must have shape [batch, token, dim]")
    if teacher_embedding.ndim not in (2, 3):
        raise SmokeInputError("teacher embedding must be rank 2 or rank 3")
    if student_tokens.shape[0] != teacher_embedding.shape[0]:
        raise SmokeInputError("student tokens and teacher embedding must share batch size")
    if student_tokens.shape[-1] != teacher_embedding.shape[-1]:
        raise SmokeInputError(
            "teacher alignment dim mismatch: "
            f"student={student_tokens.shape[-1]}, teacher={teacher_embedding.shape[-1]}"
        )
