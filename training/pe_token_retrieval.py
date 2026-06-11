from __future__ import annotations

import torch
from torch.nn import functional as F

from training.siglip_smoke_types import SmokeInputError


def pe_token_retrieval_loss(
    *,
    student_tokens: torch.Tensor,
    pe_tokens: torch.Tensor,
    wrong_pe_tokens: torch.Tensor,
    margin: float,
) -> torch.Tensor:
    """Make SigLIP PE-space tokens retrieve the matching PE reference tokens."""
    _validate_tokens(
        student_tokens=student_tokens,
        pe_tokens=pe_tokens,
        wrong_pe_tokens=wrong_pe_tokens,
    )
    student = _token_descriptor(student_tokens)
    positive = _token_descriptor(pe_tokens).to(device=student.device, dtype=student.dtype).detach()
    wrong = _token_descriptor(wrong_pe_tokens).to(device=student.device, dtype=student.dtype).detach()
    positive_distance = 1.0 - (student * positive).sum(dim=-1)
    wrong_distance = 1.0 - (student * wrong).sum(dim=-1)
    return torch.relu(positive_distance - wrong_distance + margin).mean()


def _validate_tokens(
    *,
    student_tokens: torch.Tensor,
    pe_tokens: torch.Tensor,
    wrong_pe_tokens: torch.Tensor,
) -> None:
    if student_tokens.ndim != 3 or pe_tokens.ndim != 3 or wrong_pe_tokens.ndim != 3:
        raise SmokeInputError("PE retrieval tokens must have shape [batch, token, dim]")
    if student_tokens.shape[0] != pe_tokens.shape[0]:
        raise SmokeInputError("student and positive PE tokens must share batch size")
    if pe_tokens.shape != wrong_pe_tokens.shape:
        raise SmokeInputError("positive and wrong PE tokens must share shape")
    if student_tokens.shape[-1] != pe_tokens.shape[-1]:
        raise SmokeInputError(
            "student and PE tokens must share hidden dim; use PE-space SigLIP checkpoints"
        )


def _token_descriptor(tokens: torch.Tensor) -> torch.Tensor:
    values = tokens.float()
    mean = values.mean(dim=1)
    std = values.std(dim=1, unbiased=False)
    return F.normalize(torch.cat([mean, std], dim=-1), dim=-1)
