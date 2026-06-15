from __future__ import annotations

import torch
from torch.nn import functional as F

from training.siglip_smoke_types import SmokeInputError


def qwenvl_token_retrieval_loss(
    *,
    student_tokens: torch.Tensor,
    embedding: torch.Tensor,
    wrong_embedding: torch.Tensor,
    margin: float,
) -> torch.Tensor:
    """Make QwenVL image tokens retrieve the matching QwenVL embedding."""
    _validate_shapes(
        student_tokens=student_tokens,
        embedding=embedding,
        wrong_embedding=wrong_embedding,
    )
    student = F.normalize(student_tokens.float().mean(dim=1), dim=-1)
    positive = _embedding_descriptor(embedding).to(student).detach()
    wrong = _embedding_descriptor(wrong_embedding).to(student).detach()
    positive_distance = 1.0 - (student * positive).sum(dim=-1)
    wrong_distance = 1.0 - (student * wrong).sum(dim=-1)
    return torch.relu(positive_distance - wrong_distance + margin).mean()


def _embedding_descriptor(embedding: torch.Tensor) -> torch.Tensor:
    values = embedding.float()
    if values.ndim == 3:
        values = values.mean(dim=1)
    return F.normalize(values, dim=-1)


def _validate_shapes(
    *,
    student_tokens: torch.Tensor,
    embedding: torch.Tensor,
    wrong_embedding: torch.Tensor,
) -> None:
    if student_tokens.ndim != 3:
        raise SmokeInputError("QwenVL student tokens must have shape [batch, token, dim]")
    if embedding.ndim not in (2, 3) or wrong_embedding.ndim not in (2, 3):
        raise SmokeInputError("QwenVL embeddings must be rank 2 or rank 3")
    if embedding.shape != wrong_embedding.shape:
        raise SmokeInputError("matching and wrong QwenVL embeddings must share shape")
    if student_tokens.shape[0] != embedding.shape[0]:
        raise SmokeInputError("QwenVL tokens and embeddings must share batch size")
    if student_tokens.shape[-1] != embedding.shape[-1]:
        raise SmokeInputError(
            "QwenVL token dim must match embedding dim for retrieval training"
        )
