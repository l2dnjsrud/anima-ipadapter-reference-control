from __future__ import annotations

import torch
from torch.nn import functional as F

from training.siglip_smoke_types import SmokeInputError


def deterministic_wrong_reference_indices(
    row_count: int, *, device: torch.device | None = None
) -> torch.Tensor:
    """Return a deterministic non-self wrong-reference index for each row."""
    if row_count < 2:
        raise SmokeInputError(
            "contrastive reference training requires at least two rows"
        )
    indices = torch.arange(row_count, device=device)
    return torch.remainder(indices + row_count // 2, row_count)


def wrong_reference_index(current_index: int, row_count: int) -> int:
    """Return the deterministic non-self reference index for one training row."""
    if row_count < 2:
        raise SmokeInputError(
            "contrastive reference training requires at least two rows"
        )
    return int((current_index + row_count // 2) % row_count)


def reference_margin_loss(
    correct_prediction: torch.Tensor,
    wrong_prediction: torch.Tensor,
    target: torch.Tensor,
    margin: float,
) -> torch.Tensor:
    """Penalize batches where the wrong-reference prediction is too competitive."""
    _validate_prediction_shapes(
        correct_prediction=correct_prediction,
        wrong_prediction=wrong_prediction,
        target=target,
    )
    correct_error = _mean_squared_error_per_row(correct_prediction, target)
    wrong_error = _mean_squared_error_per_row(wrong_prediction, target)
    hinge = torch.relu(correct_error - wrong_error + margin)
    return hinge.mean()


def reference_token_separation_loss(
    correct_tokens: torch.Tensor,
    wrong_tokens: torch.Tensor,
    *,
    max_similarity: float,
) -> torch.Tensor:
    """Penalize adapter image tokens that collapse across different references."""
    _validate_token_shapes(correct_tokens=correct_tokens, wrong_tokens=wrong_tokens)
    correct = F.normalize(correct_tokens.float().mean(dim=1), dim=-1)
    wrong = F.normalize(wrong_tokens.float().mean(dim=1), dim=-1)
    similarity = (correct * wrong).sum(dim=-1)
    return torch.relu(similarity - max_similarity).mean()


def _validate_prediction_shapes(
    *,
    correct_prediction: torch.Tensor,
    wrong_prediction: torch.Tensor,
    target: torch.Tensor,
) -> None:
    if correct_prediction.shape != target.shape:
        raise SmokeInputError("correct_prediction must match target shape")
    if wrong_prediction.shape != target.shape:
        raise SmokeInputError("wrong_prediction must match target shape")
    if correct_prediction.ndim < 1:
        raise SmokeInputError("predictions must include a batch dimension")


def _mean_squared_error_per_row(
    prediction: torch.Tensor, target: torch.Tensor
) -> torch.Tensor:
    squared_error = F.mse_loss(prediction.float(), target.float(), reduction="none")
    return squared_error.reshape(squared_error.shape[0], -1).mean(dim=1)


def _validate_token_shapes(
    *,
    correct_tokens: torch.Tensor,
    wrong_tokens: torch.Tensor,
) -> None:
    if correct_tokens.shape != wrong_tokens.shape:
        raise SmokeInputError("correct_tokens must match wrong_tokens shape")
    if correct_tokens.ndim != 3:
        raise SmokeInputError("reference tokens must have shape [batch, token, dim]")
