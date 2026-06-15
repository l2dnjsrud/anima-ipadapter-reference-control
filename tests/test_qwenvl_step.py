from __future__ import annotations

import torch

from training.qwenvl_step import (
    QwenVLStepLosses,
    QwenVLStepWeights,
    compute_qwenvl_loss_total,
)


def test_compute_qwenvl_loss_total_includes_retrieval_weight() -> None:
    losses = compute_qwenvl_loss_total(
        base=torch.tensor(0.1),
        contrastive=torch.tensor(0.2),
        retrieval=torch.tensor(0.3),
        weights=QwenVLStepWeights(contrastive=0.5, retrieval=2.0),
    )

    assert isinstance(losses, QwenVLStepLosses)
    assert torch.allclose(losses.total, torch.tensor(0.8))


def test_compute_qwenvl_loss_total_skips_retrieval_when_weight_is_zero() -> None:
    losses = compute_qwenvl_loss_total(
        base=torch.tensor(0.1),
        contrastive=torch.tensor(0.2),
        retrieval=torch.tensor(10.0),
        weights=QwenVLStepWeights(contrastive=0.5, retrieval=0.0),
    )

    assert torch.allclose(losses.total, torch.tensor(0.2))
