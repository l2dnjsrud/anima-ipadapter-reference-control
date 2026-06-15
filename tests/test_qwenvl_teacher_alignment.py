from __future__ import annotations

import pytest
import torch

from training.qwenvl_teacher_alignment_loss import (
    qwenvl_teacher_alignment_loss,
)
from training.qwenvl_teacher_alignment_step import (
    QwenVLTeacherAlignmentStepLosses,
    QwenVLTeacherAlignmentStepWeights,
    compute_qwenvl_teacher_alignment_total,
)
from training.siglip_smoke_types import SmokeInputError


def test_qwenvl_teacher_alignment_loss_is_zero_for_matching_descriptor() -> None:
    student_tokens = torch.tensor(
        [
            [
                [1.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
            ],
        ]
    )
    teacher_embedding = torch.tensor([[1.0, 0.0, 0.0]])

    loss = qwenvl_teacher_alignment_loss(
        student_tokens=student_tokens,
        teacher_embedding=teacher_embedding,
    )

    assert torch.allclose(loss, torch.tensor(0.0))


def test_qwenvl_teacher_alignment_loss_penalizes_wrong_descriptor() -> None:
    student_tokens = torch.tensor(
        [
            [
                [1.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
            ],
        ]
    )
    teacher_embedding = torch.tensor([[0.0, 1.0, 0.0]])

    loss = qwenvl_teacher_alignment_loss(
        student_tokens=student_tokens,
        teacher_embedding=teacher_embedding,
    )

    assert torch.allclose(loss, torch.tensor(1.0))


def test_qwenvl_teacher_alignment_loss_rejects_dim_mismatch() -> None:
    with pytest.raises(SmokeInputError, match="teacher alignment dim"):
        qwenvl_teacher_alignment_loss(
            student_tokens=torch.zeros(1, 2, 4),
            teacher_embedding=torch.zeros(1, 3),
        )


def test_compute_qwenvl_teacher_alignment_total_includes_teacher_weight() -> None:
    losses = compute_qwenvl_teacher_alignment_total(
        base=torch.tensor(0.1),
        contrastive=torch.tensor(0.2),
        retrieval=torch.tensor(0.3),
        teacher=torch.tensor(0.4),
        weights=QwenVLTeacherAlignmentStepWeights(
            contrastive=0.5,
            retrieval=2.0,
            teacher=3.0,
        ),
    )

    assert isinstance(losses, QwenVLTeacherAlignmentStepLosses)
    assert torch.allclose(losses.total, torch.tensor(2.0))
