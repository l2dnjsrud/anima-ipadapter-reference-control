from __future__ import annotations

import torch

from training.pe_token_retrieval import pe_token_retrieval_loss
from training.siglip_smoke_types import SmokeInputError


def test_pe_token_retrieval_loss_is_zero_when_positive_is_closer_than_wrong() -> None:
    student = torch.tensor([[[1.0, 0.0], [1.0, 0.0]]])
    positive = student.clone()
    wrong = torch.tensor([[[0.0, 1.0], [0.0, 1.0]]])

    loss = pe_token_retrieval_loss(
        student_tokens=student,
        pe_tokens=positive,
        wrong_pe_tokens=wrong,
        margin=0.2,
    )

    assert torch.equal(loss, torch.zeros_like(loss))


def test_pe_token_retrieval_loss_penalizes_wrong_token_match() -> None:
    student = torch.tensor([[[1.0, 0.0], [1.0, 0.0]]])
    positive = torch.tensor([[[0.0, 1.0], [0.0, 1.0]]])
    wrong = student.clone()

    loss = pe_token_retrieval_loss(
        student_tokens=student,
        pe_tokens=positive,
        wrong_pe_tokens=wrong,
        margin=0.2,
    )

    assert torch.allclose(loss, torch.tensor(1.2))


def test_pe_token_retrieval_loss_rejects_non_pe_space_tokens() -> None:
    student = torch.zeros(1, 2, 4)
    positive = torch.zeros(1, 2, 3)
    wrong = torch.zeros(1, 2, 3)

    try:
        pe_token_retrieval_loss(
            student_tokens=student,
            pe_tokens=positive,
            wrong_pe_tokens=wrong,
            margin=0.2,
        )
    except SmokeInputError as error:
        assert "hidden dim" in str(error)
    else:
        raise AssertionError("non-PE-space token dimensions should fail")
