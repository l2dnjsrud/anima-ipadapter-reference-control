from __future__ import annotations

import torch

from training.qwenvl_token_retrieval import qwenvl_token_retrieval_loss
from training.siglip_smoke_types import SmokeInputError


def test_qwenvl_token_retrieval_loss_is_zero_when_embedding_is_closer() -> None:
    student_tokens = torch.tensor([[[1.0, 0.0], [1.0, 0.0]]])
    embedding = torch.tensor([[1.0, 0.0]])
    wrong_embedding = torch.tensor([[0.0, 1.0]])

    loss = qwenvl_token_retrieval_loss(
        student_tokens=student_tokens,
        embedding=embedding,
        wrong_embedding=wrong_embedding,
        margin=0.2,
    )

    assert torch.equal(loss, torch.zeros_like(loss))


def test_qwenvl_token_retrieval_loss_penalizes_wrong_embedding_match() -> None:
    student_tokens = torch.tensor([[[1.0, 0.0], [1.0, 0.0]]])
    embedding = torch.tensor([[0.0, 1.0]])
    wrong_embedding = torch.tensor([[1.0, 0.0]])

    loss = qwenvl_token_retrieval_loss(
        student_tokens=student_tokens,
        embedding=embedding,
        wrong_embedding=wrong_embedding,
        margin=0.2,
    )

    assert torch.allclose(loss, torch.tensor(1.2))


def test_qwenvl_token_retrieval_loss_rejects_dim_mismatch() -> None:
    student_tokens = torch.zeros(1, 2, 4)
    embedding = torch.zeros(1, 2)
    wrong_embedding = torch.zeros(1, 2)

    try:
        qwenvl_token_retrieval_loss(
            student_tokens=student_tokens,
            embedding=embedding,
            wrong_embedding=wrong_embedding,
            margin=0.2,
        )
    except SmokeInputError as error:
        assert "embedding dim" in str(error)
    else:
        raise AssertionError("QwenVL token retrieval should reject mismatched dims")
