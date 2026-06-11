from __future__ import annotations

import torch

from training.siglip_reference_loss import (
    reference_margin_loss,
    reference_token_separation_loss,
    wrong_reference_index,
)
from training.siglip_smoke_types import SmokeInputError


def test_wrong_reference_index_is_deterministic_and_distinct() -> None:
    row_count = 8

    indices = [wrong_reference_index(index, row_count) for index in range(row_count)]

    assert indices == [4, 5, 6, 7, 0, 1, 2, 3]
    assert all(index != wrong for index, wrong in enumerate(indices))


def test_wrong_reference_index_handles_two_rows() -> None:
    assert wrong_reference_index(0, 2) == 1
    assert wrong_reference_index(1, 2) == 0


def test_wrong_reference_index_rejects_single_row() -> None:
    try:
        wrong_reference_index(0, 1)
    except SmokeInputError as error:
        assert "at least two rows" in str(error)
    else:
        raise AssertionError("single-row contrastive training should fail")


def test_reference_margin_loss_is_zero_when_correct_is_better_by_margin() -> None:
    target = torch.zeros(1, 1, 2, 2)
    correct = torch.zeros_like(target)
    wrong = torch.full_like(target, 2.0)

    loss = reference_margin_loss(correct, wrong, target, margin=0.25)

    assert torch.equal(loss, torch.zeros_like(loss))


def test_reference_margin_loss_positive_when_wrong_is_too_close() -> None:
    target = torch.zeros(1, 1, 2, 2)
    correct = torch.ones_like(target)
    wrong = torch.zeros_like(target)

    loss = reference_margin_loss(correct, wrong, target, margin=0.25)

    assert torch.allclose(loss, torch.tensor(1.25))


def test_reference_token_separation_loss_is_zero_when_references_are_distinct() -> None:
    correct = torch.tensor([[[1.0, 0.0], [1.0, 0.0]]])
    wrong = torch.tensor([[[0.0, 1.0], [0.0, 1.0]]])

    loss = reference_token_separation_loss(correct, wrong, max_similarity=0.2)

    assert torch.equal(loss, torch.zeros_like(loss))


def test_reference_token_separation_loss_penalizes_collapsed_tokens() -> None:
    correct = torch.tensor([[[1.0, 0.0], [1.0, 0.0]]])
    wrong = correct.clone()

    loss = reference_token_separation_loss(correct, wrong, max_similarity=0.2)

    assert torch.allclose(loss, torch.tensor(0.8))


def test_reference_token_separation_loss_rejects_shape_mismatch() -> None:
    correct = torch.zeros(1, 2, 3)
    wrong = torch.zeros(1, 3, 3)

    try:
        reference_token_separation_loss(correct, wrong, max_similarity=0.2)
    except SmokeInputError as error:
        assert "must match" in str(error)
    else:
        raise AssertionError("shape mismatch should fail")
