from __future__ import annotations

import torch

from training.pe_teacher_token_alignment import pe_token_alignment_loss
from training.siglip_smoke_types import SmokeInputError


def test_pe_token_alignment_loss_is_zero_when_projected_tokens_match() -> None:
    student = _FakeSigLIPProjector(scale=1.0)
    teacher = _FakePEProjector(scale=1.0)
    tokens = torch.ones(1, 3, 4)

    loss = pe_token_alignment_loss(
        student,
        teacher,
        student_tokens=tokens,
        pe_tokens=tokens,
        block_stride=1,
    )

    assert torch.equal(loss, torch.zeros_like(loss))


def test_pe_token_alignment_loss_is_positive_when_projected_tokens_differ() -> None:
    student = _FakeSigLIPProjector(scale=1.0)
    teacher = _FakePEProjector(scale=2.0)
    tokens = torch.ones(1, 3, 4)

    loss = pe_token_alignment_loss(
        student,
        teacher,
        student_tokens=tokens,
        pe_tokens=tokens,
        block_stride=1,
    )

    assert float(loss) > 0.0


def test_pe_token_alignment_loss_rejects_bad_block_stride() -> None:
    student = _FakeSigLIPProjector(scale=1.0)
    teacher = _FakePEProjector(scale=1.0)
    tokens = torch.ones(1, 3, 4)

    try:
        pe_token_alignment_loss(
            student,
            teacher,
            student_tokens=tokens,
            pe_tokens=tokens,
            block_stride=0,
        )
    except SmokeInputError as error:
        assert "block_stride" in str(error)
    else:
        raise AssertionError("zero block stride should fail")


class _FakeSigLIPProjector:
    def __init__(self, *, scale: float) -> None:
        self.num_blocks = 2
        self.ip_cross_attns = [_FakeStudentBlock(scale), _FakeStudentBlock(scale)]


class _FakeStudentBlock:
    def __init__(self, scale: float) -> None:
        self.scale = scale

    def project_kv(self, ip_hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        projected = ip_hidden_states * self.scale
        return projected, projected + 1.0


class _FakePEProjector:
    def __init__(self, *, scale: float) -> None:
        self.num_blocks = 2
        self.to_k_ip = torch.nn.ModuleList([_Scale(scale), _Scale(scale)])
        self.to_v_ip = torch.nn.ModuleList([_ShiftedScale(scale), _ShiftedScale(scale)])


class _Scale(torch.nn.Module):
    def __init__(self, scale: float) -> None:
        super().__init__()
        self.scale = scale

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return tokens * self.scale


class _ShiftedScale(torch.nn.Module):
    def __init__(self, scale: float) -> None:
        super().__init__()
        self.scale = scale

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return tokens * self.scale + 1.0
