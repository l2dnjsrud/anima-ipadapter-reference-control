from __future__ import annotations

import torch

from training.siglip_teacher_step import TeacherLossWeights, compute_teacher_step_losses


def test_compute_teacher_step_losses_includes_pe_retrieval_weight() -> None:
    target = torch.zeros(1, 1, 1, 1)
    matching_tokens = torch.tensor([[[1.0, 0.0], [1.0, 0.0]]])
    wrong_pe_tokens = torch.tensor([[[0.0, 1.0], [0.0, 1.0]]])

    losses = compute_teacher_step_losses(
        adapter=_FakeSigLIPProjector(),
        pe_network=_FakePEProjector(),
        correct_pred=target,
        wrong_pred=target,
        target=target,
        teacher_pred=target,
        correct_tokens=matching_tokens,
        wrong_tokens=matching_tokens,
        pe_tokens=wrong_pe_tokens,
        wrong_pe_tokens=matching_tokens,
        weights=TeacherLossWeights(
            contrastive=0.0,
            teacher=0.0,
            token=0.0,
            pe_token=0.0,
            pe_retrieval=0.5,
        ),
        contrastive_margin=0.05,
        token_max_similarity=0.2,
        pe_token_block_stride=1,
        pe_retrieval_margin=0.2,
    )

    assert torch.allclose(losses.pe_retrieval, torch.tensor(1.2))
    assert torch.allclose(losses.total, torch.tensor(0.6))


def test_compute_teacher_step_losses_skips_pe_retrieval_when_weight_is_zero() -> None:
    target = torch.zeros(1, 1, 1, 1)
    student_tokens = torch.zeros(1, 2, 4)
    pe_tokens = torch.zeros(1, 2, 2)

    losses = compute_teacher_step_losses(
        adapter=_FakeSigLIPProjector(),
        pe_network=_FakePEProjector(),
        correct_pred=target,
        wrong_pred=target,
        target=target,
        teacher_pred=target,
        correct_tokens=student_tokens,
        wrong_tokens=student_tokens,
        pe_tokens=pe_tokens,
        wrong_pe_tokens=pe_tokens,
        weights=TeacherLossWeights(
            contrastive=0.0,
            teacher=0.0,
            token=0.0,
            pe_token=0.0,
            pe_retrieval=0.0,
        ),
        contrastive_margin=0.05,
        token_max_similarity=0.2,
        pe_token_block_stride=1,
        pe_retrieval_margin=0.2,
    )

    assert torch.equal(losses.pe_retrieval, torch.zeros_like(losses.pe_retrieval))


class _FakeSigLIPProjector:
    def __init__(self) -> None:
        self.num_blocks = 1
        self.ip_cross_attns = [_FakeStudentBlock()]


class _FakeStudentBlock:
    def project_kv(self, ip_hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return ip_hidden_states, ip_hidden_states


class _FakePEProjector:
    def __init__(self) -> None:
        self.num_blocks = 1
        self.to_k_ip = torch.nn.ModuleList([torch.nn.Identity()])
        self.to_v_ip = torch.nn.ModuleList([torch.nn.Identity()])
