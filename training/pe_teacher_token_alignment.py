from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

import torch
from torch.nn import functional as F

from training.siglip_smoke_types import SmokeInputError


class StudentBlockProjector(Protocol):
    def project_kv(self, ip_hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]: ...


class TensorProjector(Protocol):
    def __call__(self, tokens: torch.Tensor) -> torch.Tensor: ...


class SigLIPKVProjector(Protocol):
    num_blocks: int
    ip_cross_attns: Sequence[StudentBlockProjector]


class PEKVProjector(Protocol):
    num_blocks: int
    to_k_ip: Sequence[TensorProjector]
    to_v_ip: Sequence[TensorProjector]


def pe_token_alignment_loss(
    student_adapter: SigLIPKVProjector,
    pe_network: PEKVProjector,
    *,
    student_tokens: torch.Tensor,
    pe_tokens: torch.Tensor,
    block_stride: int,
) -> torch.Tensor:
    """Align SigLIP adapter K/V distributions to the stronger PE teacher."""
    _validate_inputs(
        student_adapter=student_adapter,
        pe_network=pe_network,
        student_tokens=student_tokens,
        pe_tokens=pe_tokens,
        block_stride=block_stride,
    )
    losses: list[torch.Tensor] = []
    for block_idx in range(0, student_adapter.num_blocks, block_stride):
        student_key, student_value = student_adapter.ip_cross_attns[block_idx].project_kv(
            student_tokens
        )
        teacher_key = pe_network.to_k_ip[block_idx](pe_tokens).detach()
        teacher_value = pe_network.to_v_ip[block_idx](pe_tokens).detach()
        losses.append(_descriptor_loss(student_key, teacher_key))
        losses.append(_descriptor_loss(student_value, teacher_value))
    return torch.stack(losses).mean()


def _validate_inputs(
    *,
    student_adapter: SigLIPKVProjector,
    pe_network: PEKVProjector,
    student_tokens: torch.Tensor,
    pe_tokens: torch.Tensor,
    block_stride: int,
) -> None:
    if block_stride < 1:
        raise SmokeInputError("block_stride must be at least 1")
    if student_adapter.num_blocks != pe_network.num_blocks:
        raise SmokeInputError(
            "student and PE teacher must expose the same number of blocks: "
            f"student={student_adapter.num_blocks}, teacher={pe_network.num_blocks}"
        )
    if student_tokens.ndim != 3 or pe_tokens.ndim != 3:
        raise SmokeInputError("student_tokens and pe_tokens must have shape [batch, token, dim]")
    if student_tokens.shape[0] != pe_tokens.shape[0]:
        raise SmokeInputError("student_tokens and pe_tokens must share batch size")


def _descriptor_loss(student_tokens: torch.Tensor, teacher_tokens: torch.Tensor) -> torch.Tensor:
    if student_tokens.shape[0] != teacher_tokens.shape[0]:
        raise SmokeInputError("projected student and teacher tokens must share batch size")
    if student_tokens.shape[-1] != teacher_tokens.shape[-1]:
        raise SmokeInputError(
            "projected student and teacher tokens must share hidden dim: "
            f"student={student_tokens.shape[-1]}, teacher={teacher_tokens.shape[-1]}"
        )
    student = _pooled_descriptor(student_tokens)
    teacher = _pooled_descriptor(teacher_tokens).to(device=student.device, dtype=student.dtype)
    return F.mse_loss(student.float(), teacher.float())


def _pooled_descriptor(tokens: torch.Tensor) -> torch.Tensor:
    tokens = tokens.float()
    mean = tokens.mean(dim=1)
    std = tokens.std(dim=1, unbiased=False)
    return torch.cat([mean, std], dim=-1)
