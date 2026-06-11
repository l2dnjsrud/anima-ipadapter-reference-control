from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from pathlib import Path

from training.siglip_smoke_types import CheckpointVerification


@dataclass(frozen=True, slots=True)
class TeacherSmokeSummary:
    steps: int
    rows_loaded: int
    first_loss: float
    final_loss: float
    mean_loss: float
    mean_base_loss: float
    mean_contrastive_loss: float
    mean_teacher_loss: float
    mean_token_loss: float
    mean_pe_token_loss: float
    mean_pe_retrieval_loss: float
    finite_loss: bool
    trainable_parameters: int
    frozen_base_parameters: int
    checkpoint: CheckpointVerification
    init_checkpoint_path: str | None
    contrastive_weight: float
    contrastive_margin: float
    teacher_weight: float
    token_weight: float
    token_max_similarity: float
    pe_token_weight: float
    pe_token_block_stride: int
    pe_retrieval_weight: float
    pe_retrieval_margin: float


def build_teacher_smoke_summary(
    *,
    steps: int,
    rows_loaded: int,
    losses: list[float],
    base_losses: list[float],
    contrastive_losses: list[float],
    teacher_losses: list[float],
    token_losses: list[float],
    pe_token_losses: list[float],
    pe_retrieval_losses: list[float],
    trainable_parameters: int,
    frozen_base_parameters: int,
    checkpoint: CheckpointVerification,
    init_checkpoint_path: Path | None,
    contrastive_weight: float,
    contrastive_margin: float,
    teacher_weight: float,
    token_weight: float,
    token_max_similarity: float,
    pe_token_weight: float,
    pe_token_block_stride: int,
    pe_retrieval_weight: float,
    pe_retrieval_margin: float,
) -> TeacherSmokeSummary:
    return TeacherSmokeSummary(
        steps=steps,
        rows_loaded=rows_loaded,
        first_loss=losses[0],
        final_loss=losses[-1],
        mean_loss=_mean(losses),
        mean_base_loss=_mean(base_losses),
        mean_contrastive_loss=_mean(contrastive_losses),
        mean_teacher_loss=_mean(teacher_losses),
        mean_token_loss=_mean(token_losses),
        mean_pe_token_loss=_mean(pe_token_losses),
        mean_pe_retrieval_loss=_mean(pe_retrieval_losses),
        finite_loss=all(isfinite(loss) for loss in losses),
        trainable_parameters=trainable_parameters,
        frozen_base_parameters=frozen_base_parameters,
        checkpoint=checkpoint,
        init_checkpoint_path=str(init_checkpoint_path) if init_checkpoint_path else None,
        contrastive_weight=contrastive_weight,
        contrastive_margin=contrastive_margin,
        teacher_weight=teacher_weight,
        token_weight=token_weight,
        token_max_similarity=token_max_similarity,
        pe_token_weight=pe_token_weight,
        pe_token_block_stride=pe_token_block_stride,
        pe_retrieval_weight=pe_retrieval_weight,
        pe_retrieval_margin=pe_retrieval_margin,
    )


def _mean(values: list[float]) -> float:
    return sum(values) / len(values)
