from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from pathlib import Path

from training.siglip_smoke_types import CheckpointVerification, SmokeConfig


@dataclass(frozen=True, slots=True)
class QwenVLTeacherAlignmentSummary:
    steps: int
    rows_loaded: int
    first_loss: float
    final_loss: float
    mean_loss: float
    mean_base_loss: float
    mean_contrastive_loss: float
    mean_retrieval_loss: float
    mean_teacher_loss: float
    finite_loss: bool
    trainable_parameters: int
    frozen_base_parameters: int
    checkpoint: CheckpointVerification
    init_checkpoint_path: str | None
    contrastive_weight: float
    contrastive_margin: float
    retrieval_weight: float
    retrieval_margin: float
    teacher_weight: float
    calibrator_bottleneck_dim: int | None
    train_calibrator_only: bool
    explicit_negative_rows: int


def build_qwenvl_teacher_alignment_summary(
    *,
    config: SmokeConfig,
    rows_loaded: int,
    loss_lists: tuple[list[float], list[float], list[float], list[float], list[float]],
    trainable_parameters: int,
    frozen_base_parameters: int,
    checkpoint: CheckpointVerification,
    contrastive_weight: float,
    contrastive_margin: float,
    retrieval_weight: float,
    retrieval_margin: float,
    teacher_weight: float,
    calibrator_bottleneck_dim: int | None,
    train_calibrator_only: bool,
    explicit_negative_rows: int,
) -> QwenVLTeacherAlignmentSummary:
    losses, base_losses, contrastive_losses, retrieval_losses, teacher_losses = loss_lists
    return QwenVLTeacherAlignmentSummary(
        steps=config.steps,
        rows_loaded=rows_loaded,
        first_loss=losses[0],
        final_loss=losses[-1],
        mean_loss=_mean(losses),
        mean_base_loss=_mean(base_losses),
        mean_contrastive_loss=_mean(contrastive_losses),
        mean_retrieval_loss=_mean(retrieval_losses),
        mean_teacher_loss=_mean(teacher_losses),
        finite_loss=all(isfinite(loss) for loss in losses),
        trainable_parameters=trainable_parameters,
        frozen_base_parameters=frozen_base_parameters,
        checkpoint=checkpoint,
        init_checkpoint_path=str(config.init_checkpoint_path) if config.init_checkpoint_path else None,
        contrastive_weight=contrastive_weight,
        contrastive_margin=contrastive_margin,
        retrieval_weight=retrieval_weight,
        retrieval_margin=retrieval_margin,
        teacher_weight=teacher_weight,
        calibrator_bottleneck_dim=calibrator_bottleneck_dim,
        train_calibrator_only=train_calibrator_only,
        explicit_negative_rows=explicit_negative_rows,
    )


def _mean(values: list[float]) -> float:
    return sum(values) / len(values)
