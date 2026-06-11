from __future__ import annotations

from dataclasses import dataclass

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
