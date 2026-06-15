from __future__ import annotations

from dataclasses import dataclass

import torch

from training.qwenvl_real_smoke import (
    PreparedQwenVLRow,
    QwenVLEmbeddingModel,
    encode_qwenvl_embedding,
    prepare_qwenvl_training_row,
)
from training.siglip_smoke_data import resolve_pair_paths
from training.siglip_smoke_types import PairRow, SmokeConfig


@dataclass(frozen=True, slots=True)
class PreparedQwenVLTeacherAlignmentRow:
    prepared: PreparedQwenVLRow
    teacher_embedding: torch.Tensor


def prepare_qwenvl_teacher_alignment_row(
    row: PairRow,
    config: SmokeConfig,
    vae,
    text_encoder: torch.nn.Module,
    anima: torch.nn.Module,
    embedder: QwenVLEmbeddingModel,
    prepare_text_inputs,
    device: torch.device,
    dtype: torch.dtype,
    instruction: str,
) -> PreparedQwenVLTeacherAlignmentRow:
    prepared = prepare_qwenvl_training_row(
        row,
        config,
        vae,
        text_encoder,
        anima,
        embedder,
        prepare_text_inputs,
        device,
        dtype,
        instruction,
    )
    paths = resolve_pair_paths(row, config.image_root)
    teacher_embedding = encode_qwenvl_embedding(
        embedder,
        paths.target_image,
        instruction=instruction,
        device=device,
    ).detach()
    return PreparedQwenVLTeacherAlignmentRow(
        prepared=prepared,
        teacher_embedding=teacher_embedding,
    )


def prepare_qwenvl_teacher_alignment_cache(
    rows: list[PairRow],
    config: SmokeConfig,
    vae,
    text_encoder: torch.nn.Module,
    anima: torch.nn.Module,
    embedder: QwenVLEmbeddingModel,
    prepare_text_inputs,
    device: torch.device,
    dtype: torch.dtype,
    instruction: str,
) -> list[PreparedQwenVLTeacherAlignmentRow]:
    return [
        prepare_qwenvl_teacher_alignment_row(
            row,
            config,
            vae,
            text_encoder,
            anima,
            embedder,
            prepare_text_inputs,
            device,
            dtype,
            instruction,
        )
        for row in rows
    ]


def get_qwenvl_teacher_alignment_prepared(
    cache: list[PreparedQwenVLTeacherAlignmentRow] | None,
    rows: list[PairRow],
    row_index: int,
    config: SmokeConfig,
    vae,
    text_encoder: torch.nn.Module,
    anima: torch.nn.Module,
    embedder: QwenVLEmbeddingModel,
    prepare_text_inputs,
    device: torch.device,
    dtype: torch.dtype,
    instruction: str,
) -> PreparedQwenVLTeacherAlignmentRow:
    if cache is not None:
        return cache[row_index]
    return prepare_qwenvl_teacher_alignment_row(
        rows[row_index],
        config,
        vae,
        text_encoder,
        anima,
        embedder,
        prepare_text_inputs,
        device,
        dtype,
        instruction,
    )
