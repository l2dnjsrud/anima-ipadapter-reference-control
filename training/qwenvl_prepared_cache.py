from __future__ import annotations

import torch

from training.qwenvl_real_smoke import (
    PREPARED_ROW_CACHE_LIMIT,
    PreparedQwenVLRow,
    QwenVLEmbeddingModel,
    prepare_qwenvl_training_row,
)
from training.siglip_smoke_types import PairRow, SmokeConfig


def prepare_qwenvl_cache(
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
) -> list[PreparedQwenVLRow] | None:
    if len(rows) > PREPARED_ROW_CACHE_LIMIT:
        return None
    return [
        prepare_qwenvl_training_row(
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


def get_qwenvl_prepared(
    cache: list[PreparedQwenVLRow] | None,
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
) -> PreparedQwenVLRow:
    if cache is not None:
        return cache[row_index]
    return prepare_qwenvl_training_row(
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
