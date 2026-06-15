from __future__ import annotations

import torch

from training.hard_negative_rows import explicit_negative_or_fallback
from training.siglip_real_smoke import (
    PREPARED_ROW_CACHE_LIMIT,
    PreparedTrainingRow,
    prepare_training_row,
)
from training.siglip_reference_loss import wrong_reference_index
from training.siglip_smoke_types import PairRow, SmokeConfig


def prepare_cache(
    rows: list[PairRow],
    config: SmokeConfig,
    vae,
    text_encoder: torch.nn.Module,
    anima: torch.nn.Module,
    siglip: torch.nn.Module,
    processor,
    prepare_text_inputs,
    device: torch.device,
    dtype: torch.dtype,
) -> list[PreparedTrainingRow] | None:
    if len(rows) > PREPARED_ROW_CACHE_LIMIT:
        return None
    return [
        prepare_training_row(
            row,
            config,
            vae,
            text_encoder,
            anima,
            siglip,
            processor,
            prepare_text_inputs,
            device,
            dtype,
        )
        for row in rows
    ]


def get_prepared(
    cache: list[PreparedTrainingRow] | None,
    rows: list[PairRow],
    row_index: int,
    config: SmokeConfig,
    vae,
    text_encoder: torch.nn.Module,
    anima: torch.nn.Module,
    siglip: torch.nn.Module,
    processor,
    prepare_text_inputs,
    device: torch.device,
    dtype: torch.dtype,
) -> PreparedTrainingRow:
    if cache is not None:
        return cache[row_index]
    return prepare_training_row(
        rows[row_index],
        config,
        vae,
        text_encoder,
        anima,
        siglip,
        processor,
        prepare_text_inputs,
        device,
        dtype,
    )


def get_wrong_prepared(
    cache: list[PreparedTrainingRow] | None,
    rows: list[PairRow],
    row_index: int,
    config: SmokeConfig,
    vae,
    text_encoder: torch.nn.Module,
    anima: torch.nn.Module,
    siglip: torch.nn.Module,
    processor,
    prepare_text_inputs,
    device: torch.device,
    dtype: torch.dtype,
) -> PreparedTrainingRow:
    fallback_index = wrong_reference_index(row_index, len(rows))
    fallback_row = rows[fallback_index]
    negative_row = explicit_negative_or_fallback(rows[row_index], fallback_row)
    if negative_row is fallback_row:
        return get_prepared(
            cache,
            rows,
            fallback_index,
            config,
            vae,
            text_encoder,
            anima,
            siglip,
            processor,
            prepare_text_inputs,
            device,
            dtype,
        )
    return prepare_training_row(
        negative_row,
        config,
        vae,
        text_encoder,
        anima,
        siglip,
        processor,
        prepare_text_inputs,
        device,
        dtype,
    )
