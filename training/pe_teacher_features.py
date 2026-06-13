from __future__ import annotations

from collections.abc import Callable

import torch

from training.hard_negative_rows import explicit_negative_or_fallback
from training.siglip_real_smoke import PREPARED_ROW_CACHE_LIMIT
from training.siglip_reference_loss import wrong_reference_index
from training.siglip_smoke_data import load_anima_pixels, resolve_pair_paths
from training.siglip_smoke_types import PairRow, SmokeConfig

type PEEncodeFn = Callable[..., list[torch.Tensor]]


def prepare_pe_cache(
    rows: list[PairRow],
    config: SmokeConfig,
    pe_encoder,
    encode_pe_from_imageminus1to1: PEEncodeFn,
    device: torch.device,
    dtype: torch.dtype,
) -> list[torch.Tensor] | None:
    if len(rows) > PREPARED_ROW_CACHE_LIMIT:
        return None
    return [
        encode_pe_features(
            row, config, pe_encoder, encode_pe_from_imageminus1to1, device, dtype
        )
        for row in rows
    ]


def get_pe_features(
    cache: list[torch.Tensor] | None,
    rows: list[PairRow],
    row_index: int,
    config: SmokeConfig,
    pe_encoder,
    encode_pe_from_imageminus1to1: PEEncodeFn,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    if cache is not None:
        return cache[row_index]
    row = rows[row_index]
    return encode_pe_features(
        row, config, pe_encoder, encode_pe_from_imageminus1to1, device, dtype
    )


def get_wrong_pe_features(
    cache: list[torch.Tensor] | None,
    rows: list[PairRow],
    row_index: int,
    config: SmokeConfig,
    pe_encoder,
    encode_pe_from_imageminus1to1: PEEncodeFn,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    fallback_index = wrong_reference_index(row_index, len(rows))
    fallback_row = rows[fallback_index]
    negative_row = explicit_negative_or_fallback(rows[row_index], fallback_row)
    if negative_row is fallback_row:
        return get_pe_features(
            cache,
            rows,
            fallback_index,
            config,
            pe_encoder,
            encode_pe_from_imageminus1to1,
            device,
            dtype,
        )
    return encode_pe_features(
        negative_row, config, pe_encoder, encode_pe_from_imageminus1to1, device, dtype
    )


def encode_pe_features(
    row: PairRow,
    config: SmokeConfig,
    pe_encoder,
    encode_pe_from_imageminus1to1: PEEncodeFn,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    paths = resolve_pair_paths(row, config.image_root)
    encoder_dtype = _encoder_dtype(pe_encoder, dtype)
    pixels = load_anima_pixels(paths.ref_image, resolution=config.resolution).to(
        device=device, dtype=encoder_dtype
    )
    with torch.no_grad():
        features = encode_pe_from_imageminus1to1(pe_encoder, pixels, same_bucket=True)
    return torch.stack(features, dim=0).detach().to(device=device, dtype=dtype)


def _encoder_dtype(pe_encoder, fallback: torch.dtype) -> torch.dtype:
    bundle_encoder = getattr(pe_encoder, "encoder", pe_encoder)
    candidate = getattr(bundle_encoder, "inner", bundle_encoder)
    conv1 = getattr(candidate, "conv1", None)
    weight = getattr(conv1, "weight", None)
    if torch.is_tensor(weight):
        return weight.dtype
    if isinstance(candidate, torch.nn.Module):
        parameter = next(candidate.parameters(), None)
        if parameter is not None:
            return parameter.dtype
    bundle_dtype = getattr(pe_encoder, "dtype", None)
    if isinstance(bundle_dtype, torch.dtype):
        return bundle_dtype
    return fallback
