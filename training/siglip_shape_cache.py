from __future__ import annotations

import torch

from training.siglip_real_smoke import PREPARED_ROW_CACHE_LIMIT
from training.siglip_smoke_data import resolve_pair_paths
from training.siglip_smoke_runtime import encode_target_latents
from training.siglip_smoke_types import PairRow, SmokeConfig


def prepare_reference_latent_cache(
    rows: list[PairRow],
    config: SmokeConfig,
    vae,
    device: torch.device,
    dtype: torch.dtype,
) -> list[torch.Tensor] | None:
    if len(rows) > PREPARED_ROW_CACHE_LIMIT:
        return None
    return [
        encode_reference_latents(row, config, vae, device, dtype)
        for row in rows
    ]


def get_reference_latents(
    cache: list[torch.Tensor] | None,
    rows: list[PairRow],
    row_index: int,
    config: SmokeConfig,
    vae,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    if cache is not None:
        return cache[row_index]
    return encode_reference_latents(rows[row_index], config, vae, device, dtype)


def encode_reference_latents(
    row: PairRow,
    config: SmokeConfig,
    vae,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    paths = resolve_pair_paths(row, config.image_root)
    latents = encode_target_latents(
        vae,
        paths.ref_image,
        config.resolution,
        device,
        dtype,
    )
    return latents.detach()
