from __future__ import annotations

import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Protocol, TypedDict

import torch
from PIL import Image

ROOT: Final[Path] = Path(__file__).resolve().parents[1]
ANIMA_ROOT: Final[Path] = Path("/home/wktwin/anima-lora-training-bundle/anima_lora")
for candidate in (ROOT, ANIMA_ROOT):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from training.siglip_real_smoke import freeze_module, trainable_parameter_count  # noqa: E402
from training.siglip_smoke_data import load_pair_rows, load_siglip_image, resolve_pair_paths  # noqa: E402
from training.siglip_smoke_patch import patched_cross_attention  # noqa: E402
from training.siglip_smoke_runtime import (  # noqa: E402
    encode_prompt,
    encode_target_latents,
    noise_args,
    seed_everything,
    validate_config,
)
from training.siglip_smoke_types import (  # noqa: E402
    PairRow,
    SmokeConfig,
    SmokeInputError,
    SmokeSummary,
)
from training.qwenvl_smoke_checkpoint import (  # noqa: E402
    load_trainable_qwenvl_adapter,
    save_qwenvl_adapter_checkpoint,
    verify_qwenvl_checkpoint,
)


DEFAULT_QWENVL: Final[str] = "Qwen/Qwen3-VL-Embedding-2B"
DEFAULT_INSTRUCTION: Final[str] = (
    "Represent this manhwa/anime reference image for visual style, color palette, "
    "composition, character identity, and panel layout."
)
PREPARED_ROW_CACHE_LIMIT: Final[int] = 128


class QwenVLImageInput(TypedDict):
    image: Image.Image


class QwenVLEmbeddingModel(Protocol):
    def encode(
        self,
        inputs: list[QwenVLImageInput],
        *,
        normalize_embeddings: bool,
        convert_to_tensor: bool,
        prompt: str,
    ) -> torch.Tensor: ...


@dataclass(frozen=True, slots=True)
class PreparedQwenVLRow:
    latents: torch.Tensor
    crossattn_emb: torch.Tensor
    embedding: torch.Tensor


def encode_qwenvl_embedding(
    model: QwenVLEmbeddingModel,
    image_path: Path,
    *,
    instruction: str,
    device: torch.device,
) -> torch.Tensor:
    image = load_siglip_image(image_path)
    with torch.no_grad():
        raw = model.encode(
            [{"image": image}],
            normalize_embeddings=True,
            convert_to_tensor=True,
            prompt=instruction,
        )
    embedding = torch.as_tensor(raw).detach().float()
    if embedding.ndim == 1:
        embedding = embedding.unsqueeze(0)
    if embedding.ndim != 2:
        raise SmokeInputError("QwenVL embedding must be rank 1 or rank 2")
    return embedding.to(device=device)


def prepare_qwenvl_training_row(
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
) -> PreparedQwenVLRow:
    paths = resolve_pair_paths(row, config.image_root)
    latents = encode_target_latents(
        vae, paths.target_image, config.resolution, device, dtype
    ).detach()
    crossattn_emb = encode_prompt(
        row.prompt,
        config.text_encoder_path,
        text_encoder,
        anima,
        prepare_text_inputs,
        device,
        dtype,
    ).detach()
    embedding = encode_qwenvl_embedding(
        embedder, paths.ref_image, instruction=instruction, device=device
    ).detach()
    return PreparedQwenVLRow(latents, crossattn_emb, embedding)


def run_qwenvl_smoke(
    config: SmokeConfig, *, instruction: str = DEFAULT_INSTRUCTION
) -> SmokeSummary:
    validate_config(config)
    seed_everything(config.seed)
    device = torch.device(config.device)
    dtype = torch.float32
    rows = load_pair_rows(config.manifest_path, limit=config.max_rows)
    random.Random(config.seed).shuffle(rows)

    from library.anima.weights import load_anima_model, load_qwen3_text_encoder
    from library.inference.text import prepare_text_inputs
    from library.models.qwen_vae import load_vae
    from library.runtime.noise import (
        FlowMatchEulerDiscreteScheduler,
        get_noisy_model_input_and_timesteps,
    )
    from sentence_transformers import SentenceTransformer

    anima = load_anima_model(device, str(config.dit_path), "torch", device, dtype)
    anima.to(device=device, dtype=dtype)
    frozen_params = freeze_module(anima)
    vae = load_vae(str(config.vae_path), device=device, dtype=dtype, eval=True)
    frozen_params += freeze_module(vae)
    text_encoder, _ = load_qwen3_text_encoder(
        str(config.text_encoder_path), dtype=dtype, device=str(device)
    )
    frozen_params += freeze_module(text_encoder)
    embedder = SentenceTransformer(
        config.siglip_model_id,
        device=str(device),
        model_kwargs={"torch_dtype": torch.bfloat16} if device.type == "cuda" else {},
    )
    frozen_params += freeze_module(embedder)

    adapter = load_trainable_qwenvl_adapter(config, device)
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=config.lr)
    scheduler = FlowMatchEulerDiscreteScheduler(num_train_timesteps=1000, shift=1.0)
    losses: list[float] = []
    prepared_cache = (
        _prepare_cache(
            rows,
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
        if len(rows) <= PREPARED_ROW_CACHE_LIMIT
        else None
    )

    for step in range(config.steps):
        row_index = step % len(rows)
        prepared = (
            prepared_cache[row_index]
            if prepared_cache is not None
            else prepare_qwenvl_training_row(
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
        )
        latents = prepared.latents
        noise = torch.randn_like(latents)
        noisy, timesteps, _sigmas = get_noisy_model_input_and_timesteps(
            noise_args(), scheduler, latents, noise, device, dtype
        )
        image_tokens = adapter.encode_ref(prepared.embedding, timestep=timesteps)
        padding_mask = torch.zeros(
            latents.shape[0], 1, latents.shape[-2], latents.shape[-1], device=device, dtype=dtype
        )
        optimizer.zero_grad(set_to_none=True)
        with patched_cross_attention(anima, adapter, image_tokens):
            model_pred = anima(
                noisy.unsqueeze(2),
                timesteps,
                prepared.crossattn_emb,
                padding_mask=padding_mask,
            )
        target = noise - latents
        loss = torch.nn.functional.mse_loss(
            model_pred.squeeze(2).float(), target.float()
        )
        if not torch.isfinite(loss):
            raise SmokeInputError(
                f"non-finite loss at step {step}: {float(loss.detach().cpu())}"
            )
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu()))

    save_qwenvl_adapter_checkpoint(adapter, config.output_path)
    checkpoint = verify_qwenvl_checkpoint(config.output_path, config.pe_checkpoint_path)
    return SmokeSummary(
        steps=config.steps,
        rows_loaded=len(rows),
        first_loss=losses[0],
        final_loss=losses[-1],
        mean_loss=sum(losses) / len(losses),
        finite_loss=all(math.isfinite(loss) for loss in losses),
        loss_history=tuple(losses),
        trainable_parameters=trainable_parameter_count(adapter),
        frozen_base_parameters=frozen_params,
        checkpoint=checkpoint,
        init_checkpoint_path=(
            str(config.init_checkpoint_path) if config.init_checkpoint_path else None
        ),
    )


def _prepare_cache(
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
) -> list[PreparedQwenVLRow]:
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
