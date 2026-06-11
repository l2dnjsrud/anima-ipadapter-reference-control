from __future__ import annotations

import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
ANIMA_ROOT = Path("/home/wktwin/anima-lora-training-bundle/anima_lora")
for candidate in (ROOT, ANIMA_ROOT):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from training.qwenvl_prepared_cache import (  # noqa: E402
    get_qwenvl_prepared,
    prepare_qwenvl_cache,
)
from training.qwenvl_real_smoke import DEFAULT_INSTRUCTION  # noqa: E402
from training.qwenvl_smoke_checkpoint import (  # noqa: E402
    load_trainable_qwenvl_adapter,
    save_qwenvl_adapter_checkpoint,
    verify_qwenvl_checkpoint,
)
from training.qwenvl_step import QwenVLStepWeights, run_qwenvl_step  # noqa: E402
from training.siglip_real_smoke import freeze_module, trainable_parameter_count  # noqa: E402
from training.siglip_reference_loss import wrong_reference_index  # noqa: E402
from training.siglip_smoke_data import load_pair_rows  # noqa: E402
from training.siglip_smoke_runtime import seed_everything, validate_config  # noqa: E402
from training.siglip_smoke_types import (  # noqa: E402
    CheckpointVerification,
    SmokeConfig,
    SmokeInputError,
)


@dataclass(frozen=True, slots=True)
class QwenVLContrastiveSummary:
    steps: int
    rows_loaded: int
    first_loss: float
    final_loss: float
    mean_loss: float
    mean_base_loss: float
    mean_contrastive_loss: float
    mean_retrieval_loss: float
    finite_loss: bool
    trainable_parameters: int
    frozen_base_parameters: int
    checkpoint: CheckpointVerification
    init_checkpoint_path: str | None
    contrastive_weight: float
    contrastive_margin: float
    retrieval_weight: float
    retrieval_margin: float
    calibrator_bottleneck_dim: int | None


def run_qwenvl_contrastive_smoke(
    config: SmokeConfig,
    *,
    contrastive_weight: float,
    contrastive_margin: float,
    retrieval_weight: float = 0.0,
    retrieval_margin: float = 0.2,
    calibrator_bottleneck_dim: int | None = None,
    instruction: str = DEFAULT_INSTRUCTION,
) -> QwenVLContrastiveSummary:
    validate_config(config)
    if config.max_rows < 2:
        raise SmokeInputError("contrastive smoke requires at least two rows")
    seed_everything(config.seed)
    device = torch.device(config.device)
    dtype = torch.float32
    rows = load_pair_rows(config.manifest_path, limit=config.max_rows)
    if len(rows) < 2:
        raise SmokeInputError("contrastive smoke requires at least two loaded rows")
    random.Random(config.seed).shuffle(rows)

    from library.anima.weights import load_anima_model, load_qwen3_text_encoder
    from library.inference.text import prepare_text_inputs
    from library.models.qwen_vae import load_vae
    from library.runtime.noise import FlowMatchEulerDiscreteScheduler
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

    adapter = load_trainable_qwenvl_adapter(
        config,
        device,
        calibrator_bottleneck_dim=calibrator_bottleneck_dim,
    )
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=config.lr)
    scheduler = FlowMatchEulerDiscreteScheduler(num_train_timesteps=1000, shift=1.0)
    cache = prepare_qwenvl_cache(
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
    losses: list[float] = []
    base_losses: list[float] = []
    contrastive_losses: list[float] = []
    retrieval_losses: list[float] = []
    weights = QwenVLStepWeights(
        contrastive=contrastive_weight,
        retrieval=retrieval_weight,
    )

    for step in range(config.steps):
        row_index = step % len(rows)
        prepared = get_qwenvl_prepared(
            cache,
            rows,
            row_index,
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
        wrong_prepared = get_qwenvl_prepared(
            cache,
            rows,
            wrong_reference_index(row_index, len(rows)),
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
        step_losses = run_qwenvl_step(
            anima=anima,
            adapter=adapter,
            prepared=prepared,
            wrong_prepared=wrong_prepared,
            scheduler=scheduler,
            device=device,
            dtype=dtype,
            weights=weights,
            contrastive_margin=contrastive_margin,
            retrieval_margin=retrieval_margin,
        )
        loss = step_losses.total
        if not torch.isfinite(loss):
            raise SmokeInputError(
                f"non-finite loss at step {step}: {float(loss.detach().cpu())}"
            )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
        base_losses.append(float(step_losses.base.detach().cpu()))
        contrastive_losses.append(float(step_losses.contrastive.detach().cpu()))
        retrieval_losses.append(float(step_losses.retrieval.detach().cpu()))

    save_qwenvl_adapter_checkpoint(adapter, config.output_path)
    checkpoint = verify_qwenvl_checkpoint(config.output_path, config.pe_checkpoint_path)
    return QwenVLContrastiveSummary(
        steps=config.steps,
        rows_loaded=len(rows),
        first_loss=losses[0],
        final_loss=losses[-1],
        mean_loss=sum(losses) / len(losses),
        mean_base_loss=sum(base_losses) / len(base_losses),
        mean_contrastive_loss=sum(contrastive_losses) / len(contrastive_losses),
        mean_retrieval_loss=sum(retrieval_losses) / len(retrieval_losses),
        finite_loss=all(math.isfinite(loss) for loss in losses),
        trainable_parameters=trainable_parameter_count(adapter),
        frozen_base_parameters=frozen_params,
        checkpoint=checkpoint,
        init_checkpoint_path=(
            str(config.init_checkpoint_path) if config.init_checkpoint_path else None
        ),
        contrastive_weight=contrastive_weight,
        contrastive_margin=contrastive_margin,
        retrieval_weight=retrieval_weight,
        retrieval_margin=retrieval_margin,
        calibrator_bottleneck_dim=calibrator_bottleneck_dim,
    )
