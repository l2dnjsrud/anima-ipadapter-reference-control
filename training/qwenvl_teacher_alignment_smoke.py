from __future__ import annotations

import random
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
ANIMA_ROOT = Path("/home/wktwin/anima-lora-training-bundle/anima_lora")
for candidate in (ROOT, ANIMA_ROOT):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from training.hard_negative_rows import explicit_negative_or_fallback  # noqa: E402
from training.qwenvl_real_smoke import DEFAULT_INSTRUCTION, PREPARED_ROW_CACHE_LIMIT  # noqa: E402
from training.qwenvl_smoke_checkpoint import (  # noqa: E402
    load_trainable_qwenvl_adapter,
    save_qwenvl_adapter_checkpoint,
    verify_qwenvl_checkpoint,
)
from training.qwenvl_teacher_alignment_rows import (  # noqa: E402
    get_qwenvl_teacher_alignment_prepared,
    prepare_qwenvl_teacher_alignment_cache,
    prepare_qwenvl_teacher_alignment_row,
)
from training.qwenvl_teacher_alignment_step import (  # noqa: E402
    QwenVLTeacherAlignmentStepWeights,
    run_qwenvl_teacher_alignment_step,
)
from training.qwenvl_teacher_alignment_summary import (  # noqa: E402
    QwenVLTeacherAlignmentSummary,
    build_qwenvl_teacher_alignment_summary,
)
from training.siglip_real_smoke import freeze_module, trainable_parameter_count  # noqa: E402
from training.siglip_reference_loss import wrong_reference_index  # noqa: E402
from training.siglip_smoke_data import load_pair_rows  # noqa: E402
from training.siglip_smoke_runtime import seed_everything, validate_config  # noqa: E402
from training.siglip_smoke_types import (  # noqa: E402
    SmokeConfig,
    SmokeInputError,
)


def run_qwenvl_teacher_alignment_smoke(
    config: SmokeConfig,
    *,
    contrastive_weight: float,
    contrastive_margin: float,
    retrieval_weight: float,
    retrieval_margin: float,
    teacher_weight: float,
    calibrator_bottleneck_dim: int | None,
    train_calibrator_only: bool,
    instruction: str = DEFAULT_INSTRUCTION,
) -> QwenVLTeacherAlignmentSummary:
    validate_config(config)
    if config.max_rows < 2:
        raise SmokeInputError("teacher alignment smoke requires at least two rows")
    seed_everything(config.seed)
    device = torch.device(config.device)
    dtype = torch.float32
    rows = load_pair_rows(config.manifest_path, limit=config.max_rows)
    if len(rows) < 2:
        raise SmokeInputError("teacher alignment smoke requires at least two loaded rows")
    explicit_negative_rows = sum(1 for row in rows if row.neg_id is not None)
    random.Random(config.seed).shuffle(rows)

    runtime, frozen_base_parameters = _load_runtime(config, device, dtype)
    adapter = load_trainable_qwenvl_adapter(
        config,
        device,
        calibrator_bottleneck_dim=calibrator_bottleneck_dim,
        train_calibrator_only=train_calibrator_only,
    )
    trainable_parameters = [
        parameter for parameter in adapter.parameters() if parameter.requires_grad
    ]
    if not trainable_parameters:
        raise SmokeInputError("QwenVL teacher alignment has no trainable parameters")
    optimizer = torch.optim.AdamW(trainable_parameters, lr=config.lr)
    from library.runtime.noise import FlowMatchEulerDiscreteScheduler

    scheduler = FlowMatchEulerDiscreteScheduler(num_train_timesteps=1000, shift=1.0)
    cache = (
        prepare_qwenvl_teacher_alignment_cache(rows, config, *runtime, device, dtype, instruction)
        if len(rows) <= PREPARED_ROW_CACHE_LIMIT
        else None
    )
    loss_lists: tuple[list[float], list[float], list[float], list[float], list[float]] = (
        [],
        [],
        [],
        [],
        [],
    )
    weights = QwenVLTeacherAlignmentStepWeights(
        contrastive=contrastive_weight,
        retrieval=retrieval_weight,
        teacher=teacher_weight,
    )
    for step in range(config.steps):
        row_index = step % len(rows)
        prepared = get_qwenvl_teacher_alignment_prepared(
            cache, rows, row_index, config, *runtime, device, dtype, instruction
        )
        wrong_prepared = _wrong_prepared(
            cache, rows, row_index, config, runtime, device, dtype, instruction
        )
        step_losses = run_qwenvl_teacher_alignment_step(
            anima=runtime[2],
            adapter=adapter,
            prepared=prepared.prepared,
            teacher_embedding=prepared.teacher_embedding,
            wrong_prepared=wrong_prepared.prepared,
            scheduler=scheduler,
            device=device,
            dtype=dtype,
            weights=weights,
            contrastive_margin=contrastive_margin,
            retrieval_margin=retrieval_margin,
        )
        _apply_step(step_losses.total, optimizer, step)
        _append_losses(loss_lists, step_losses)

    save_qwenvl_adapter_checkpoint(adapter, config.output_path)
    checkpoint = verify_qwenvl_checkpoint(config.output_path, config.pe_checkpoint_path)
    return build_qwenvl_teacher_alignment_summary(
        config=config,
        rows_loaded=len(rows),
        loss_lists=loss_lists,
        trainable_parameters=trainable_parameter_count(adapter),
        frozen_base_parameters=frozen_base_parameters,
        checkpoint=checkpoint,
        contrastive_weight=contrastive_weight,
        contrastive_margin=contrastive_margin,
        retrieval_weight=retrieval_weight,
        retrieval_margin=retrieval_margin,
        teacher_weight=teacher_weight,
        calibrator_bottleneck_dim=calibrator_bottleneck_dim,
        train_calibrator_only=train_calibrator_only,
        explicit_negative_rows=explicit_negative_rows,
    )


def _load_runtime(config: SmokeConfig, device: torch.device, dtype: torch.dtype):
    from library.anima.weights import load_anima_model, load_qwen3_text_encoder
    from library.inference.text import prepare_text_inputs
    from library.models.qwen_vae import load_vae
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
    return (vae, text_encoder, anima, embedder, prepare_text_inputs), frozen_params


def _wrong_prepared(cache, rows, row_index, config, runtime, device, dtype, instruction):
    fallback_index = wrong_reference_index(row_index, len(rows))
    fallback_row = rows[fallback_index]
    negative_row = explicit_negative_or_fallback(rows[row_index], fallback_row)
    if negative_row is fallback_row:
        return get_qwenvl_teacher_alignment_prepared(
            cache, rows, fallback_index, config, *runtime, device, dtype, instruction
        )
    return prepare_qwenvl_teacher_alignment_row(
        negative_row, config, *runtime, device, dtype, instruction
    )


def _apply_step(loss: torch.Tensor, optimizer: torch.optim.Optimizer, step: int) -> None:
    if not torch.isfinite(loss):
        raise SmokeInputError(f"non-finite loss at step {step}: {float(loss.detach().cpu())}")
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()


def _append_losses(loss_lists, step_losses) -> None:
    values = (
        step_losses.total,
        step_losses.base,
        step_losses.contrastive,
        step_losses.retrieval,
        step_losses.teacher,
    )
    for losses, value in zip(loss_lists, values, strict=True):
        losses.append(float(value.detach().cpu()))
