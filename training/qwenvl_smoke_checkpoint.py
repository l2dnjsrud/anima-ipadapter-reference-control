from __future__ import annotations

from pathlib import Path

import torch
from safetensors.torch import save_file

from qwenvl_checkpoint import QwenVLCheckpointError, load_qwenvl_adapter
from qwenvl_feature_calibration import wrap_qwenvl_with_calibrator
from qwenvl_model import IPAdapterQwenVL
from training.siglip_smoke_types import CheckpointVerification, SmokeConfig


def load_trainable_qwenvl_adapter(
    config: SmokeConfig,
    device: torch.device,
    *,
    calibrator_bottleneck_dim: int | None = None,
    train_calibrator_only: bool = False,
) -> IPAdapterQwenVL:
    adapter = (
        IPAdapterQwenVL()
        if config.init_checkpoint_path is None
        else load_qwenvl_adapter(config.init_checkpoint_path)
    )
    if calibrator_bottleneck_dim is not None and not hasattr(
        adapter,
        "feature_calibrator",
    ):
        adapter = wrap_qwenvl_with_calibrator(
            adapter,
            bottleneck_dim=calibrator_bottleneck_dim,
        )
    adapter.to(device=device, dtype=torch.float32)
    adapter.train()
    _set_qwenvl_trainable_parameters(
        adapter,
        train_calibrator_only=train_calibrator_only,
    )
    return adapter


def _set_qwenvl_trainable_parameters(
    adapter: IPAdapterQwenVL,
    *,
    train_calibrator_only: bool,
) -> None:
    if not train_calibrator_only:
        for parameter in adapter.parameters():
            parameter.requires_grad_(True)
        return
    if not hasattr(adapter, "feature_calibrator"):
        raise ValueError("train_calibrator_only requires a calibrated QwenVL adapter")
    for parameter in adapter.parameters():
        parameter.requires_grad_(False)
    for parameter in adapter.feature_calibrator.parameters():
        parameter.requires_grad_(True)


def save_qwenvl_adapter_checkpoint(
    adapter: IPAdapterQwenVL, output_path: Path
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    state = {
        key: value.detach().cpu().contiguous()
        for key, value in adapter.state_dict().items()
    }
    save_file(
        state,
        str(output_path),
        metadata={
            "format": "pt",
            "ss_encoder": "qwen3-vl-embedding",
            "ss_adapter": "IPAdapterQwenVL",
        },
    )


def verify_qwenvl_checkpoint(
    output_path: Path, pe_checkpoint_path: Path
) -> CheckpointVerification:
    if not output_path.is_file():
        raise QwenVLCheckpointError(f"QwenVL checkpoint not found: {output_path}")
    load_qwenvl_adapter(output_path)
    pe_rejected = False
    try:
        load_qwenvl_adapter(pe_checkpoint_path)
    except QwenVLCheckpointError:
        pe_rejected = True
    return CheckpointVerification(str(output_path), True, pe_rejected)
