from __future__ import annotations

from pathlib import Path

import torch
from safetensors.torch import save_file

from siglip_checkpoint import SigLIPCheckpointError, load_siglip_adapter
from siglip_feature_calibration import wrap_siglip_with_calibrator
from siglip_feature_bridge import wrap_siglip_with_feature_bridge
from siglip_model import IPAdapterSigLIP
from training.siglip_smoke_types import (
    CheckpointVerification,
    SmokeConfig,
    SmokeInputError,
)


def load_trainable_adapter(
    config: SmokeConfig,
    device: torch.device,
    dtype: torch.dtype,
    *,
    calibrator_bottleneck_dim: int | None = None,
    train_calibrator_only: bool = False,
    feature_bridge_bottleneck_dim: int | None = None,
    train_feature_bridge_only: bool = False,
) -> IPAdapterSigLIP:
    if calibrator_bottleneck_dim is not None and feature_bridge_bottleneck_dim is not None:
        raise SmokeInputError("SigLIP training cannot combine calibrator and feature bridge")
    adapter = (
        IPAdapterSigLIP()
        if config.init_checkpoint_path is None
        else load_siglip_adapter(config.init_checkpoint_path)
    )
    if calibrator_bottleneck_dim is not None and not hasattr(
        adapter,
        "feature_calibrator",
    ):
        adapter = wrap_siglip_with_calibrator(
            adapter,
            bottleneck_dim=calibrator_bottleneck_dim,
        )
    if feature_bridge_bottleneck_dim is not None and not hasattr(
        adapter,
        "feature_bridge",
    ):
        adapter = wrap_siglip_with_feature_bridge(
            adapter,
            bottleneck_dim=feature_bridge_bottleneck_dim,
        )
    adapter.to(device=device, dtype=torch.float32)
    adapter.train()
    set_siglip_trainable_parameters(
        adapter,
        train_calibrator_only=train_calibrator_only,
        train_feature_bridge_only=train_feature_bridge_only,
    )
    return adapter


def set_siglip_trainable_parameters(
    adapter: IPAdapterSigLIP,
    *,
    train_calibrator_only: bool,
    train_feature_bridge_only: bool,
) -> None:
    if train_calibrator_only and train_feature_bridge_only:
        raise SmokeInputError("choose either train_calibrator_only or train_feature_bridge_only")
    if not train_calibrator_only and not train_feature_bridge_only:
        for parameter in adapter.parameters():
            parameter.requires_grad_(True)
        return
    if not hasattr(adapter, "feature_calibrator"):
        if train_calibrator_only:
            raise ValueError("train_calibrator_only requires a calibrated SigLIP adapter")
    if train_feature_bridge_only and not hasattr(adapter, "feature_bridge"):
        raise SmokeInputError("train_feature_bridge_only requires a bridged SigLIP adapter")
    for parameter in adapter.parameters():
        parameter.requires_grad_(False)
    if train_calibrator_only:
        for parameter in adapter.feature_calibrator.parameters():
            parameter.requires_grad_(True)
    if train_feature_bridge_only:
        for parameter in adapter.feature_bridge.parameters():
            parameter.requires_grad_(True)


def trainable_adapter_parameters(
    adapter: IPAdapterSigLIP,
) -> list[torch.nn.Parameter]:
    parameters = [
        parameter for parameter in adapter.parameters() if parameter.requires_grad
    ]
    if not parameters:
        raise SmokeInputError("SigLIP training has no trainable parameters")
    return parameters


def save_adapter_checkpoint(adapter: IPAdapterSigLIP, output_path: Path) -> None:
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
            "ss_encoder": "siglip2",
            "ss_adapter": "IPAdapterSigLIP",
        },
    )


def verify_checkpoint(
    output_path: Path, pe_checkpoint_path: Path
) -> CheckpointVerification:
    load_siglip_adapter(output_path)
    pe_rejected = False
    try:
        load_siglip_adapter(pe_checkpoint_path)
    except SigLIPCheckpointError:
        pe_rejected = True
    return CheckpointVerification(
        output_path=str(output_path),
        loadable=True,
        pe_checkpoint_rejected=pe_rejected,
    )
