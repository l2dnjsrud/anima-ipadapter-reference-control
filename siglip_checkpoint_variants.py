from __future__ import annotations

from typing import Final, Mapping

import torch

try:
    from .siglip_checkpoint_errors import SigLIPCheckpointError
except ImportError:
    from siglip_checkpoint_errors import SigLIPCheckpointError


TensorState = Mapping[str, torch.Tensor]

CALIBRATOR_KEYS: Final = (
    "feature_calibrator.deep_norm.weight",
    "feature_calibrator.deep_norm.bias",
    "feature_calibrator.deep_down.weight",
    "feature_calibrator.deep_up.weight",
    "feature_calibrator.shallow_norm.weight",
    "feature_calibrator.shallow_norm.bias",
    "feature_calibrator.shallow_down.weight",
    "feature_calibrator.shallow_up.weight",
)
BRIDGE_KEYS: Final = (
    "feature_bridge.norm.weight",
    "feature_bridge.norm.bias",
    "feature_bridge.down.weight",
    "feature_bridge.up.weight",
)


def detect_calibrator_bottleneck(
    state: TensorState, siglip_dim: int, siglip_shallow_dim: int
) -> int | None:
    keys = set(state)
    if not any(key.startswith("feature_calibrator.") for key in keys):
        return None
    _require_keys(keys, CALIBRATOR_KEYS, "calibration")
    bottleneck_dim = _rank2_shape(
        state, "feature_calibrator.deep_down.weight", "calibration"
    )[0]
    _check_shapes(
        state,
        {
            "feature_calibrator.deep_norm.weight": (siglip_dim,),
            "feature_calibrator.deep_norm.bias": (siglip_dim,),
            "feature_calibrator.deep_down.weight": (bottleneck_dim, siglip_dim),
            "feature_calibrator.deep_up.weight": (siglip_dim, bottleneck_dim),
            "feature_calibrator.shallow_norm.weight": (siglip_shallow_dim,),
            "feature_calibrator.shallow_norm.bias": (siglip_shallow_dim,),
            "feature_calibrator.shallow_down.weight": (
                bottleneck_dim,
                siglip_shallow_dim,
            ),
            "feature_calibrator.shallow_up.weight": (
                siglip_shallow_dim,
                bottleneck_dim,
            ),
        },
        "calibration",
    )
    return bottleneck_dim


def detect_feature_bridge_bottleneck(
    state: TensorState,
    token_dim: int,
) -> int | None:
    keys = set(state)
    if not any(key.startswith("feature_bridge.") for key in keys):
        return None
    _require_keys(keys, BRIDGE_KEYS, "feature bridge")
    bottleneck_dim = _rank2_shape(state, "feature_bridge.down.weight", "feature bridge")[0]
    _check_shapes(
        state,
        {
            "feature_bridge.norm.weight": (token_dim,),
            "feature_bridge.norm.bias": (token_dim,),
            "feature_bridge.down.weight": (bottleneck_dim, token_dim),
            "feature_bridge.up.weight": (token_dim, bottleneck_dim),
        },
        "feature bridge",
    )
    return bottleneck_dim


def _require_keys(keys: set[str], required: tuple[str, ...], variant: str) -> None:
    missing = [key for key in required if key not in keys]
    if missing:
        raise SigLIPCheckpointError(
            f"Malformed SigLIP {variant} checkpoint: missing "
            + ", ".join(sorted(missing))
        )


def _rank2_shape(state: TensorState, key: str, variant: str) -> tuple[int, ...]:
    shape = tuple(state[key].shape)
    if len(shape) != 2:
        raise SigLIPCheckpointError(
            f"Malformed SigLIP {variant} checkpoint: {key} must be rank 2"
        )
    return shape


def _check_shapes(
    state: TensorState,
    expected_shapes: Mapping[str, tuple[int, ...]],
    variant: str,
) -> None:
    for key, expected_shape in expected_shapes.items():
        actual_shape = tuple(state[key].shape)
        if actual_shape != expected_shape:
            raise SigLIPCheckpointError(
                f"Malformed SigLIP {variant} checkpoint: {key} shape "
                f"{actual_shape} != {expected_shape}"
            )
