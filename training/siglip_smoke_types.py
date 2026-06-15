from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


MAX_PILOT_STEPS = 1024
MAX_PILOT_ROWS = 2048


@dataclass(frozen=True, slots=True)
class SmokeInputError(Exception):
    detail: str

    def __str__(self) -> str:
        return self.detail


@dataclass(frozen=True, slots=True)
class PairRow:
    ref_id: str
    tgt_id: str
    prompt: str
    neg_id: str | None = None


@dataclass(frozen=True, slots=True)
class PairPaths:
    ref_image: Path
    target_image: Path
    target_caption: Path


@dataclass(frozen=True, slots=True)
class SmokeConfig:
    manifest_path: Path
    image_root: Path
    output_path: Path
    dit_path: Path
    text_encoder_path: Path
    vae_path: Path
    pe_checkpoint_path: Path
    siglip_model_id: str
    device: str
    steps: int
    resolution: int
    lr: float
    seed: int
    max_rows: int
    init_checkpoint_path: Path | None = None


@dataclass(frozen=True, slots=True)
class CheckpointVerification:
    output_path: str
    loadable: bool
    pe_checkpoint_rejected: bool


@dataclass(frozen=True, slots=True)
class SmokeSummary:
    steps: int
    rows_loaded: int
    first_loss: float
    final_loss: float
    mean_loss: float
    finite_loss: bool
    loss_history: tuple[float, ...]
    trainable_parameters: int
    frozen_base_parameters: int
    checkpoint: CheckpointVerification
    init_checkpoint_path: str | None = None
