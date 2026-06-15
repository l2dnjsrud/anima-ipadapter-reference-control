# /// script
# dependencies = [
#   "rich",
#   "safetensors",
#   "torch",
#   "typer",
# ]
# ///
# --- How to run -----------------------------------------------------
# /home/wktwin/anima-lora-training-bundle/anima_lora/.venv/bin/python tools/merge_qwenvl_checkpoints.py \
#   --base-checkpoint-path checkpoints/anima_qwenvl_ip_adapter_single_character_retrieval_0128_20260611.safetensors \
#   --update-checkpoint-path checkpoints/anima_qwenvl_ip_adapter_c055_mixed_retrieval_0064_20260612.safetensors \
#   --output-dir checkpoints \
#   --output-prefix anima_qwenvl_ip_adapter_c059_merge_prev_c055 \
#   --alpha 0.25 \
#   --alpha 0.40 \
#   --summary-path eval/qwenvl_c059_checkpoint_merge_gate_20260612/merge_summary.json

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Annotated, Final

import torch
import typer
from rich.console import Console
from safetensors.torch import load_file, save_file

type TensorState = dict[str, torch.Tensor]

QWENVL_METADATA: Final = {
    "format": "pt",
    "ss_encoder": "qwen3-vl-embedding",
    "ss_adapter": "IPAdapterQwenVL",
    "ss_merge": "parameter_interpolation",
}

app = typer.Typer(add_completion=False)
console = Console()


@dataclass(frozen=True, slots=True)
class CheckpointMergeError(Exception):
    reason: str

    def __str__(self) -> str:
        return self.reason


@dataclass(frozen=True, slots=True)
class MergeStats:
    alpha: float
    tensor_count: int
    float_tensor_count: int
    skipped_non_float_count: int


@dataclass(frozen=True, slots=True)
class MergeOutputSummary:
    alpha: float
    output_path: str
    file_size_bytes: int
    tensor_count: int
    float_tensor_count: int
    skipped_non_float_count: int


@dataclass(frozen=True, slots=True)
class MergeFileSummary:
    base_checkpoint_path: str
    update_checkpoint_path: str
    outputs: tuple[MergeOutputSummary, ...]


def merge_checkpoint_states(
    base_state: TensorState,
    update_state: TensorState,
    *,
    alpha: float,
) -> tuple[TensorState, MergeStats]:
    """Interpolate compatible tensor states as base * (1-alpha) + update * alpha."""
    _validate_alpha(alpha)
    _assert_same_keys(base_state, update_state)
    merged: TensorState = {}
    float_tensor_count = 0
    skipped_non_float_count = 0

    for key in sorted(base_state):
        base_tensor = base_state[key]
        update_tensor = update_state[key]
        if base_tensor.shape != update_tensor.shape:
            raise CheckpointMergeError(
                f"shape mismatch for {key}: {tuple(base_tensor.shape)} != "
                f"{tuple(update_tensor.shape)}"
            )
        if base_tensor.is_floating_point() and update_tensor.is_floating_point():
            merged[key] = (
                base_tensor.detach().float() * (1.0 - alpha)
                + update_tensor.detach().float() * alpha
            ).to(dtype=base_tensor.dtype)
            float_tensor_count += 1
        else:
            if not torch.equal(base_tensor, update_tensor):
                raise CheckpointMergeError(f"non-floating tensor mismatch for {key}")
            merged[key] = base_tensor.detach().clone()
            skipped_non_float_count += 1

    return (
        merged,
        MergeStats(
            alpha=alpha,
            tensor_count=len(merged),
            float_tensor_count=float_tensor_count,
            skipped_non_float_count=skipped_non_float_count,
        ),
    )


def merge_checkpoint_files(
    *,
    base_checkpoint_path: Path,
    update_checkpoint_path: Path,
    output_dir: Path,
    output_prefix: str,
    alphas: tuple[float, ...],
    summary_path: Path | None = None,
) -> MergeFileSummary:
    base_state = load_file(str(base_checkpoint_path))
    update_state = load_file(str(update_checkpoint_path))
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs: list[MergeOutputSummary] = []

    for alpha in alphas:
        merged, stats = merge_checkpoint_states(base_state, update_state, alpha=alpha)
        output_path = output_dir / f"{output_prefix}_{_alpha_suffix(alpha)}.safetensors"
        save_file(merged, str(output_path), metadata=QWENVL_METADATA)
        del merged
        outputs.append(
            MergeOutputSummary(
                alpha=stats.alpha,
                output_path=str(output_path),
                file_size_bytes=output_path.stat().st_size,
                tensor_count=stats.tensor_count,
                float_tensor_count=stats.float_tensor_count,
                skipped_non_float_count=stats.skipped_non_float_count,
            )
        )

    summary = MergeFileSummary(
        base_checkpoint_path=str(base_checkpoint_path),
        update_checkpoint_path=str(update_checkpoint_path),
        outputs=tuple(outputs),
    )
    if summary_path is not None:
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(
            json.dumps(asdict(summary), indent=2) + "\n",
            encoding="utf-8",
        )
    return summary


@app.command()
def merge(
    base_checkpoint_path: Annotated[Path, typer.Option()],
    update_checkpoint_path: Annotated[Path, typer.Option()],
    output_dir: Annotated[Path, typer.Option()],
    output_prefix: Annotated[str, typer.Option()],
    alpha: Annotated[list[float] | None, typer.Option("--alpha")] = None,
    summary_path: Annotated[Path | None, typer.Option()] = None,
) -> None:
    alphas = tuple(alpha or [0.25])
    summary = merge_checkpoint_files(
        base_checkpoint_path=base_checkpoint_path,
        update_checkpoint_path=update_checkpoint_path,
        output_dir=output_dir,
        output_prefix=output_prefix,
        alphas=alphas,
        summary_path=summary_path,
    )
    console.print_json(json.dumps(asdict(summary), indent=2))


def _validate_alpha(alpha: float) -> None:
    if not 0.0 <= alpha <= 1.0:
        raise CheckpointMergeError(f"alpha must be between 0 and 1, got {alpha}")


def _assert_same_keys(base_state: TensorState, update_state: TensorState) -> None:
    base_keys = set(base_state)
    update_keys = set(update_state)
    if base_keys == update_keys:
        return
    missing = sorted(base_keys - update_keys)[:5]
    extra = sorted(update_keys - base_keys)[:5]
    raise CheckpointMergeError(
        f"key mismatch: missing_in_update={missing}, extra_in_update={extra}"
    )


def _alpha_suffix(alpha: float) -> str:
    return f"a{int(round(alpha * 1000)):04d}"


if __name__ == "__main__":
    app()
