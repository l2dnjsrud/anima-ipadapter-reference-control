# /// script
# dependencies = [
#   "pillow",
#   "rich",
#   "safetensors",
#   "sentence-transformers",
#   "torch",
#   "typer",
# ]
# ///
# --- How to run -----------------------------------------------------
# /home/wktwin/anima-lora-training-bundle/anima_lora/.venv/bin/python training/qwenvl_teacher_alignment_cli.py \
#   --manifest-path training/manifests/c108_qwen_teacher_embedding_alignment_20260615.jsonl \
#   --image-root .tmp/c097_siglip_hard_shape_expanded_root \
#   --init-checkpoint-path checkpoints/anima_qwenvl_ip_adapter_single_character_retrieval_0128_20260611.safetensors \
#   --steps 128 --resolution 256 --device cuda:0 \
#   --output-path checkpoints/anima_qwenvl_ip_adapter_c108_teacher_alignment_20260615.safetensors

from __future__ import annotations

import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Annotated, Final

import typer
from rich.console import Console

ROOT: Final[Path] = Path(__file__).resolve().parents[1]
ANIMA_ROOT: Final[Path] = Path("/home/wktwin/anima-lora-training-bundle/anima_lora")
for candidate in (ROOT, ANIMA_ROOT):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from training.qwenvl_real_smoke import DEFAULT_INSTRUCTION, DEFAULT_QWENVL  # noqa: E402
from training.qwenvl_teacher_alignment_smoke import (  # noqa: E402
    run_qwenvl_teacher_alignment_smoke,
)
from training.siglip_smoke_types import SmokeConfig  # noqa: E402


DEFAULT_DIT: Final[Path] = ANIMA_ROOT / "models/diffusion_models/anima-base-v1.0.safetensors"
DEFAULT_TEXT: Final[Path] = ANIMA_ROOT / "models/text_encoders/qwen_3_06b_base.safetensors"
DEFAULT_VAE: Final[Path] = ANIMA_ROOT / "models/vae/qwen_image_vae.safetensors"
DEFAULT_PE: Final[Path] = Path("/data/ai/models/ipadapter/anima_ip_adapter_quality_20260610.safetensors")
DEFAULT_OUTPUT: Final[Path] = ROOT / "checkpoints/anima_qwenvl_ip_adapter_c108_teacher_alignment_20260615.safetensors"

app = typer.Typer(add_completion=False)
console = Console()


@app.command()
def train(
    manifest_path: Annotated[Path, typer.Option()],
    image_root: Annotated[Path, typer.Option()],
    init_checkpoint_path: Annotated[Path, typer.Option()],
    output_path: Annotated[Path, typer.Option()] = DEFAULT_OUTPUT,
    steps: Annotated[int, typer.Option()] = 128,
    resolution: Annotated[int, typer.Option()] = 256,
    device: Annotated[str, typer.Option()] = "cuda:0",
    max_rows: Annotated[int, typer.Option()] = 56,
    lr: Annotated[float, typer.Option()] = 1e-5,
    seed: Annotated[int, typer.Option()] = 20260808,
    contrastive_weight: Annotated[float, typer.Option()] = 0.35,
    contrastive_margin: Annotated[float, typer.Option()] = 0.05,
    retrieval_weight: Annotated[float, typer.Option()] = 0.15,
    retrieval_margin: Annotated[float, typer.Option()] = 0.2,
    teacher_weight: Annotated[float, typer.Option()] = 0.4,
    calibrator_bottleneck_dim: Annotated[int | None, typer.Option()] = 128,
    train_calibrator_only: Annotated[bool, typer.Option()] = True,
    qwenvl_model_id: Annotated[str, typer.Option()] = DEFAULT_QWENVL,
    instruction: Annotated[str, typer.Option()] = DEFAULT_INSTRUCTION,
) -> None:
    config = SmokeConfig(
        manifest_path=manifest_path,
        image_root=image_root,
        output_path=output_path,
        dit_path=DEFAULT_DIT,
        text_encoder_path=DEFAULT_TEXT,
        vae_path=DEFAULT_VAE,
        pe_checkpoint_path=DEFAULT_PE,
        siglip_model_id=qwenvl_model_id,
        device=device,
        steps=steps,
        resolution=resolution,
        lr=lr,
        seed=seed,
        max_rows=max_rows,
        init_checkpoint_path=init_checkpoint_path,
    )
    summary = run_qwenvl_teacher_alignment_smoke(
        config,
        contrastive_weight=contrastive_weight,
        contrastive_margin=contrastive_margin,
        retrieval_weight=retrieval_weight,
        retrieval_margin=retrieval_margin,
        teacher_weight=teacher_weight,
        calibrator_bottleneck_dim=calibrator_bottleneck_dim,
        train_calibrator_only=train_calibrator_only,
        instruction=instruction,
    )
    console.print_json(json.dumps(asdict(summary), indent=2))


if __name__ == "__main__":
    app()
