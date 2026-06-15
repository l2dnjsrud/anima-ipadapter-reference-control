# /// script
# dependencies = [
#   "numpy",
#   "pillow",
#   "safetensors",
#   "torch",
#   "transformers",
#   "typer",
#   "rich",
# ]
# ///
# --- How to run -----------------------------------------------------
# /home/wktwin/anima-lora-training-bundle/anima_lora/.venv/bin/python training/siglip_teacher_cli.py \
#   --manifest-path training/manifests/local_color_self_identity128_20260611.jsonl \
#   --image-root /home/wktwin/anima-lora-training-bundle/image_dataset_color_panel_style_v5_best \
#   --steps 32 --max-rows 8 --resolution 256 --device cuda:0 \
#   --output-path checkpoints/anima_siglip_ip_adapter_teacher_smoke.safetensors

from __future__ import annotations

import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.siglip_real_smoke import (  # noqa: E402
    DEFAULT_DIT,
    DEFAULT_OUTPUT,
    DEFAULT_PE,
    DEFAULT_SIGLIP,
    DEFAULT_TEXT,
    DEFAULT_VAE,
)
from training.siglip_smoke_types import MAX_PILOT_ROWS, MAX_PILOT_STEPS, SmokeConfig  # noqa: E402
from training.siglip_teacher_smoke import run_teacher_smoke  # noqa: E402

app = typer.Typer(add_completion=False)
console = Console()


@app.command()
def main(
    manifest_path: Annotated[Path, typer.Option()] = Path(
        "training/manifests/local_color_self_identity128_20260611.jsonl"
    ),
    image_root: Annotated[Path, typer.Option()] = Path(
        "/home/wktwin/anima-lora-training-bundle/image_dataset_color_panel_style_v5_best"
    ),
    output_path: Annotated[Path, typer.Option()] = DEFAULT_OUTPUT,
    dit_path: Annotated[Path, typer.Option()] = DEFAULT_DIT,
    text_encoder_path: Annotated[Path, typer.Option()] = DEFAULT_TEXT,
    vae_path: Annotated[Path, typer.Option()] = DEFAULT_VAE,
    pe_checkpoint_path: Annotated[Path, typer.Option()] = DEFAULT_PE,
    siglip_model_id: Annotated[str, typer.Option()] = DEFAULT_SIGLIP,
    device: Annotated[str, typer.Option()] = "cuda:0",
    steps: Annotated[int, typer.Option(min=1, max=MAX_PILOT_STEPS)] = 32,
    resolution: Annotated[int, typer.Option(min=64, max=512)] = 256,
    lr: Annotated[float, typer.Option(min=1e-7, max=1e-2)] = 5e-6,
    seed: Annotated[int, typer.Option()] = 20260618,
    max_rows: Annotated[int, typer.Option(min=2, max=MAX_PILOT_ROWS)] = 16,
    init_checkpoint_path: Annotated[Path | None, typer.Option()] = None,
    contrastive_weight: Annotated[float, typer.Option(min=0.0, max=10.0)] = 0.25,
    contrastive_margin: Annotated[float, typer.Option(min=0.0, max=10.0)] = 0.05,
    teacher_weight: Annotated[float, typer.Option(min=0.0, max=10.0)] = 0.5,
    token_weight: Annotated[float, typer.Option(min=0.0, max=10.0)] = 0.0,
    token_max_similarity: Annotated[float, typer.Option(min=-1.0, max=1.0)] = 0.2,
    pe_token_weight: Annotated[float, typer.Option(min=0.0, max=10.0)] = 0.0,
    pe_token_block_stride: Annotated[int, typer.Option(min=1, max=28)] = 4,
    pe_retrieval_weight: Annotated[float, typer.Option(min=0.0, max=10.0)] = 0.0,
    pe_retrieval_margin: Annotated[float, typer.Option(min=0.0, max=2.0)] = 0.2,
    pe_kv_init: Annotated[bool, typer.Option()] = False,
    pe_encoder_name: Annotated[str, typer.Option()] = "pe",
    calibrator_bottleneck_dim: Annotated[int | None, typer.Option(min=1)] = None,
    train_calibrator_only: Annotated[bool, typer.Option()] = False,
) -> None:
    summary = run_teacher_smoke(
        SmokeConfig(
            manifest_path=manifest_path,
            image_root=image_root,
            output_path=output_path,
            dit_path=dit_path,
            text_encoder_path=text_encoder_path,
            vae_path=vae_path,
            pe_checkpoint_path=pe_checkpoint_path,
            siglip_model_id=siglip_model_id,
            device=device,
            steps=steps,
            resolution=resolution,
            lr=lr,
            seed=seed,
            max_rows=max_rows,
            init_checkpoint_path=init_checkpoint_path,
        ),
        contrastive_weight=contrastive_weight,
        contrastive_margin=contrastive_margin,
        teacher_weight=teacher_weight,
        token_weight=token_weight,
        token_max_similarity=token_max_similarity,
        pe_token_weight=pe_token_weight,
        pe_token_block_stride=pe_token_block_stride,
        pe_retrieval_weight=pe_retrieval_weight,
        pe_retrieval_margin=pe_retrieval_margin,
        pe_kv_init=pe_kv_init,
        pe_encoder_name=pe_encoder_name,
        calibrator_bottleneck_dim=calibrator_bottleneck_dim,
        train_calibrator_only=train_calibrator_only,
    )
    console.print_json(json.dumps(asdict(summary), ensure_ascii=True))


if __name__ == "__main__":
    app()
