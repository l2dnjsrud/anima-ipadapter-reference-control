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
    DEFAULT_PE,
    DEFAULT_SIGLIP,
    DEFAULT_TEXT,
    DEFAULT_VAE,
)
from training.siglip_shape_contrastive_smoke import run_shape_contrastive_smoke  # noqa: E402
from training.siglip_smoke_types import MAX_PILOT_ROWS, MAX_PILOT_STEPS, SmokeConfig  # noqa: E402


DEFAULT_C094_OUTPUT = Path(
    "checkpoints/anima_siglip_ip_adapter_c094_shape_supervised_0064_20260613.safetensors"
)

app = typer.Typer(add_completion=False)
console = Console()


@app.command()
def main(
    manifest_path: Annotated[Path, typer.Option()] = Path(
        "training/manifests/c093_siglip_qwen_target_anti_collapse_20260613.jsonl"
    ),
    image_root: Annotated[Path, typer.Option()] = Path(".tmp/c093_anti_collapse_root"),
    output_path: Annotated[Path, typer.Option()] = DEFAULT_C094_OUTPUT,
    dit_path: Annotated[Path, typer.Option()] = DEFAULT_DIT,
    text_encoder_path: Annotated[Path, typer.Option()] = DEFAULT_TEXT,
    vae_path: Annotated[Path, typer.Option()] = DEFAULT_VAE,
    pe_checkpoint_path: Annotated[Path, typer.Option()] = DEFAULT_PE,
    siglip_model_id: Annotated[str, typer.Option()] = DEFAULT_SIGLIP,
    device: Annotated[str, typer.Option()] = "cuda:0",
    steps: Annotated[int, typer.Option(min=1, max=MAX_PILOT_STEPS)] = 64,
    resolution: Annotated[int, typer.Option(min=64, max=512)] = 256,
    lr: Annotated[float, typer.Option(min=1e-7, max=1e-2)] = 4e-6,
    seed: Annotated[int, typer.Option()] = 20260694,
    max_rows: Annotated[int, typer.Option(min=2, max=MAX_PILOT_ROWS)] = 10,
    init_checkpoint_path: Annotated[Path | None, typer.Option()] = Path(
        "checkpoints/anima_siglip_ip_adapter_c093_qwen_target_anti_collapse_0048_20260613.safetensors"
    ),
    contrastive_weight: Annotated[float, typer.Option(min=0.0, max=10.0)] = 0.25,
    contrastive_margin: Annotated[float, typer.Option(min=0.0, max=10.0)] = 0.08,
    shape_weight: Annotated[float, typer.Option(min=0.0, max=10.0)] = 0.20,
    reference_shape_weight: Annotated[float, typer.Option(min=0.0, max=10.0)] = 0.35,
    feature_bridge_bottleneck_dim: Annotated[int | None, typer.Option(min=1)] = None,
    train_feature_bridge_only: Annotated[bool, typer.Option()] = False,
) -> None:
    summary = run_shape_contrastive_smoke(
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
        shape_weight=shape_weight,
        reference_shape_weight=reference_shape_weight,
        feature_bridge_bottleneck_dim=feature_bridge_bottleneck_dim,
        train_feature_bridge_only=train_feature_bridge_only,
    )
    console.print_json(json.dumps(asdict(summary), ensure_ascii=True))


if __name__ == "__main__":
    app()
