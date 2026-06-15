from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console

from training.siglip_real_smoke import (
    DEFAULT_DIT,
    DEFAULT_OUTPUT,
    DEFAULT_PE,
    DEFAULT_SIGLIP,
    DEFAULT_TEXT,
    DEFAULT_VAE,
    run_real_smoke,
)
from training.siglip_smoke_types import MAX_PILOT_ROWS, MAX_PILOT_STEPS, SmokeConfig


app = typer.Typer(add_completion=False)
console = Console()


@app.command()
def main(
    manifest_path: Annotated[Path, typer.Option()] = Path(
        "training/manifests/local_color_pairs_pilot_20260610.jsonl"
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
    steps: Annotated[int, typer.Option(min=1, max=MAX_PILOT_STEPS)] = 1,
    resolution: Annotated[int, typer.Option(min=64, max=512)] = 256,
    lr: Annotated[float, typer.Option(min=1e-7, max=1e-2)] = 1e-5,
    seed: Annotated[int, typer.Option()] = 20260610,
    max_rows: Annotated[int, typer.Option(min=1, max=MAX_PILOT_ROWS)] = 4,
    init_checkpoint_path: Annotated[Path | None, typer.Option()] = None,
) -> None:
    config = SmokeConfig(
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
    )
    summary = run_real_smoke(config)
    console.print_json(json.dumps(asdict(summary), ensure_ascii=True))


if __name__ == "__main__":
    app()
