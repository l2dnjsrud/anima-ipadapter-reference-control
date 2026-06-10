# /// script
# dependencies = [
#   "torch",
#   "typer",
#   "rich",
# ]
# ///
# --- How to run -----------------------------------------------------
# uv run training/siglip_proof.py --pairs-path sample_pairs.jsonl --image-dir ./images
# /home/wktwin/anima-lora-training-bundle/anima_lora/.venv/bin/python training/siglip_proof.py

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated

import torch
import typer
from rich.console import Console

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# The path bootstrap keeps this script runnable from both repo root and training/.
from siglip_model import IPAdapterSigLIP, SigLIPFeatures  # noqa: E402


@dataclass(frozen=True, slots=True)
class PairRow:
    ref_id: str
    tgt_id: str
    prompt: str


app = typer.Typer(add_completion=False)
console = Console()


@app.command()
def main(
    pairs_path: Annotated[Path | None, typer.Option(help="Optional Wenaka-style JSONL pairs.")]=None,
    image_dir: Annotated[Path | None, typer.Option(help="Optional directory containing <id>.jpg files.")]=None,
    batch_size: Annotated[int, typer.Option(min=1, max=8)] = 2,
    rows_to_check: Annotated[int, typer.Option(min=0, max=64)] = 4,
    device: Annotated[str, typer.Option()] = "cpu",
) -> None:
    """Run a tiny trainability proof without downloading the full dataset."""

    rows = _load_rows(pairs_path, rows_to_check) if pairs_path is not None else []
    missing_images = _missing_images(rows, image_dir) if image_dir is not None else []
    loss = _synthetic_step(batch_size, torch.device(device))
    console.print_json(
        data={
            "proof": "siglip_time_resampler_ipcrossattn",
            "synthetic_loss": loss,
            "checked_rows": len(rows),
            "missing_images": [str(path) for path in missing_images],
            "full_training_started": False,
        }
    )


def _load_rows(path: Path, limit: int) -> list[PairRow]:
    rows: list[PairRow] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if len(rows) >= limit:
                break
            raw = json.loads(line)
            rows.append(PairRow(ref_id=str(raw["ref_id"]), tgt_id=str(raw["tgt_id"]), prompt=str(raw["prompt"])))
    return rows


def _missing_images(rows: list[PairRow], image_dir: Path) -> list[Path]:
    missing: list[Path] = []
    for row in rows:
        for image_id in (row.ref_id, row.tgt_id):
            path = image_dir / f"{image_id}.jpg"
            if not path.exists():
                missing.append(path)
    return missing


def _synthetic_step(batch_size: int, device: torch.device) -> float:
    torch.manual_seed(20260610)
    adapter = IPAdapterSigLIP(
        siglip_dim=8,
        siglip_shallow_dim=8,
        dit_dim=16,
        num_blocks=2,
        num_queries=3,
        resampler_depth=1,
        resampler_heads=2,
        resampler_dim=16,
        resampler_dim_head=8,
        intermediate_dim=8,
        intermediate_layers=1,
        intermediate_heads=2,
        ip_heads=4,
        time_embed_dim=10,
        use_intermediate_encoder=True,
    ).to(device)
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=1e-4)
    features = SigLIPFeatures(
        deep=torch.randn(batch_size, 5, 8, device=device),
        shallow=torch.randn(batch_size, 7, 8, device=device),
    )
    timestep = torch.rand(batch_size, device=device)
    image_tokens = adapter.encode_ref(features, timestep=timestep)
    query = torch.randn(batch_size, 4, 16, device=device)
    target = torch.zeros_like(query)
    loss = torch.nn.functional.mse_loss(adapter.forward_block(1, query, image_tokens), target)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return float(loss.detach().cpu())


if __name__ == "__main__":
    app()
