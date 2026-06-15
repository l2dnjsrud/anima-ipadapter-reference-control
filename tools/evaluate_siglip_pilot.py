# /// script
# dependencies = [
#   "safetensors",
#   "torch",
#   "typer",
# ]
# ///
# --- How to run -----------------------------------------------------
# /home/wktwin/anima-lora-training-bundle/anima_lora/.venv/bin/python tools/evaluate_siglip_pilot.py \
#   --smoke-checkpoint checkpoints/anima_siglip_ip_adapter_smoke_20260610.safetensors \
#   --pilot-checkpoint checkpoints/anima_siglip_ip_adapter_pilot_20260610.safetensors \
#   --pe-summary-path eval/comfy_pe_full_contactsheet_20260610/summary.json \
#   --out-dir eval/siglip_color_pilot_20260610

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

try:
    from tools.siglip_pilot_eval_core import evaluate_pilot
except ModuleNotFoundError:
    from siglip_pilot_eval_core import evaluate_pilot


app = typer.Typer(add_completion=False)


@app.command()
def main(
    smoke_checkpoint: Annotated[Path, typer.Option()] = Path(
        "checkpoints/anima_siglip_ip_adapter_smoke_20260610.safetensors"
    ),
    pilot_checkpoint: Annotated[Path, typer.Option()] = Path(
        "checkpoints/anima_siglip_ip_adapter_pilot_20260610.safetensors"
    ),
    pe_summary_path: Annotated[Path, typer.Option()] = Path(
        "eval/comfy_pe_full_contactsheet_20260610/summary.json"
    ),
    out_dir: Annotated[Path, typer.Option()] = Path("eval/siglip_color_pilot_20260610"),
) -> None:
    result = evaluate_pilot(
        smoke_checkpoint=smoke_checkpoint,
        pilot_checkpoint=pilot_checkpoint,
        pe_summary_path=pe_summary_path,
        out_dir=out_dir,
        siglip_visual_eval_available=False,
    )
    typer.echo(f"metrics={result.metrics_path}")
    typer.echo(f"report={result.report_path}")
    typer.echo(f"decision={result.decision.label}")


if __name__ == "__main__":
    app()
