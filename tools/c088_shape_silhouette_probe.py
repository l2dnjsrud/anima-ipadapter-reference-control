# /// script
# dependencies = ["typer", "pillow", "numpy"]
# ///
# --- How to run -----------------------------------------------------
# PYTHONPATH=. /home/wktwin/anima-lora-training-bundle/anima_lora/.venv/bin/python \
#   tools/c088_shape_silhouette_probe.py build
# PYTHONPATH=. /home/wktwin/anima-lora-training-bundle/anima_lora/.venv/bin/python \
#   tools/c088_shape_silhouette_probe.py score-shape

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer

from tools.c088_probe_io import DEFAULT_CROP_SUMMARY, DEFAULT_FULL_SUMMARY, DEFAULT_OUT_DIR
from tools.c088_probe_manifest import C088BuildConfig, build_c088_probe_manifest
from tools.c088_shape_metrics import score_shape_silhouette_manifest


app = typer.Typer(add_completion=False)


@app.command()
def build(
    crop_summary_path: Annotated[Path, typer.Option()] = DEFAULT_CROP_SUMMARY,
    full_summary_path: Annotated[Path, typer.Option()] = DEFAULT_FULL_SUMMARY,
    output_dir: Annotated[Path, typer.Option()] = DEFAULT_OUT_DIR,
    crop_limit: Annotated[int, typer.Option()] = 10,
) -> None:
    summary = build_c088_probe_manifest(
        C088BuildConfig(
            crop_summary_path=crop_summary_path,
            full_summary_path=full_summary_path,
            output_dir=output_dir,
            crop_limit=crop_limit,
        )
    )
    typer.echo(json.dumps(summary, ensure_ascii=False, indent=2))


@app.command()
def score_shape(
    manifest_path: Annotated[Path, typer.Option()] = DEFAULT_OUT_DIR / "probe_manifest.jsonl",
    output_dir: Annotated[Path, typer.Option()] = DEFAULT_OUT_DIR,
) -> None:
    result = score_shape_silhouette_manifest(manifest_path, output_dir)
    typer.echo(json.dumps(result["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    app()
