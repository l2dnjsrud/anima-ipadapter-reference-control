from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING, Annotated

from PIL import Image, ImageOps
import typer

from tools.image_feature_embedders import DEFAULT_SIGLIP_MODEL_ID
from tools.siglip_auto_caption_types import JsonObject
from tools.token_pair_probe_metrics import (
    TokenPairScore,
    TokenProbeInputError,
    render_token_probe_report,
    score_token_pair,
    summarize_token_metrics,
)

if TYPE_CHECKING:
    import torch


class SigLIPTokenImageEmbedder:
    def __init__(self, *, model_id: str, device: str, layer: int, resolution: int) -> None:
        import torch
        from transformers import AutoImageProcessor, SiglipVisionModel

        resolved_device = _resolve_device(device)
        dtype = torch.bfloat16 if resolved_device == "cuda" else torch.float32
        self._device = torch.device(resolved_device)
        self._dtype = dtype
        self._layer = layer
        self._resolution = resolution
        self._processor = AutoImageProcessor.from_pretrained(model_id)
        self._model = SiglipVisionModel.from_pretrained(
            model_id,
            torch_dtype=dtype,
            trust_remote_code=True,
        ).to(self._device)
        self._model.eval()

    def encode_image(self, image_path: Path) -> torch.Tensor:
        import torch

        image = _load_square_rgb(image_path, resolution=self._resolution)
        inputs = self._processor(images=[image], return_tensors="pt", do_resize=False)
        prepared = {
            key: value.to(device=self._device, dtype=self._dtype)
            for key, value in inputs.items()
        }
        with torch.no_grad():
            outputs = self._model(**prepared, output_hidden_states=True)
        hidden_states = outputs.hidden_states
        if hidden_states is None:
            raise TokenProbeInputError("SigLIP did not return hidden states")
        tokens = hidden_states[self._layer][0].detach().float().cpu()
        return torch.nn.functional.normalize(tokens, dim=-1)


def score_token_pair_probe_manifest(
    manifest_path: Path,
    *,
    data_root: Path,
    embedder: SigLIPTokenImageEmbedder,
    encoder_name: str,
    layer: int,
    topk: int,
) -> JsonObject:
    embedding_cache: dict[Path, torch.Tensor] = {}
    rows: list[TokenPairScore] = []
    for raw in _read_manifest_rows(manifest_path):
        anchor_id = str(raw["anchor_id"])
        candidate_id = str(raw["candidate_id"])
        anchor_path = data_root / f"{anchor_id}.jpg"
        candidate_path = data_root / f"{candidate_id}.jpg"
        anchor_tokens = _embedding_cached(embedder, embedding_cache, anchor_path)
        candidate_tokens = _embedding_cached(embedder, embedding_cache, candidate_path)
        rows.append(
            TokenPairScore(
                pair_id=str(raw["pair_id"]),
                label=str(raw["label"]),
                anchor_id=anchor_id,
                candidate_id=candidate_id,
                scores=score_token_pair(anchor_tokens, candidate_tokens, topk=topk),
            )
        )
    return {
        "manifest_path": str(manifest_path),
        "data_root": str(data_root),
        "encoder": encoder_name,
        "layer": layer,
        "topk": topk,
        "rows": [asdict(row) for row in rows],
        "summaries": summarize_token_metrics(rows),
    }


def write_token_probe_report(result: JsonObject, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(render_token_probe_report(result), encoding="utf-8")


def _read_manifest_rows(manifest_path: Path) -> tuple[JsonObject, ...]:
    rows: list[JsonObject] = []
    with manifest_path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            raw = json.loads(line)
            if not isinstance(raw, dict):
                raise TokenProbeInputError(f"manifest row {line_number} must be an object")
            rows.append(raw)
    if not rows:
        raise TokenProbeInputError(f"manifest has no rows: {manifest_path}")
    return tuple(rows)


def _embedding_cached(
    embedder: SigLIPTokenImageEmbedder,
    cache: dict[Path, torch.Tensor],
    image_path: Path,
) -> torch.Tensor:
    if image_path not in cache:
        cache[image_path] = embedder.encode_image(image_path)
    return cache[image_path]


def _resolve_device(device: str) -> str:
    if device != "auto":
        return device
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def _load_square_rgb(path: Path, *, resolution: int) -> Image.Image:
    with Image.open(path) as image:
        return ImageOps.fit(
            image.convert("RGB"),
            (resolution, resolution),
            method=Image.Resampling.LANCZOS,
            centering=(0.5, 0.5),
        )


app = typer.Typer(add_completion=False)


@app.command()
def main(
    manifest_path: Annotated[Path, typer.Argument()],
    output_path: Annotated[Path, typer.Argument()],
    data_root: Annotated[Path, typer.Option()],
    model_id: Annotated[str, typer.Option()] = DEFAULT_SIGLIP_MODEL_ID,
    device: Annotated[str, typer.Option()] = "auto",
    layer: Annotated[int, typer.Option()] = -1,
    resolution: Annotated[int, typer.Option(min=64)] = 512,
    topk: Annotated[int, typer.Option(min=1)] = 64,
    report_path: Annotated[Path | None, typer.Option()] = None,
) -> None:
    embedder = SigLIPTokenImageEmbedder(
        model_id=model_id,
        device=device,
        layer=layer,
        resolution=resolution,
    )
    result = score_token_pair_probe_manifest(
        manifest_path,
        data_root=data_root,
        embedder=embedder,
        encoder_name=model_id,
        layer=layer,
        topk=topk,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    if report_path is not None:
        write_token_probe_report(result, report_path)
    typer.echo(f"wrote {output_path}")


if __name__ == "__main__":
    app()
