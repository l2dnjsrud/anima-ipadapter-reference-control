# /// script
# dependencies = [
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
import math
import random
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Annotated

import torch
import typer
from rich.console import Console

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from siglip_encoder_lora import (  # noqa: E402
    apply_saved_siglip_lora,
    apply_siglip_lora,
    default_lora_module_names,
    lora_parameter_names,
    save_siglip_lora,
    trainable_lora_parameters,
    verify_siglip_lora,
)
from training.siglip_real_smoke import DEFAULT_SIGLIP, freeze_module  # noqa: E402
from training.siglip_smoke_data import load_pair_rows, load_siglip_image, resolve_pair_paths  # noqa: E402
from training.siglip_smoke_runtime import seed_everything  # noqa: E402
from training.siglip_smoke_types import MAX_PILOT_ROWS, MAX_PILOT_STEPS, SmokeInputError  # noqa: E402


DEFAULT_MANIFEST = Path("training/manifests/c093_siglip_qwen_target_anti_collapse_20260613.jsonl")
DEFAULT_IMAGE_ROOT = Path(".tmp/c093_anti_collapse_root")
DEFAULT_OUTPUT = Path("checkpoints/anima_siglip_encoder_lora_c096_rank8_0096_20260613.safetensors")


@dataclass(frozen=True, slots=True)
class EncoderLoRASummary:
    steps: int
    rows_loaded: int
    explicit_negative_rows: int
    first_loss: float
    final_loss: float
    mean_loss: float
    mean_positive_similarity: float
    mean_negative_similarity: float
    finite_loss: bool
    trainable_parameter_names: tuple[str, ...]
    checkpoint_path: str
    checkpoint_loadable: bool
    heldout_rows: tuple[str, ...]
    module_names: tuple[str, ...]
    rank: int
    alpha: float


def run_encoder_lora_contrastive(
    *,
    manifest_path: Path,
    image_root: Path,
    output_path: Path,
    siglip_model_id: str,
    device: str,
    steps: int,
    max_rows: int,
    lr: float,
    seed: int,
    rank: int,
    alpha: float,
    margin: float,
    layer_count: int,
) -> EncoderLoRASummary:
    _validate(manifest_path, image_root, steps, max_rows)
    seed_everything(seed)
    rows = load_pair_rows(manifest_path, limit=max_rows)
    explicit_negative_rows = sum(1 for row in rows if row.neg_id is not None)
    heldout_rows = tuple(row.ref_id for row in rows if "heldout" in row.ref_id or "heldout" in row.tgt_id)
    if heldout_rows:
        raise SmokeInputError("encoder LoRA training manifest contains heldout rows")
    random.Random(seed).shuffle(rows)
    torch_device = torch.device(device)
    dtype = torch.float32

    from transformers import AutoImageProcessor, SiglipVisionModel

    model = SiglipVisionModel.from_pretrained(
        siglip_model_id, torch_dtype=dtype, trust_remote_code=True
    ).to(torch_device)
    processor = AutoImageProcessor.from_pretrained(siglip_model_id)
    freeze_module(model)
    spec = apply_siglip_lora(
        model,
        module_names=default_lora_module_names(model, layer_count=layer_count),
        rank=rank,
        alpha=alpha,
    )
    model.train()
    optimizer = torch.optim.AdamW(trainable_lora_parameters(model), lr=lr)
    losses: list[float] = []
    positives: list[float] = []
    negatives: list[float] = []

    for step in range(steps):
        row = rows[step % len(rows)]
        paths = resolve_pair_paths(row, image_root)
        negative_path = image_root / f"{row.neg_id}.jpg" if row.neg_id else paths.ref_image
        anchor = _pooled_features(model, processor, paths.ref_image, torch_device, dtype)
        positive = _pooled_features(model, processor, paths.target_image, torch_device, dtype)
        negative = _pooled_features(model, processor, negative_path, torch_device, dtype)
        positive_sim = torch.nn.functional.cosine_similarity(anchor, positive, dim=-1)
        negative_sim = torch.nn.functional.cosine_similarity(anchor, negative, dim=-1)
        loss = torch.relu(margin + negative_sim - positive_sim).mean()
        loss = loss + 0.05 * (1.0 - positive_sim).mean()
        if not torch.isfinite(loss):
            raise SmokeInputError(f"non-finite loss at step {step}")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
        positives.append(float(positive_sim.detach().cpu().mean()))
        negatives.append(float(negative_sim.detach().cpu().mean()))

    save_siglip_lora(
        model,
        output_path,
        spec=spec,
        metadata={"source_manifest": str(manifest_path), "heldout_rows_used": "0"},
    )
    _verify_loadable(siglip_model_id, output_path, torch_device, dtype)
    loaded_spec = verify_siglip_lora(output_path)
    return EncoderLoRASummary(
        steps=steps,
        rows_loaded=len(rows),
        explicit_negative_rows=explicit_negative_rows,
        first_loss=losses[0],
        final_loss=losses[-1],
        mean_loss=sum(losses) / len(losses),
        mean_positive_similarity=sum(positives) / len(positives),
        mean_negative_similarity=sum(negatives) / len(negatives),
        finite_loss=all(math.isfinite(loss) for loss in losses),
        trainable_parameter_names=lora_parameter_names(model),
        checkpoint_path=str(output_path),
        checkpoint_loadable=True,
        heldout_rows=heldout_rows,
        module_names=loaded_spec.module_names,
        rank=loaded_spec.rank,
        alpha=loaded_spec.alpha,
    )


def _pooled_features(model, processor, image_path: Path, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    image = load_siglip_image(image_path)
    inputs = processor(images=[image], return_tensors="pt", do_resize=False)
    inputs = {key: value.to(device=device, dtype=dtype) for key, value in inputs.items()}
    outputs = model(**inputs, output_hidden_states=False)
    return torch.nn.functional.normalize(outputs.last_hidden_state.mean(dim=1), dim=-1)


def _verify_loadable(model_id: str, path: Path, device: torch.device, dtype: torch.dtype) -> None:
    from transformers import SiglipVisionModel

    del device
    model = SiglipVisionModel.from_pretrained(model_id, torch_dtype=dtype, trust_remote_code=True).to("cpu")
    apply_saved_siglip_lora(model, path)


def _validate(manifest_path: Path, image_root: Path, steps: int, max_rows: int) -> None:
    if steps < 1 or steps > MAX_PILOT_STEPS:
        raise SmokeInputError(f"steps must be between 1 and {MAX_PILOT_STEPS}")
    if max_rows < 2 or max_rows > MAX_PILOT_ROWS:
        raise SmokeInputError(f"max_rows must be between 2 and {MAX_PILOT_ROWS}")
    if not manifest_path.is_file():
        raise SmokeInputError(f"manifest not found: {manifest_path}")
    if not image_root.is_dir():
        raise SmokeInputError(f"image_root not found: {image_root}")


app = typer.Typer(add_completion=False)
console = Console()


@app.command()
def main(
    manifest_path: Annotated[Path, typer.Option()] = DEFAULT_MANIFEST,
    image_root: Annotated[Path, typer.Option()] = DEFAULT_IMAGE_ROOT,
    output_path: Annotated[Path, typer.Option()] = DEFAULT_OUTPUT,
    siglip_model_id: Annotated[str, typer.Option()] = DEFAULT_SIGLIP,
    device: Annotated[str, typer.Option()] = "cuda:0",
    steps: Annotated[int, typer.Option(min=1, max=MAX_PILOT_STEPS)] = 96,
    max_rows: Annotated[int, typer.Option(min=2, max=MAX_PILOT_ROWS)] = 10,
    lr: Annotated[float, typer.Option(min=1e-7, max=1e-2)] = 1e-4,
    seed: Annotated[int, typer.Option()] = 20260696,
    rank: Annotated[int, typer.Option(min=1, max=64)] = 8,
    alpha: Annotated[float, typer.Option(min=1e-3, max=128.0)] = 8.0,
    margin: Annotated[float, typer.Option(min=0.0, max=2.0)] = 0.08,
    layer_count: Annotated[int, typer.Option(min=1, max=12)] = 2,
) -> None:
    summary = run_encoder_lora_contrastive(
        manifest_path=manifest_path,
        image_root=image_root,
        output_path=output_path,
        siglip_model_id=siglip_model_id,
        device=device,
        steps=steps,
        max_rows=max_rows,
        lr=lr,
        seed=seed,
        rank=rank,
        alpha=alpha,
        margin=margin,
        layer_count=layer_count,
    )
    console.print_json(json.dumps(asdict(summary), ensure_ascii=True))


if __name__ == "__main__":
    app()
