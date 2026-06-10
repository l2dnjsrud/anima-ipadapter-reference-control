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
# /home/wktwin/anima-lora-training-bundle/anima_lora/.venv/bin/python training/siglip_real_smoke.py \
#   --manifest-path training/manifests/local_color_pairs_pilot_20260610.jsonl \
#   --image-root /home/wktwin/anima-lora-training-bundle/image_dataset_color_panel_style_v5_best \
#   --steps 1 --resolution 256 --device cuda:0

from __future__ import annotations

import json
import math
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Annotated

import torch
import typer
from rich.console import Console
from safetensors.torch import save_file

ROOT = Path(__file__).resolve().parents[1]
ANIMA_ROOT = Path("/home/wktwin/anima-lora-training-bundle/anima_lora")
for candidate in (ROOT, ANIMA_ROOT):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from siglip_checkpoint import SigLIPCheckpointError, load_siglip_adapter  # noqa: E402
from siglip_model import IPAdapterSigLIP  # noqa: E402
from training.siglip_smoke_data import load_pair_rows, resolve_pair_paths  # noqa: E402
from training.siglip_smoke_patch import patched_cross_attention  # noqa: E402
from training.siglip_smoke_runtime import (  # noqa: E402
    encode_prompt,
    encode_siglip_features,
    encode_target_latents,
    noise_args,
    seed_everything,
    validate_config,
)
from training.siglip_smoke_types import (  # noqa: E402
    CheckpointVerification,
    MAX_PILOT_ROWS,
    MAX_PILOT_STEPS,
    SmokeConfig,
    SmokeInputError,
    SmokeSummary,
)


DEFAULT_DIT = ANIMA_ROOT / "models/diffusion_models/anima-base-v1.0.safetensors"
DEFAULT_TEXT = ANIMA_ROOT / "models/text_encoders/qwen_3_06b_base.safetensors"
DEFAULT_VAE = ANIMA_ROOT / "models/vae/qwen_image_vae.safetensors"
DEFAULT_OUTPUT = Path("/data/ai/models/ipadapter/anima_siglip_ip_adapter_smoke_20260610.safetensors")
DEFAULT_PE = Path("/data/ai/models/ipadapter/anima_ip_adapter_quality_20260610.safetensors")
DEFAULT_SIGLIP = "google/siglip2-base-patch16-512"

app = typer.Typer(add_completion=False)
console = Console()


def freeze_module(module: torch.nn.Module) -> int:
    module.eval()
    total = 0
    for parameter in module.parameters():
        total += parameter.numel()
        parameter.requires_grad_(False)
    return total


def trainable_parameter_count(module: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters() if parameter.requires_grad)


def save_adapter_checkpoint(adapter: IPAdapterSigLIP, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    state = {
        key: value.detach().cpu().contiguous()
        for key, value in adapter.state_dict().items()
    }
    save_file(
        state,
        str(output_path),
        metadata={"format": "pt", "ss_encoder": "siglip2", "ss_adapter": "IPAdapterSigLIP"},
    )


def verify_checkpoint(output_path: Path, pe_checkpoint_path: Path) -> CheckpointVerification:
    load_siglip_adapter(output_path)
    pe_rejected = False
    try:
        load_siglip_adapter(pe_checkpoint_path)
    except SigLIPCheckpointError:
        pe_rejected = True
    return CheckpointVerification(
        output_path=str(output_path),
        loadable=True,
        pe_checkpoint_rejected=pe_rejected,
    )


def run_real_smoke(config: SmokeConfig) -> SmokeSummary:
    validate_config(config)
    seed_everything(config.seed)
    device = torch.device(config.device)
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    rows = load_pair_rows(config.manifest_path, limit=config.max_rows)

    from library.anima.weights import load_anima_model, load_qwen3_text_encoder
    from library.inference.text import prepare_text_inputs
    from library.models.qwen_vae import load_vae
    from library.runtime.noise import (
        FlowMatchEulerDiscreteScheduler,
        get_noisy_model_input_and_timesteps,
    )
    from transformers import AutoImageProcessor, SiglipVisionModel

    anima = load_anima_model(device, str(config.dit_path), "torch", device, dtype)
    anima.to(device=device, dtype=dtype)
    frozen_params = freeze_module(anima)
    vae = load_vae(str(config.vae_path), device=device, dtype=dtype, eval=True)
    frozen_params += freeze_module(vae)
    text_encoder, _ = load_qwen3_text_encoder(str(config.text_encoder_path), dtype=dtype, device=str(device))
    frozen_params += freeze_module(text_encoder)
    siglip = SiglipVisionModel.from_pretrained(
        config.siglip_model_id,
        torch_dtype=dtype,
        trust_remote_code=True,
    ).to(device)
    processor = AutoImageProcessor.from_pretrained(config.siglip_model_id)
    frozen_params += freeze_module(siglip)

    adapter = IPAdapterSigLIP().to(device=device, dtype=dtype)
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=config.lr)
    scheduler = FlowMatchEulerDiscreteScheduler(num_train_timesteps=1000, shift=1.0)
    losses: list[float] = []

    for step in range(config.steps):
        row = rows[step % len(rows)]
        paths = resolve_pair_paths(row, config.image_root)
        latents = encode_target_latents(vae, paths.target_image, config.resolution, device, dtype)
        noise = torch.randn_like(latents)
        noisy, timesteps, _sigmas = get_noisy_model_input_and_timesteps(
            noise_args(),
            scheduler,
            latents,
            noise,
            device,
            dtype,
        )
        crossattn_emb = encode_prompt(
            row.prompt,
            config.text_encoder_path,
            text_encoder,
            anima,
            prepare_text_inputs,
            device,
            dtype,
        )
        features = encode_siglip_features(
            siglip,
            processor,
            paths.ref_image,
            device,
            dtype,
        )
        image_tokens = adapter.encode_ref(features, timestep=timesteps)
        padding_mask = torch.zeros(
            latents.shape[0], 1, latents.shape[-2], latents.shape[-1], device=device, dtype=dtype
        )

        optimizer.zero_grad(set_to_none=True)
        with patched_cross_attention(anima, adapter, image_tokens):
            model_pred = anima(noisy.unsqueeze(2), timesteps, crossattn_emb, padding_mask=padding_mask)
        target = noise - latents
        loss = torch.nn.functional.mse_loss(model_pred.squeeze(2).float(), target.float())
        if not torch.isfinite(loss):
            raise SmokeInputError(f"non-finite loss at step {step}: {float(loss.detach().cpu())}")
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu()))

    save_adapter_checkpoint(adapter, config.output_path)
    checkpoint = verify_checkpoint(config.output_path, config.pe_checkpoint_path)
    return SmokeSummary(
        steps=config.steps,
        rows_loaded=len(rows),
        first_loss=losses[0],
        final_loss=losses[-1],
        mean_loss=sum(losses) / len(losses),
        finite_loss=all(math.isfinite(loss) for loss in losses),
        loss_history=tuple(losses),
        trainable_parameters=trainable_parameter_count(adapter),
        frozen_base_parameters=frozen_params,
        checkpoint=checkpoint,
    )


@app.command()
def main(
    manifest_path: Annotated[Path, typer.Option()] = Path("training/manifests/local_color_pairs_pilot_20260610.jsonl"),
    image_root: Annotated[Path, typer.Option()] = Path("/home/wktwin/anima-lora-training-bundle/image_dataset_color_panel_style_v5_best"),
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
    )
    summary = run_real_smoke(config)
    console.print_json(json.dumps(asdict(summary), ensure_ascii=True))


if __name__ == "__main__":
    app()
