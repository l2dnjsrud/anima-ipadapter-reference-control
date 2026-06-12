from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Annotated

import typer

from tools.comfy_api_client import first_image_info, post_json, wait_history
from tools.siglip_auto_caption_artifacts import (
    copy_output_image,
    copy_reference,
    write_contact_sheet,
    write_report,
    write_summary,
)
from tools.siglip_auto_caption_types import (
    DEFAULT_BASE_URL,
    DEFAULT_COMFY_INPUT,
    DEFAULT_COMFY_OUTPUT,
    DEFAULT_DATA_ROOT,
    DEFAULT_OUT_DIR,
    NEGATIVE,
    SIGLIP_PE_RETRIEVAL_CHECKPOINT,
    SIGLIP_PE_SPACE_CHECKPOINT,
    AutoPromptRow,
    EvalConfig,
    JsonObject,
    JsonValue,
    Sample,
    Variant,
)


def load_auto_prompt_rows(path: Path) -> tuple[Sample, ...]:
    rows: list[Sample] = []
    with path.open(encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            raw = json.loads(line)
            prompt_row = AutoPromptRow(
                ref_id=str(raw["ref_id"]),
                tgt_id=str(raw["tgt_id"]),
                source_prompt=str(raw["source_prompt"]),
                prompt=str(raw["prompt"]),
                selected_attributes=tuple(str(item) for item in raw["selected_attributes"]),
            )
            rows.append(
                Sample(
                    label=f"auto{index:02d}",
                    ref_id=prompt_row.ref_id,
                    seed=20260650 + index,
                    prompt_row=prompt_row,
                )
            )
    return tuple(rows)


def no_ip_prompt(sample: Sample, variant: Variant, *, output_prefix: str) -> JsonObject:
    return base_prompt(sample, variant, output_prefix=output_prefix)


def adapter_prompt(
    sample: Sample,
    image_name: str,
    variant: Variant,
    *,
    output_prefix: str,
) -> JsonObject:
    prompt = base_prompt(sample, variant, output_prefix=output_prefix)
    prompt["1"] = {"class_type": "LoadImage", "inputs": {"image": image_name}}
    prompt["2"] = {
        "class_type": "AnimaSigLIPIPAdapterLoader",
        "inputs": {"ipadapter_name": variant.checkpoint},
    }
    prompt["3"] = {
        "class_type": "AnimaSigLIPEncodeImage",
        "inputs": {
            "image": ["1", 0],
            "siglip_model_id": "google/siglip2-base-patch16-512",
            "include_shallow": True,
        },
    }
    prompt["4"] = {
        "class_type": "AnimaSigLIPIPAdapterApply",
        "inputs": {
            "model": ["18", 0],
            "ipadapter": ["2", 0],
            "siglip_features": ["3", 0],
            "weight": variant.weight,
            "start_at": 0.0,
            "end_at": 0.85,
        },
    }
    prompt["10"] = cfg_node("4")
    prompt["12"] = scheduler_node("4")
    return prompt


def base_prompt(sample: Sample, variant: Variant, *, output_prefix: str) -> JsonObject:
    prefix = f"{output_prefix}/{sample.label}_{variant.label}"
    return {
        "5": {"class_type": "UNETLoader", "inputs": {"unet_name": "anima-base-v1.0.safetensors", "weight_dtype": "default"}},
        "6": {"class_type": "CLIPLoader", "inputs": {"clip_name": "qwen_3_06b_base.safetensors", "type": "qwen_image", "device": "default"}},
        "7": text_node(sample.prompt_row.prompt),
        "8": text_node(NEGATIVE),
        "9": {"class_type": "EmptySD3LatentImage", "inputs": {"width": 768, "height": 1024, "batch_size": 1}},
        "10": cfg_node("18"),
        "11": {"class_type": "RandomNoise", "inputs": {"noise_seed": sample.seed}},
        "12": scheduler_node("18"),
        "13": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "er_sde"}},
        "14": {"class_type": "SamplerCustomAdvanced", "inputs": {"noise": ["11", 0], "guider": ["10", 0], "sampler": ["13", 0], "sigmas": ["12", 0], "latent_image": ["9", 0]}},
        "15": {"class_type": "VAELoader", "inputs": {"vae_name": "qwen/qwen_image_vae.safetensors"}},
        "16": {"class_type": "VAEDecode", "inputs": {"samples": ["14", 0], "vae": ["15", 0]}},
        "17": {"class_type": "SaveImage", "inputs": {"images": ["16", 0], "filename_prefix": prefix}},
        "18": {"class_type": "ModelSamplingFlux", "inputs": {"model": ["5", 0], "max_shift": 3.0, "base_shift": 3.0, "width": 768, "height": 1024}},
    }


def text_node(text: str) -> JsonObject:
    return {
        "class_type": "CLIPTextEncodeFlux",
        "inputs": {"clip": ["6", 0], "clip_l": text, "t5xxl": text, "guidance": 3.5},
    }


def cfg_node(model_node: str) -> JsonObject:
    return {
        "class_type": "CFGGuider",
        "inputs": {"model": [model_node, 0], "positive": ["7", 0], "negative": ["8", 0], "cfg": 3.2},
    }


def scheduler_node(model_node: str) -> JsonObject:
    return {
        "class_type": "BasicScheduler",
        "inputs": {"model": [model_node, 0], "scheduler": "simple", "steps": 18, "denoise": 1.0},
    }


def run_eval(manifest_path: Path, config: EvalConfig) -> None:
    config.out_dir.mkdir(parents=True, exist_ok=True)
    samples = load_auto_prompt_rows(manifest_path)
    variants = (
        Variant("no_ip", None, 0.0),
        Variant("siglip_pe_space_w14", SIGLIP_PE_SPACE_CHECKPOINT, 1.4),
        Variant("siglip_pe_retrieval_w14", SIGLIP_PE_RETRIEVAL_CHECKPOINT, 1.4),
    )
    results: dict[str, JsonValue] = {}
    for sample in samples:
        image_name = copy_reference(sample, config)
        for variant in variants:
            results[f"{sample.label}_{variant.label}"] = run_prompt(
                sample,
                variant,
                image_name,
                config,
            )
    write_contact_sheet(samples, variants, config)
    write_summary(samples, variants, results, config)
    write_report(samples, variants, config)


def run_prompt(
    sample: Sample,
    variant: Variant,
    image_name: str,
    config: EvalConfig,
) -> JsonObject:
    output_prefix = f"anima_ipadapter/{config.out_dir.name}"
    prompt = (
        no_ip_prompt(sample, variant, output_prefix=output_prefix)
        if variant.checkpoint is None
        else adapter_prompt(sample, image_name, variant, output_prefix=output_prefix)
    )
    payload: JsonObject = {"prompt": prompt, "client_id": str(uuid.uuid4())}
    name = f"{sample.label}_{variant.label}"
    (config.out_dir / f"{name}.api_prompt.json").write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )
    response = post_json(config.base_url, "/prompt", payload)
    prompt_id = str(response["prompt_id"])
    history = wait_history(config.base_url, prompt_id)
    (config.out_dir / f"{name}.response.json").write_text(json.dumps(response, indent=2), encoding="utf-8")
    (config.out_dir / f"{name}.history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    image_info = first_image_info(history)
    image_path = copy_output_image(image_info, name, config)
    return {"prompt_id": prompt_id, "image": str(image_path), "image_info": image_info}


app = typer.Typer(add_completion=False)


@app.command()
def main(
    manifest_path: Annotated[Path, typer.Argument()],
    out_dir: Annotated[Path, typer.Option()] = DEFAULT_OUT_DIR,
    base_url: Annotated[str, typer.Option()] = DEFAULT_BASE_URL,
    data_root: Annotated[Path, typer.Option()] = DEFAULT_DATA_ROOT,
    comfy_input: Annotated[Path, typer.Option()] = DEFAULT_COMFY_INPUT,
    comfy_output: Annotated[Path, typer.Option()] = DEFAULT_COMFY_OUTPUT,
) -> None:
    run_eval(
        manifest_path,
        EvalConfig(
            data_root=data_root,
            base_url=base_url,
            out_dir=out_dir,
            comfy_input=comfy_input,
            comfy_output=comfy_output,
        ),
    )
    typer.echo(f"wrote {out_dir / 'contact_sheet.jpg'}")


if __name__ == "__main__":
    app()
