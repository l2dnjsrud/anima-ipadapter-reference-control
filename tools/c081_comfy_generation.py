from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from PIL import Image

from tools.c081_identity_pair_acquisition import C081Config, build_c081_prompt_package
from tools.comfy_api_client import first_image_info, post_json, view_image_bytes, wait_history
from tools.siglip_auto_caption_types import JsonObject, JsonValue

DEFAULT_BASE_URL: Final = "http://127.0.0.1:8102"


@dataclass(frozen=True, slots=True)
class C081GenerationConfig:
    acquisition: C081Config = C081Config()
    base_url: str = DEFAULT_BASE_URL
    width: int = 768
    height: int = 1024
    steps: int = 18


def run_c081_generation(config: C081GenerationConfig) -> JsonObject:
    build_c081_prompt_package(config.acquisition)
    rows = _read_jsonl(config.acquisition.out_dir / "prompt_manifest.jsonl")
    generated_dir = config.acquisition.scratch_dir / "generated"
    generated_dir.mkdir(parents=True, exist_ok=True)
    generated = tuple(_run_prompt(row, config, generated_dir) for row in rows)
    manifest_path = config.acquisition.out_dir / "generation_manifest.jsonl"
    manifest_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in generated),
        encoding="utf-8",
    )
    summary = {
        "source": "c081_identity_preserving_pair_generation",
        "base_url": config.base_url,
        "prompt_count": len(rows),
        "generated_count": sum(1 for row in generated if row["status"] == "generated"),
        "failed_count": sum(1 for row in generated if row["status"] != "generated"),
        "generation_manifest": str(manifest_path),
        "heldout_rows_used": 0,
        "training_started": False,
    }
    (config.acquisition.out_dir / "generation_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary


def _run_prompt(row: JsonObject, config: C081GenerationConfig, generated_dir: Path) -> JsonObject:
    candidate_id = str(row["candidate_id"])
    response = post_json(config.base_url, "/prompt", {"prompt": _workflow(row, config)})
    prompt_id = str(response["prompt_id"])
    history = wait_history(config.base_url, prompt_id)
    image_info = first_image_info(history)
    image_bytes = view_image_bytes(
        config.base_url,
        filename=str(image_info["filename"]),
        subfolder=str(image_info.get("subfolder", "")),
        image_type=str(image_info.get("type", "output")),
    )
    image_path = generated_dir / f"{candidate_id}.png"
    image_path.write_bytes(image_bytes)
    width, height, blank = _image_info(image_path)
    return dict(row) | {
        "prompt_id": prompt_id,
        "status": "generated",
        "local_image_path": str(image_path),
        "image_width": width,
        "image_height": height,
        "blank": blank,
        "comfy_filename": str(image_info["filename"]),
        "comfy_subfolder": str(image_info.get("subfolder", "")),
        "comfy_type": str(image_info.get("type", "output")),
    }


def _workflow(row: JsonObject, config: C081GenerationConfig) -> JsonObject:
    prompt = str(row["prompt"])
    negative = str(row["negative"])
    prefix = f"anima_ipadapter/{config.acquisition.out_dir.name}/{row['candidate_id']}"
    return {
        "5": {"class_type": "UNETLoader", "inputs": {"unet_name": "anima-base-v1.0.safetensors", "weight_dtype": "default"}},
        "6": {"class_type": "CLIPLoader", "inputs": {"clip_name": "qwen_3_06b_base.safetensors", "type": "qwen_image", "device": "default"}},
        "7": _text_node(prompt),
        "8": _text_node(negative),
        "9": {"class_type": "EmptySD3LatentImage", "inputs": {"width": config.width, "height": config.height, "batch_size": 1}},
        "10": {"class_type": "CFGGuider", "inputs": {"model": ["18", 0], "positive": ["7", 0], "negative": ["8", 0], "cfg": 3.2}},
        "11": {"class_type": "RandomNoise", "inputs": {"noise_seed": int(row["seed"])}},
        "12": {"class_type": "BasicScheduler", "inputs": {"model": ["18", 0], "scheduler": "simple", "steps": config.steps, "denoise": 1.0}},
        "13": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "er_sde"}},
        "14": {"class_type": "SamplerCustomAdvanced", "inputs": {"noise": ["11", 0], "guider": ["10", 0], "sampler": ["13", 0], "sigmas": ["12", 0], "latent_image": ["9", 0]}},
        "15": {"class_type": "VAELoader", "inputs": {"vae_name": "qwen/qwen_image_vae.safetensors"}},
        "16": {"class_type": "VAEDecode", "inputs": {"samples": ["14", 0], "vae": ["15", 0]}},
        "17": {"class_type": "SaveImage", "inputs": {"images": ["16", 0], "filename_prefix": prefix}},
        "18": {"class_type": "ModelSamplingFlux", "inputs": {"model": ["5", 0], "max_shift": 3.0, "base_shift": 3.0, "width": config.width, "height": config.height}},
    }


def _text_node(text: str) -> JsonObject:
    return {
        "class_type": "CLIPTextEncodeFlux",
        "inputs": {"clip": ["6", 0], "clip_l": text, "t5xxl": text, "guidance": 3.5},
    }


def _image_info(path: Path) -> tuple[int, int, bool]:
    with Image.open(path) as image:
        extrema = image.convert("L").getextrema()
        return image.width, image.height, extrema[0] == extrema[1]


def _read_jsonl(path: Path) -> tuple[JsonObject, ...]:
    rows: list[JsonObject] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        raw: JsonValue = json.loads(line)
        if isinstance(raw, dict):
            rows.append(raw)
    return tuple(rows)


if __name__ == "__main__":
    run_c081_generation(C081GenerationConfig())
