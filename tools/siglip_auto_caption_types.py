from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Final


type JsonValue = None | bool | int | float | str | list[JsonValue] | dict[str, JsonValue]
type JsonObject = dict[str, JsonValue]

DEFAULT_DATA_ROOT: Final = Path(
    "/home/wktwin/anima-lora-training-bundle/image_dataset_color_panel_style_v5_best"
)
DEFAULT_BASE_URL: Final = "http://127.0.0.1:8116"
DEFAULT_OUT_DIR: Final = Path(
    "eval/siglip_runtime_quality_20260612_c033_auto_caption_runtime"
)
DEFAULT_COMFY_INPUT: Final = Path(".tmp/comfy_qwenvl_single/input")
DEFAULT_COMFY_OUTPUT: Final = Path(".tmp/comfy_qwenvl_single/output")
NEGATIVE: Final = (
    "low quality, blurry, bad anatomy, deformed face, text, watermark, extra limbs, "
    "abstract, washed out, noise, nude, explicit, gore, multiple people"
)
SIGLIP_KV_INIT_LABEL: Final = "siglip_kv_init_w14"
SIGLIP_REF_RETRIEVAL_LABEL: Final = "siglip_ref_retrieval_w14"
SIGLIP_KV_INIT_CHECKPOINT: Final = (
    "anima_siglip_ip_adapter_single_character_clean32_pe_space_init_0512_20260611.safetensors"
)
SIGLIP_REF_RETRIEVAL_CHECKPOINT: Final = (
    "anima_siglip_ip_adapter_single_character_clean32_pe_retrieval_0128_20260611.safetensors"
)


@dataclass(frozen=True, slots=True)
class AutoPromptRow:
    ref_id: str
    tgt_id: str
    source_prompt: str
    prompt: str
    selected_attributes: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class Sample:
    label: str
    ref_id: str
    seed: int
    prompt_row: AutoPromptRow


@dataclass(frozen=True, slots=True)
class Variant:
    label: str
    checkpoint: str | None
    weight: float


@dataclass(frozen=True, slots=True)
class EvalConfig:
    data_root: Path = DEFAULT_DATA_ROOT
    base_url: str = DEFAULT_BASE_URL
    out_dir: Path = DEFAULT_OUT_DIR
    comfy_input: Path = DEFAULT_COMFY_INPUT
    comfy_output: Path = DEFAULT_COMFY_OUTPUT
