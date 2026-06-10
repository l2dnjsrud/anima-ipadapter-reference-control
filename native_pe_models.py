from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Mapping

import torch
from safetensors import safe_open

try:
    import folder_paths
except ModuleNotFoundError:
    folder_paths = None


DEFAULT_ANIMA_ROOT: Final[Path] = Path(
    os.environ.get("ANIMA_LORA_ROOT", "/home/wktwin/anima-lora-training-bundle/anima_lora")
)
DEFAULT_COMFY_MODELS_ROOT: Final[Path] = Path(
    os.environ.get("ANIMA_COMFY_MODELS_ROOT", "/data/ai/models")
)
DEFAULT_IPADAPTER_NAME: Final[str] = "anima_ip_adapter_quality_20260610.safetensors"
SUPPORTED_ENCODERS: Final[tuple[str, ...]] = ("pe",)
DTYPE_CHOICES: Final[tuple[str, ...]] = ("bfloat16", "float16", "float32")
DEVICE_CHOICES: Final[tuple[str, ...]] = ("auto", "cuda", "cpu")


@dataclass(frozen=True, slots=True)
class AnimaPEIPAdapterSpec:
    name: str
    path: Path
    metadata: Mapping[str, str]


@dataclass(frozen=True, slots=True)
class AnimaPEFeatures:
    features: torch.Tensor
    encoder_name: str
    source_size: tuple[int, int]


def ensure_model_folders() -> None:
    if folder_paths is None:
        return
    if "ipadapter" not in folder_paths.folder_names_and_paths:
        folder_paths.folder_names_and_paths["ipadapter"] = (
            [str(Path(folder_paths.models_dir) / "ipadapter")],
            folder_paths.supported_pt_extensions,
        )
    folder = DEFAULT_COMFY_MODELS_ROOT / "ipadapter"
    if folder.exists():
        folder_paths.add_model_folder_path("ipadapter", str(folder), is_default=True)


def model_names(folder_name: str, preferred_name: str) -> list[str]:
    if folder_paths is None:
        return [preferred_name]
    ensure_model_folders()
    names = folder_paths.get_filename_list(folder_name)
    if preferred_name in names:
        return [preferred_name, *[name for name in names if name != preferred_name]]
    return names or [preferred_name]


def model_path(folder_name: str, model_name: str) -> Path:
    if folder_paths is None:
        return DEFAULT_COMFY_MODELS_ROOT / folder_name / model_name
    ensure_model_folders()
    return Path(folder_paths.get_full_path_or_raise(folder_name, model_name))


def read_safetensors_metadata(path: Path) -> dict[str, str]:
    if path.suffix != ".safetensors":
        raise ValueError(f"PE IP-Adapter checkpoint must be .safetensors, got: {path.name}")
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        return dict(handle.metadata() or {})


def validate_pe_ipadapter_metadata(path: Path, metadata: Mapping[str, str]) -> None:
    network_spec = metadata.get("ss_network_spec")
    encoder = metadata.get("ss_encoder")
    if network_spec != "ip_adapter" or encoder != "pe":
        raise ValueError(
            "Expected a PE-Core Anima IP-Adapter checkpoint "
            f"(ss_network_spec='ip_adapter', ss_encoder='pe'), got "
            f"ss_network_spec={network_spec!r}, ss_encoder={encoder!r} for {path.name}."
        )
    required = (
        "ss_encoder_dim",
        "ss_context_dim",
        "ss_num_ip_tokens",
        "ss_num_blocks",
        "ss_hidden_size",
        "ss_num_heads",
    )
    missing = [key for key in required if key not in metadata]
    if missing:
        raise ValueError(f"PE IP-Adapter checkpoint {path.name} is missing metadata: {missing}")


def load_pe_adapter_spec(path: Path, name: str | None = None) -> AnimaPEIPAdapterSpec:
    metadata = read_safetensors_metadata(path)
    validate_pe_ipadapter_metadata(path, metadata)
    return AnimaPEIPAdapterSpec(
        name=name or path.name,
        path=path,
        metadata=metadata,
    )
