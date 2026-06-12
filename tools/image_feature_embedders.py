from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Final, Literal, Protocol, assert_never

import numpy as np
from PIL import Image, ImageOps

from tools.score_auto_caption_qwenvl_metrics import (
    DEFAULT_IMAGE_INSTRUCTION,
    DEFAULT_QWENVL_MODEL_ID,
    Qwen3VLImageEmbedder,
    Qwen3VLImageEmbedderConfig,
)

if TYPE_CHECKING:
    import torch


DEFAULT_SIGLIP_MODEL_ID: Final = "google/siglip2-base-patch16-512"
EncoderName = Literal["qwenvl", "siglip", "pe"]


class ImageEmbedder(Protocol):
    def encode_image(self, image_path: Path) -> torch.Tensor: ...


@dataclass(frozen=True, slots=True)
class SigLIPImageEmbedderConfig:
    model_id: str = DEFAULT_SIGLIP_MODEL_ID
    device: str = "auto"


class SigLIPImageEmbedder:
    def __init__(self, config: SigLIPImageEmbedderConfig) -> None:
        import torch
        from transformers import AutoImageProcessor, SiglipVisionModel

        resolved_device = _resolve_device(config.device)
        dtype = torch.bfloat16 if resolved_device == "cuda" else torch.float32
        self._device = torch.device(resolved_device)
        self._dtype = dtype
        self._processor = AutoImageProcessor.from_pretrained(config.model_id)
        self._model = SiglipVisionModel.from_pretrained(
            config.model_id,
            torch_dtype=dtype,
            trust_remote_code=True,
        ).to(self._device)
        self._model.eval()

    def encode_image(self, image_path: Path) -> torch.Tensor:
        import torch

        image = _load_square_rgb(image_path, resolution=512)
        inputs = self._processor(images=[image], return_tensors="pt", do_resize=False)
        prepared = {
            key: value.to(device=self._device, dtype=self._dtype)
            for key, value in inputs.items()
        }
        with torch.no_grad():
            outputs = self._model(**prepared)
        pooled = outputs.last_hidden_state.mean(dim=1)[0].detach().float().cpu()
        return torch.nn.functional.normalize(pooled, dim=0)


@dataclass(frozen=True, slots=True)
class PEImageEmbedderConfig:
    device: str = "auto"


class PEImageEmbedder:
    def __init__(self, config: PEImageEmbedderConfig) -> None:
        import torch
        from library.vision.encoder import load_pe_encoder

        self._device = torch.device(_resolve_device(config.device))
        self._bundle = load_pe_encoder(self._device, name="pe", dtype=torch.bfloat16)

    def encode_image(self, image_path: Path) -> torch.Tensor:
        import torch
        from library.training.cmmd import pool_and_normalize
        from library.vision.encoder import encode_pe_from_imageminus1to1

        tensor = _image_to_minus1to1(image_path)
        with torch.no_grad():
            feats = encode_pe_from_imageminus1to1(
                self._bundle,
                tensor.unsqueeze(0),
                same_bucket=True,
            )[0]
        return pool_and_normalize(feats).detach().float().cpu()


def build_image_embedder(
    encoder: EncoderName,
    *,
    model_id: str | None,
    device: str,
) -> tuple[str, ImageEmbedder]:
    match encoder:
        case "qwenvl":
            qwen_model_id = model_id or DEFAULT_QWENVL_MODEL_ID
            return (
                qwen_model_id,
                Qwen3VLImageEmbedder(
                    Qwen3VLImageEmbedderConfig(
                        model_id=qwen_model_id,
                        instruction=DEFAULT_IMAGE_INSTRUCTION,
                    )
                ),
            )
        case "siglip":
            siglip_model_id = model_id or DEFAULT_SIGLIP_MODEL_ID
            return (
                siglip_model_id,
                SigLIPImageEmbedder(
                    SigLIPImageEmbedderConfig(
                        model_id=siglip_model_id,
                        device=device,
                    )
                ),
            )
        case "pe":
            return ("pe", PEImageEmbedder(PEImageEmbedderConfig(device=device)))
        case unreachable:
            assert_never(unreachable)


def _resolve_device(device: str) -> str:
    if device != "auto":
        return device
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def _image_to_minus1to1(image_path: Path) -> torch.Tensor:
    import torch

    with Image.open(image_path) as image:
        array = np.asarray(image.convert("RGB"), dtype=np.float32)
    return torch.from_numpy(array / 127.5 - 1.0).permute(2, 0, 1).contiguous()


def _load_square_rgb(path: Path, *, resolution: int) -> Image.Image:
    with Image.open(path) as image:
        return ImageOps.fit(
            image.convert("RGB"),
            (resolution, resolution),
            method=Image.Resampling.LANCZOS,
            centering=(0.5, 0.5),
        )
