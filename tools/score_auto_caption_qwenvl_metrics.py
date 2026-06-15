from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Final, Protocol, TypedDict

import typer
from PIL import Image

from tools.siglip_auto_caption_types import JsonObject, JsonValue

if TYPE_CHECKING:
    import torch


DEFAULT_QWENVL_MODEL_ID: Final = "Qwen/Qwen3-VL-Embedding-2B"
DEFAULT_IMAGE_INSTRUCTION: Final = (
    "Represent this manhwa/anime character reference image for identity, "
    "visible traits, palette, costume, expression, and style."
)


class QwenVLImageInput(TypedDict):
    image: Image.Image


class ImageEmbedder(Protocol):
    def encode_image(self, image_path: Path) -> torch.Tensor: ...


@dataclass(frozen=True, slots=True)
class MetricRow:
    sample: str
    variant: str
    cosine: float
    no_ip_cosine: float
    uplift: float


@dataclass(frozen=True, slots=True)
class VariantMetricSummary:
    variant: str
    cases: int
    mean_cosine: float
    mean_no_ip_cosine: float
    mean_uplift: float
    improved_rate: float


@dataclass(frozen=True, slots=True)
class Qwen3VLImageEmbedderConfig:
    model_id: str = DEFAULT_QWENVL_MODEL_ID
    instruction: str = DEFAULT_IMAGE_INSTRUCTION


class Qwen3VLImageEmbedder:
    def __init__(self, config: Qwen3VLImageEmbedderConfig) -> None:
        import torch
        from sentence_transformers import SentenceTransformer

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model_kwargs = {"torch_dtype": torch.bfloat16} if device == "cuda" else {}
        self._model = SentenceTransformer(
            config.model_id,
            device=device,
            model_kwargs=model_kwargs,
        )
        self._config = config

    def encode_image(self, image_path: Path) -> "torch.Tensor":
        import torch

        with Image.open(image_path) as image:
            image_input: QwenVLImageInput = {"image": image.convert("RGB")}
            raw = self._model.encode(
                [image_input],
                normalize_embeddings=True,
                convert_to_tensor=True,
                prompt=self._config.instruction,
            )
        tensor = torch.as_tensor(raw).detach().float().cpu()
        if tensor.ndim == 2:
            return tensor[0]
        if tensor.ndim == 1:
            return tensor
        raise QwenVLEmbeddingShapeError(image_path=image_path, ndim=tensor.ndim)


@dataclass(frozen=True, slots=True)
class QwenVLEmbeddingShapeError(Exception):
    image_path: Path
    ndim: int

    def __str__(self) -> str:
        return f"QwenVL image embedding must be rank 1 or 2: {self.image_path} ndim={self.ndim}"


def score_auto_caption_summary(
    summary_path: Path,
    *,
    data_root: Path,
    embedder: ImageEmbedder,
    encoder_name: str = DEFAULT_QWENVL_MODEL_ID,
) -> JsonObject:
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    rows: list[MetricRow] = []
    embedding_cache: dict[Path, torch.Tensor] = {}
    for sample in summary["samples"]:
        sample_label = str(sample["label"])
        ref_path = data_root / f"{sample['ref_id']}.jpg"
        ref_embedding = _embedding_cached(embedder, embedding_cache, ref_path)
        no_ip_path = Path(str(summary["results"][f"{sample_label}_no_ip"]["image"]))
        no_ip_cosine = _cosine(
            ref_embedding,
            _embedding_cached(embedder, embedding_cache, no_ip_path),
        )
        for variant in summary["variants"]:
            variant_label = str(variant["label"])
            if variant_label == "no_ip":
                continue
            image_path = Path(str(summary["results"][f"{sample_label}_{variant_label}"]["image"]))
            cosine = _cosine(ref_embedding, _embedding_cached(embedder, embedding_cache, image_path))
            rows.append(
                MetricRow(
                    sample=sample_label,
                    variant=variant_label,
                    cosine=cosine,
                    no_ip_cosine=no_ip_cosine,
                    uplift=cosine - no_ip_cosine,
                )
            )
    return {
        "summary_path": str(summary_path),
        "data_root": str(data_root),
        "encoder": encoder_name,
        "rows": [asdict(row) for row in rows],
        "variant_summaries": [asdict(item) for item in _summarize_variants(rows)],
    }


def _embedding_cached(
    embedder: ImageEmbedder,
    cache: dict[Path, torch.Tensor],
    image_path: Path,
) -> torch.Tensor:
    if image_path not in cache:
        cache[image_path] = embedder.encode_image(image_path)
    return cache[image_path]


def _cosine(left: torch.Tensor, right: torch.Tensor) -> float:
    import torch

    return float(torch.nn.functional.cosine_similarity(left, right, dim=0).item())


def _summarize_variants(rows: list[MetricRow]) -> tuple[VariantMetricSummary, ...]:
    by_variant: dict[str, list[MetricRow]] = {}
    for row in rows:
        by_variant.setdefault(row.variant, []).append(row)
    summaries: list[VariantMetricSummary] = []
    for variant, variant_rows in sorted(by_variant.items()):
        cases = len(variant_rows)
        summaries.append(
            VariantMetricSummary(
                variant=variant,
                cases=cases,
                mean_cosine=sum(row.cosine for row in variant_rows) / cases,
                mean_no_ip_cosine=sum(row.no_ip_cosine for row in variant_rows) / cases,
                mean_uplift=sum(row.uplift for row in variant_rows) / cases,
                improved_rate=sum(1 for row in variant_rows if row.uplift > 0.0) / cases,
            )
        )
    return tuple(summaries)


app = typer.Typer(add_completion=False)


@app.command()
def main(
    summary_path: Annotated[Path, typer.Argument()],
    output_path: Annotated[Path, typer.Argument()],
    data_root: Annotated[Path, typer.Option()],
    model_id: Annotated[str, typer.Option()] = DEFAULT_QWENVL_MODEL_ID,
) -> None:
    metrics = score_auto_caption_summary(
        summary_path,
        data_root=data_root,
        embedder=Qwen3VLImageEmbedder(Qwen3VLImageEmbedderConfig(model_id=model_id)),
        encoder_name=model_id,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    typer.echo(f"wrote {output_path}")


if __name__ == "__main__":
    app()
