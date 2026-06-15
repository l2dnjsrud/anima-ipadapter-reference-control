from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Annotated

import numpy as np
import typer
from PIL import Image


@dataclass(frozen=True, slots=True)
class MetricRow:
    sample: str
    variant: str
    cosine: float
    no_ip_cosine: float
    uplift: float
    pixel_std: float


@dataclass(frozen=True, slots=True)
class VariantMetricSummary:
    variant: str
    cases: int
    mean_cosine: float
    mean_no_ip_cosine: float
    mean_uplift: float
    improved_rate: float


def score_auto_caption_summary(
    summary_path: Path,
    *,
    data_root: Path,
    device: str,
) -> dict:
    import torch
    from library.vision.encoder import load_pe_encoder

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    bundle = load_pe_encoder(torch.device(device), name="pe", dtype=torch.bfloat16)
    pooled_cache: dict[Path, torch.Tensor] = {}
    rows: list[MetricRow] = []
    for sample in summary["samples"]:
        sample_label = str(sample["label"])
        ref_path = data_root / f"{sample['ref_id']}.jpg"
        no_ip_path = Path(str(summary["results"][f"{sample_label}_no_ip"]["image"]))
        ref_embedding = _pooled_cached(bundle, pooled_cache, ref_path)
        no_ip_cosine = _cosine(ref_embedding, _pooled_cached(bundle, pooled_cache, no_ip_path))
        for variant in summary["variants"]:
            variant_label = str(variant["label"])
            if variant_label == "no_ip":
                continue
            image_path = Path(str(summary["results"][f"{sample_label}_{variant_label}"]["image"]))
            cosine = _cosine(ref_embedding, _pooled_cached(bundle, pooled_cache, image_path))
            rows.append(
                MetricRow(
                    sample=sample_label,
                    variant=variant_label,
                    cosine=cosine,
                    no_ip_cosine=no_ip_cosine,
                    uplift=cosine - no_ip_cosine,
                    pixel_std=_pixel_std(image_path),
                )
            )
    variant_summaries = _summarize_variants(rows)
    return {
        "summary_path": str(summary_path),
        "data_root": str(data_root),
        "rows": [asdict(row) for row in rows],
        "variant_summaries": [asdict(item) for item in variant_summaries],
    }


def _pooled_cached(bundle, cache, image_path: Path):
    if image_path not in cache:
        import torch

        with torch.no_grad():
            cache[image_path] = _encode_pe_pooled(bundle, image_path)
    return cache[image_path]


def _image_to_minus1to1(image_path: Path):
    import torch

    with Image.open(image_path) as image:
        arr = np.asarray(image.convert("RGB"), dtype=np.float32)
    return torch.from_numpy(arr / 127.5 - 1.0).permute(2, 0, 1).contiguous()


def _encode_pe_pooled(bundle, image_path: Path):
    from library.training.cmmd import pool_and_normalize
    from library.vision.encoder import encode_pe_from_imageminus1to1

    tensor = _image_to_minus1to1(image_path)
    feats = encode_pe_from_imageminus1to1(bundle, tensor.unsqueeze(0), same_bucket=True)[0]
    return pool_and_normalize(feats).cpu()


def _cosine(left, right) -> float:
    import torch

    return float(torch.nn.functional.cosine_similarity(left, right, dim=0).item())


def _pixel_std(image_path: Path) -> float:
    with Image.open(image_path) as image:
        arr = np.asarray(image.convert("RGB"), dtype=np.float32)
    return float(arr.std())


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
    device: Annotated[str, typer.Option()] = "cuda:0",
) -> None:
    metrics = score_auto_caption_summary(summary_path, data_root=data_root, device=device)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    typer.echo(f"wrote {output_path}")


if __name__ == "__main__":
    app()
