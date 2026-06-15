# /// script
# dependencies = ["typer", "torch"]
# ///
# --- How to run -----------------------------------------------------
# PYTHONPATH=. /home/wktwin/anima-lora-training-bundle/anima_lora/.venv/bin/python \
#   tools/c088_embedding_metrics.py eval/c088_shape_silhouette_feature_probe_20260613/probe_manifest.jsonl \
#   eval/c088_shape_silhouette_feature_probe_20260613/qwenvl_embedding_metrics.json --encoder qwenvl

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Annotated

import typer

from tools.c088_probe_io import DEFAULT_OUT_DIR, VARIANTS, json_object, read_manifest_rows, write_json
from tools.image_feature_embedders import EncoderName, ImageEmbedder, build_image_embedder
from tools.siglip_auto_caption_types import JsonObject

if TYPE_CHECKING:
    import torch


SUPPORT_UPLIFT = 0.05
SUPPORT_MARGIN = 0.01


@dataclass(frozen=True, slots=True)
class EmbeddingScoreRow:
    sample: str
    shape_group: str
    variant: str
    cosine: float
    no_ip_cosine: float
    uplift: float
    rank: int


@dataclass(frozen=True, slots=True)
class EmbeddingCaseDecision:
    sample: str
    shape_group: str
    best_variant: str
    best_uplift: float
    top_margin: float
    decision: str


def score_embedding_manifest(
    manifest_path: Path,
    *,
    embedder: ImageEmbedder,
    encoder_name: str,
) -> JsonObject:
    cases = read_manifest_rows(manifest_path)
    cache: dict[Path, torch.Tensor] = {}
    rows: list[EmbeddingScoreRow] = []
    decisions: list[EmbeddingCaseDecision] = []
    for case in cases:
        ref = _embedding_cached(embedder, cache, Path(str(case["reference_path"])))
        scored = tuple(_score_variant(case, variant, ref, embedder, cache) for variant in VARIANTS)
        ranked = sorted(scored, key=lambda row: row.cosine, reverse=True)
        no_ip_cosine = next(row.cosine for row in ranked if row.variant == "no_ip")
        ranks = {row.variant: index for index, row in enumerate(ranked, start=1)}
        rows.extend(_ranked_rows(scored, ranks, no_ip_cosine))
        decisions.append(_case_decision(ranked, no_ip_cosine))
    return {
        "manifest_path": str(manifest_path),
        "encoder": encoder_name,
        "rows": [asdict(row) for row in sorted(rows, key=lambda row: (row.sample, row.rank))],
        "case_decisions": [asdict(decision) for decision in decisions],
        "summary": _summary(decisions),
    }


def _score_variant(
    case: JsonObject,
    variant: str,
    ref: "torch.Tensor",
    embedder: ImageEmbedder,
    cache: dict[Path, "torch.Tensor"],
) -> EmbeddingScoreRow:
    candidate_path = Path(str(json_object(case, "candidates")[variant]))
    cosine = _cosine(ref, _embedding_cached(embedder, cache, candidate_path))
    return EmbeddingScoreRow(
        sample=str(case["sample"]),
        shape_group=str(case["shape_group"]),
        variant=variant,
        cosine=cosine,
        no_ip_cosine=0.0,
        uplift=0.0,
        rank=0,
    )


def _ranked_rows(
    rows: tuple[EmbeddingScoreRow, ...],
    ranks: dict[str, int],
    no_ip_cosine: float,
) -> tuple[EmbeddingScoreRow, ...]:
    return tuple(
        EmbeddingScoreRow(
            sample=row.sample,
            shape_group=row.shape_group,
            variant=row.variant,
            cosine=row.cosine,
            no_ip_cosine=no_ip_cosine,
            uplift=row.cosine - no_ip_cosine,
            rank=ranks[row.variant],
        )
        for row in rows
    )


def _embedding_cached(
    embedder: ImageEmbedder,
    cache: dict[Path, "torch.Tensor"],
    image_path: Path,
) -> "torch.Tensor":
    if image_path not in cache:
        cache[image_path] = embedder.encode_image(image_path)
    return cache[image_path]


def _cosine(left: "torch.Tensor", right: "torch.Tensor") -> float:
    import torch

    return float(torch.nn.functional.cosine_similarity(left, right, dim=0).item())


def _case_decision(
    ranked: list[EmbeddingScoreRow],
    no_ip_cosine: float,
) -> EmbeddingCaseDecision:
    best = ranked[0]
    second = ranked[1]
    best_uplift = best.cosine - no_ip_cosine
    top_margin = best.cosine - second.cosine
    supports = best.variant != "no_ip" and best_uplift >= SUPPORT_UPLIFT and top_margin >= SUPPORT_MARGIN
    return EmbeddingCaseDecision(
        sample=best.sample,
        shape_group=best.shape_group,
        best_variant=best.variant,
        best_uplift=best_uplift,
        top_margin=top_margin,
        decision="embedding_signal_supports_supervised_objective" if supports else "embedding_signal_not_enough",
    )


def _summary(decisions: list[EmbeddingCaseDecision]) -> JsonObject:
    supported = [item for item in decisions if item.decision == "embedding_signal_supports_supervised_objective"]
    support_rate = len(supported) / len(decisions)
    return {
        "cases": len(decisions),
        "supported_cases": len(supported),
        "support_rate": support_rate,
        "decision": "embedding_signal_viable" if support_rate >= 0.5 else "embedding_signal_not_viable",
    }


app = typer.Typer(add_completion=False)


@app.command()
def main(
    manifest_path: Annotated[Path, typer.Argument()] = DEFAULT_OUT_DIR / "probe_manifest.jsonl",
    output_path: Annotated[Path, typer.Argument()] = DEFAULT_OUT_DIR / "qwenvl_embedding_metrics.json",
    encoder: Annotated[EncoderName, typer.Option()] = "qwenvl",
    model_id: Annotated[str | None, typer.Option()] = None,
    device: Annotated[str, typer.Option()] = "auto",
) -> None:
    encoder_name, embedder = build_image_embedder(encoder, model_id=model_id, device=device)
    result = score_embedding_manifest(
        manifest_path,
        embedder=embedder,
        encoder_name=encoder_name,
    )
    write_json(output_path, result)
    typer.echo(f"wrote {output_path}")


if __name__ == "__main__":
    app()
