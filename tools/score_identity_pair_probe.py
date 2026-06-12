from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Annotated

import typer

from tools.image_feature_embedders import EncoderName, ImageEmbedder, build_image_embedder
from tools.siglip_auto_caption_types import JsonObject

if TYPE_CHECKING:
    import torch


@dataclass(frozen=True, slots=True)
class PairProbeRow:
    pair_id: str
    label: str
    anchor_id: str
    candidate_id: str
    anchor_group: str
    candidate_group: str
    cosine: float


@dataclass(frozen=True, slots=True)
class PairProbeSummary:
    pairs: int
    positive_pairs: int
    negative_pairs: int
    positive_mean: float
    negative_mean: float
    separation_margin: float
    pairwise_auc: float
    midpoint_accuracy: float
    decision: str


@dataclass(frozen=True, slots=True)
class PairProbeInputError(Exception):
    detail: str

    def __str__(self) -> str:
        return self.detail


def score_pair_probe_manifest(
    manifest_path: Path,
    *,
    data_root: Path,
    embedder: ImageEmbedder,
    encoder_name: str,
) -> JsonObject:
    embedding_cache: dict[Path, torch.Tensor] = {}
    rows: list[PairProbeRow] = []
    for raw in _read_manifest_rows(manifest_path):
        anchor_id = str(raw["anchor_id"])
        candidate_id = str(raw["candidate_id"])
        anchor_path = data_root / f"{anchor_id}.jpg"
        candidate_path = data_root / f"{candidate_id}.jpg"
        cosine = _cosine(
            _embedding_cached(embedder, embedding_cache, anchor_path),
            _embedding_cached(embedder, embedding_cache, candidate_path),
        )
        rows.append(
            PairProbeRow(
                pair_id=str(raw["pair_id"]),
                label=str(raw["label"]),
                anchor_id=anchor_id,
                candidate_id=candidate_id,
                anchor_group=str(raw["anchor_group"]),
                candidate_group=str(raw["candidate_group"]),
                cosine=cosine,
            )
        )
    summary = _summarize(rows)
    return {
        "manifest_path": str(manifest_path),
        "data_root": str(data_root),
        "encoder": encoder_name,
        "rows": [asdict(row) for row in rows],
        "summary": asdict(summary),
    }


def write_pair_probe_report(result: JsonObject, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(_render_report(result), encoding="utf-8")


def _read_manifest_rows(manifest_path: Path) -> tuple[JsonObject, ...]:
    rows: list[JsonObject] = []
    with manifest_path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            raw = json.loads(line)
            if not isinstance(raw, dict):
                raise PairProbeInputError(f"manifest row {line_number} must be an object")
            rows.append(raw)
    if not rows:
        raise PairProbeInputError(f"manifest has no rows: {manifest_path}")
    return tuple(rows)


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


def _summarize(rows: list[PairProbeRow]) -> PairProbeSummary:
    positives = [row.cosine for row in rows if row.label == "positive"]
    negatives = [row.cosine for row in rows if row.label == "negative"]
    if not positives or not negatives:
        raise PairProbeInputError("manifest must contain positive and negative rows")
    positive_mean = sum(positives) / len(positives)
    negative_mean = sum(negatives) / len(negatives)
    margin = positive_mean - negative_mean
    auc = _pairwise_auc(positives, negatives)
    threshold = (positive_mean + negative_mean) / 2.0
    midpoint_accuracy = (
        sum(1 for value in positives if value >= threshold)
        + sum(1 for value in negatives if value < threshold)
    ) / (len(positives) + len(negatives))
    return PairProbeSummary(
        pairs=len(rows),
        positive_pairs=len(positives),
        negative_pairs=len(negatives),
        positive_mean=positive_mean,
        negative_mean=negative_mean,
        separation_margin=margin,
        pairwise_auc=auc,
        midpoint_accuracy=midpoint_accuracy,
        decision=_decision(margin=margin, auc=auc),
    )


def _pairwise_auc(positives: list[float], negatives: list[float]) -> float:
    wins = 0.0
    for positive in positives:
        for negative in negatives:
            if positive > negative:
                wins += 1.0
            elif positive == negative:
                wins += 0.5
    return wins / (len(positives) * len(negatives))


def _decision(*, margin: float, auc: float) -> str:
    if margin >= 0.05 and auc >= 0.70:
        return "feature_separates_proxy_pairs"
    return "feature_not_sufficiently_separated"


def _render_report(result: JsonObject) -> str:
    summary = result["summary"]
    if not isinstance(summary, dict):
        raise PairProbeInputError("result summary must be an object")
    return "\n".join(
        [
            "# Identity Feature Probe",
            "",
            f"- Encoder: `{result['encoder']}`",
            f"- Manifest: `{result['manifest_path']}`",
            f"- Pairs: `{summary['pairs']}`",
            f"- Positive mean: `{summary['positive_mean']}`",
            f"- Negative mean: `{summary['negative_mean']}`",
            f"- Separation margin: `{summary['separation_margin']}`",
            f"- Pairwise AUC: `{summary['pairwise_auc']}`",
            f"- Midpoint accuracy: `{summary['midpoint_accuracy']}`",
            f"- Decision: `{summary['decision']}`",
            "",
            "Positive pairs are a weak same-SG proxy, not verified same-character labels.",
            "This gate only decides whether an encoder deserves a stricter identity-pair run.",
            "",
        ]
    )


app = typer.Typer(add_completion=False)


@app.command()
def main(
    manifest_path: Annotated[Path, typer.Argument()],
    output_path: Annotated[Path, typer.Argument()],
    data_root: Annotated[Path, typer.Option()],
    encoder: Annotated[EncoderName, typer.Option()] = "qwenvl",
    model_id: Annotated[str | None, typer.Option()] = None,
    device: Annotated[str, typer.Option()] = "auto",
    report_path: Annotated[Path | None, typer.Option()] = None,
) -> None:
    encoder_name, embedder = build_image_embedder(
        encoder,
        model_id=model_id,
        device=device,
    )
    result = score_pair_probe_manifest(
        manifest_path,
        data_root=data_root,
        embedder=embedder,
        encoder_name=encoder_name,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    if report_path is not None:
        write_pair_probe_report(result, report_path)
    typer.echo(f"wrote {output_path}")


if __name__ == "__main__":
    app()
