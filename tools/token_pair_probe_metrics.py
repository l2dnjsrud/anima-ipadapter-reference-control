from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Final

from tools.siglip_auto_caption_types import JsonObject

if TYPE_CHECKING:
    import torch


TOKEN_METRICS: Final = ("pooled", "mean_max_token", "topk_token")


@dataclass(frozen=True, slots=True)
class TokenPairScore:
    pair_id: str
    label: str
    anchor_id: str
    candidate_id: str
    scores: dict[str, float]


@dataclass(frozen=True, slots=True)
class TokenMetricSummary:
    positive_mean: float
    negative_mean: float
    separation_margin: float
    pairwise_auc: float
    midpoint_accuracy: float
    decision: str


@dataclass(frozen=True, slots=True)
class TokenProbeInputError(Exception):
    detail: str

    def __str__(self) -> str:
        return self.detail


def score_token_pair(
    anchor_tokens: torch.Tensor,
    candidate_tokens: torch.Tensor,
    *,
    topk: int,
) -> dict[str, float]:
    similarities = anchor_tokens @ candidate_tokens.T
    pooled = _cosine(anchor_tokens.mean(dim=0), candidate_tokens.mean(dim=0))
    mean_max = similarities.max(dim=1).values.mean()
    capped_topk = min(topk, similarities.numel())
    topk_score = similarities.flatten().topk(capped_topk).values.mean()
    return {
        "pooled": float(pooled.item()),
        "mean_max_token": float(mean_max.item()),
        "topk_token": float(topk_score.item()),
    }


def summarize_token_metrics(rows: list[TokenPairScore]) -> dict[str, JsonObject]:
    return {metric: asdict(_summarize_metric(rows, metric=metric)) for metric in TOKEN_METRICS}


def render_token_probe_report(result: JsonObject) -> str:
    summaries = result["summaries"]
    if not isinstance(summaries, dict):
        raise TokenProbeInputError("result summaries must be an object")
    lines = [
        "# SigLIP Token Pair Probe",
        "",
        f"- Encoder: `{result['encoder']}`",
        f"- Layer: `{result['layer']}`",
        f"- Top-k: `{result['topk']}`",
        f"- Manifest: `{result['manifest_path']}`",
        "",
        "| metric | positive mean | negative mean | margin | pairwise AUC | decision |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for metric in TOKEN_METRICS:
        summary = summaries[metric]
        lines.append(
            f"| `{metric}` | {summary['positive_mean']:.6f} | "
            f"{summary['negative_mean']:.6f} | {summary['separation_margin']:.6f} | "
            f"{summary['pairwise_auc']:.6f} | `{summary['decision']}` |"
        )
    lines.extend(
        [
            "",
            "This probe compares hidden-token similarity, not generated image quality.",
            "",
        ]
    )
    return "\n".join(lines)


def _cosine(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    import torch

    return torch.nn.functional.cosine_similarity(left, right, dim=0)


def _summarize_metric(rows: list[TokenPairScore], *, metric: str) -> TokenMetricSummary:
    positives = [row.scores[metric] for row in rows if row.label == "positive"]
    negatives = [row.scores[metric] for row in rows if row.label == "negative"]
    if not positives or not negatives:
        raise TokenProbeInputError("manifest must contain positive and negative rows")
    positive_mean = sum(positives) / len(positives)
    negative_mean = sum(negatives) / len(negatives)
    margin = positive_mean - negative_mean
    auc = _pairwise_auc(positives, negatives)
    threshold = (positive_mean + negative_mean) / 2.0
    midpoint_accuracy = (
        sum(1 for value in positives if value >= threshold)
        + sum(1 for value in negatives if value < threshold)
    ) / (len(positives) + len(negatives))
    return TokenMetricSummary(
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
        return "token_feature_separates_pairs"
    return "token_feature_not_sufficiently_separated"
