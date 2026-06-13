from __future__ import annotations

from dataclasses import dataclass
from typing import Final

from tools.siglip_auto_caption_types import JsonObject

DIRECT_GREEN_QUERY: Final = "direct_green_non_human_face"
NEGATIVE_LABEL_HINTS: Final = (
    "false_positive",
    "guard_false_positive",
    "background_object",
    "human",
)
NEGATIVE_QUERY_HINTS: Final = (
    "human_negative",
    "old_headwear_negative",
    "background_object_green",
)
PROXY_LABEL_HINTS: Final = (
    "useful_proxy",
    "proxy",
    "red_eye",
    "side_profile",
    "beard",
)


@dataclass(frozen=True, slots=True)
class C101LabelDecision:
    manual_label: str
    confidence: str
    evidence: str
    review_source: str


def decide_c101_label(candidate: JsonObject, prior: JsonObject | None) -> C101LabelDecision:
    if prior is None:
        return C101LabelDecision(
            "unclear",
            "low",
            f"no prior visual review; source_bucket={candidate.get('source_bucket', '')}",
            "conservative_auto",
        )
    prior_label = str(prior.get("review_label", ""))
    query_key = str(prior.get("query_key", ""))
    if prior_label == "target_positive" and query_key == DIRECT_GREEN_QUERY:
        return C101LabelDecision(
            "local_positive",
            "high",
            _prior_evidence(prior),
            "prior_visual_review",
        )
    lowered = prior_label.lower()
    query_lowered = query_key.lower()
    if any(hint in lowered for hint in NEGATIVE_LABEL_HINTS) or any(
        hint in query_lowered for hint in NEGATIVE_QUERY_HINTS
    ):
        return C101LabelDecision(
            "local_negative",
            "high",
            _prior_evidence(prior),
            "prior_visual_review",
        )
    if prior_label == "target_positive" or any(hint in lowered for hint in PROXY_LABEL_HINTS):
        return C101LabelDecision(
            "unclear",
            "medium",
            f"proxy is not direct-green/non-human positive: {_prior_evidence(prior)}",
            "prior_visual_review",
        )
    return C101LabelDecision(
        "unclear",
        "low",
        _prior_evidence(prior),
        "prior_visual_review",
    )


def _prior_evidence(prior: JsonObject) -> str:
    return "; ".join(
        (
            f"review_label={prior.get('review_label', '')}",
            f"query_key={prior.get('query_key', '')}",
            f"note={prior.get('review_note', '')}",
        )
    )
