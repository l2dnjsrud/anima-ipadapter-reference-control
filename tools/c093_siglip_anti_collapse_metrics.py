from __future__ import annotations

from tools.c090_siglip_hard_shape_data import BASELINE_VARIANTS
from tools.c092_siglip_qwen_target_eval import (
    _best_named,
    _best_prefixed,
    _variant_summaries,
)
from tools.siglip_auto_caption_types import JsonObject


def summarize_c093_shape_rows(
    rows: tuple[JsonObject, ...],
    *,
    diversity_by_variant: dict[str, float] | None = None,
) -> JsonObject:
    summaries = _variant_summaries(rows)
    c093 = _best_prefixed(summaries, "c093_")
    c092 = _best_named(summaries, ("c092_qwen_target_w10", "c092_qwen_target_w14"))
    c089 = _best_named(summaries, ("c089_shape_w14",))
    qwen = _best_named(summaries, BASELINE_VARIANTS)
    diversity = diversity_by_variant or {}
    heldout = _heldout_summary(rows, c093[0])
    return {
        "variant_summaries": summaries,
        "best_c093_variant": c093[0],
        "best_c092_variant": c092[0],
        "best_c089_variant": c089[0],
        "best_qwen_baseline_variant": qwen[0],
        "heldout07": heldout,
        "train_crop": _train_crop_summary(rows, c093[0]),
        "diversity_proxy": {
            "best_c093": diversity.get(c093[0], 0.0),
            "c092_qwen_target_w14": diversity.get("c092_qwen_target_w14", 0.0),
            "by_variant": diversity,
        },
        "decision": _c093_decision(c093, summaries, heldout, diversity),
    }


def _heldout_summary(rows: tuple[JsonObject, ...], best_c093: str) -> JsonObject:
    return {
        "best_c093_variant": best_c093,
        "best_c093_uplift": _uplift(rows, "heldout07", best_c093),
        "c092_w14_uplift": _uplift(rows, "heldout07", "c092_qwen_target_w14"),
    }


def _train_crop_summary(rows: tuple[JsonObject, ...], best_c093: str) -> JsonObject:
    values = [
        float(row["uplift"])
        for row in rows
        if str(row.get("sample", "")).startswith("crop_pair")
        and row.get("variant") == best_c093
    ]
    return {"best_c093_mean_uplift": sum(values) / len(values) if values else 0.0}


def _uplift(rows: tuple[JsonObject, ...], sample: str, variant: str) -> float:
    for row in rows:
        if row.get("sample") == sample and row.get("variant") == variant:
            return float(row["uplift"])
    return float("-inf")


def _c093_decision(
    c093: tuple[str, float],
    summaries: JsonObject,
    heldout: JsonObject,
    diversity: dict[str, float],
) -> str:
    best_raw = summaries.get(c093[0], {})
    improved_rate = (
        float(best_raw.get("improved_rate", 0.0)) if isinstance(best_raw, dict) else 0.0
    )
    heldout_best = float(heldout["best_c093_uplift"])
    heldout_c092 = float(heldout["c092_w14_uplift"])
    if (
        c093[1] >= 0.080
        and improved_rate >= 0.90
        and heldout_best >= 0.025
        and heldout_best >= heldout_c092 + 0.02
        and diversity.get(c093[0], 0.0)
        >= diversity.get("c092_qwen_target_w14", 0.0) + 0.005
    ):
        return "c093_candidate_for_next_gate"
    return "c093_anti_collapse_not_promoted"
