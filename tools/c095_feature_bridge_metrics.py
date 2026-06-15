from __future__ import annotations

from tools.c090_siglip_hard_shape_data import BASELINE_VARIANTS
from tools.c092_siglip_qwen_target_eval import (
    _best_named,
    _best_prefixed,
    _variant_summaries,
)
from tools.siglip_auto_caption_types import JsonObject


def summarize_c095_shape_rows(
    rows: tuple[JsonObject, ...],
    *,
    diversity_by_variant: dict[str, float] | None = None,
    pixel_audit: JsonObject | None = None,
) -> JsonObject:
    summaries = _variant_summaries(rows)
    c095 = _best_prefixed(summaries, "c095_")
    c094 = _best_named(summaries, ("c094_shape_supervised_w14",))
    c093 = _best_named(summaries, ("c093_anti_collapse_w14",))
    c092 = _best_named(summaries, ("c092_qwen_target_w10", "c092_qwen_target_w14"))
    qwen = _best_named(summaries, BASELINE_VARIANTS)
    diversity = diversity_by_variant or {}
    heldout = _heldout_summary(rows, c095[0])
    blank_rows = _c095_blank_like_rows(pixel_audit or {})
    return {
        "variant_summaries": summaries,
        "best_c095_variant": c095[0],
        "best_c094_variant": c094[0],
        "best_c093_variant": c093[0],
        "best_c092_variant": c092[0],
        "best_qwen_baseline_variant": qwen[0],
        "heldout07": heldout,
        "train_crop": _train_crop_summary(rows, c095[0]),
        "diversity_proxy": {
            "best_c095": diversity.get(c095[0], 0.0),
            "c094_shape_supervised_w14": diversity.get("c094_shape_supervised_w14", 0.0),
            "by_variant": diversity,
        },
        "c095_blank_like_rows": blank_rows,
        "decision": _c095_decision(c095, c094, qwen, summaries, heldout, diversity, blank_rows),
    }


def _heldout_summary(rows: tuple[JsonObject, ...], best_c095: str) -> JsonObject:
    return {
        "best_c095_variant": best_c095,
        "best_c095_uplift": _uplift(rows, "heldout07", best_c095),
        "c094_w14_uplift": _uplift(rows, "heldout07", "c094_shape_supervised_w14"),
        "c093_w14_uplift": _uplift(rows, "heldout07", "c093_anti_collapse_w14"),
        "c092_w14_uplift": _uplift(rows, "heldout07", "c092_qwen_target_w14"),
    }


def _train_crop_summary(rows: tuple[JsonObject, ...], best_c095: str) -> JsonObject:
    values = [
        float(row["uplift"])
        for row in rows
        if str(row.get("sample", "")).startswith("crop_pair")
        and row.get("variant") == best_c095
    ]
    return {"best_c095_mean_uplift": sum(values) / len(values) if values else 0.0}


def _uplift(rows: tuple[JsonObject, ...], sample: str, variant: str) -> float:
    for row in rows:
        if row.get("sample") == sample and row.get("variant") == variant:
            return float(row["uplift"])
    return float("-inf")


def _c095_blank_like_rows(pixel_audit: JsonObject) -> list[str]:
    raw_rows = pixel_audit.get("rows")
    if not isinstance(raw_rows, list):
        return []
    names: list[str] = []
    for row in raw_rows:
        if not isinstance(row, dict) or row.get("nonblank") is not False:
            continue
        name = str(row.get("name", ""))
        if "_c095_" in name:
            names.append(name)
    return names


def _c095_decision(
    c095: tuple[str, float],
    c094: tuple[str, float],
    qwen: tuple[str, float],
    summaries: JsonObject,
    heldout: JsonObject,
    diversity: dict[str, float],
    blank_rows: list[str],
) -> str:
    best_raw = summaries.get(c095[0], {})
    improved_rate = (
        float(best_raw.get("improved_rate", 0.0)) if isinstance(best_raw, dict) else 0.0
    )
    heldout_best = float(heldout["best_c095_uplift"])
    heldout_c094 = float(heldout["c094_w14_uplift"])
    if (
        not blank_rows
        and c095[1] >= 0.100
        and c095[1] >= c094[1] + 0.010
        and c095[1] >= qwen[1] - 0.010
        and improved_rate >= 0.90
        and heldout_best >= 0.030
        and heldout_best >= heldout_c094 + 0.020
        and diversity.get(c095[0], 0.0)
        >= diversity.get("c094_shape_supervised_w14", 0.0) + 0.005
    ):
        return "c095_candidate_for_broader_shape_gate"
    return "c095_feature_bridge_not_promoted_requires_siglip_encoder_finetune_or_data_expansion"
