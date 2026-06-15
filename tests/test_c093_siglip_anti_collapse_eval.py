from __future__ import annotations

from tools.c093_siglip_anti_collapse_eval import summarize_c093_shape_rows


def test_summarize_c093_requires_heldout_and_diversity_improvement() -> None:
    rows = (
        *_rows("crop_pair00", 0.09, 0.08),
        *_rows("crop_pair01", 0.10, 0.08),
        *_rows("heldout07", 0.05, 0.02),
        {"sample": "heldout07", "variant": "c087_expanded_crop_positive_w14", "uplift": 0.10},
    )

    summary = summarize_c093_shape_rows(
        rows,
        diversity_by_variant={
            "c093_anti_collapse_w12": 0.040,
            "c092_qwen_target_w14": 0.030,
        },
    )

    assert summary["best_c093_variant"] == "c093_anti_collapse_w12"
    assert summary["heldout07"]["best_c093_uplift"] == 0.05
    assert summary["decision"] == "c093_candidate_for_next_gate"


def test_summarize_c093_rejects_mean_only_improvement_when_heldout_fails() -> None:
    rows = (
        *_rows("crop_pair00", 0.12, 0.08),
        *_rows("crop_pair01", 0.12, 0.08),
        *_rows("heldout07", 0.021, 0.02),
        {"sample": "heldout07", "variant": "c087_expanded_crop_positive_w14", "uplift": 0.10},
    )

    summary = summarize_c093_shape_rows(
        rows,
        diversity_by_variant={
            "c093_anti_collapse_w12": 0.050,
            "c092_qwen_target_w14": 0.030,
        },
    )

    assert summary["variant_summaries"]["c093_anti_collapse_w12"]["mean_uplift"] > 0.08
    assert summary["decision"] == "c093_anti_collapse_not_promoted"


def _rows(sample: str, c093: float, c092: float) -> tuple[dict[str, object], ...]:
    return (
        {"sample": sample, "variant": "c089_shape_w14", "uplift": 0.02},
        {"sample": sample, "variant": "c092_qwen_target_w14", "uplift": c092},
        {"sample": sample, "variant": "c093_anti_collapse_w12", "uplift": c093},
    )
