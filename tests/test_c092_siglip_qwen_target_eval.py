from __future__ import annotations

from tools.c092_siglip_qwen_target_eval import summarize_c092_shape_rows


def test_summarize_c092_shape_rows_detects_improvement_over_c089() -> None:
    rows = (
        {"variant": "c089_shape_w14", "uplift": 0.02},
        {"variant": "c089_shape_w14", "uplift": 0.02},
        {"variant": "c092_qwen_target_w14", "uplift": 0.04},
        {"variant": "c092_qwen_target_w14", "uplift": 0.04},
        {"variant": "c087_expanded_crop_positive_w14", "uplift": 0.10},
        {"variant": "c087_expanded_crop_positive_w14", "uplift": 0.10},
    )

    summary = summarize_c092_shape_rows(rows)

    assert summary["best_c092_variant"] == "c092_qwen_target_w14"
    assert summary["best_c089_variant"] == "c089_shape_w14"
    assert summary["best_qwen_baseline_variant"] == "c087_expanded_crop_positive_w14"
    assert summary["decision"] == "c092_improves_c089_but_not_qwen_baseline"
