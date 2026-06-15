from __future__ import annotations

from pathlib import Path

from tools.c091_siglip_hard_shape_eval import summarize_c091_shape_rows, write_c091_report


def test_summarize_c091_shape_rows_matches_c089_when_not_beating_qwen() -> None:
    rows = (
        {"variant": "c089_shape_w14", "uplift": 0.025},
        {"variant": "c089_shape_w14", "uplift": 0.025},
        {"variant": "c091_feature_calibrator_w14", "uplift": 0.024},
        {"variant": "c091_feature_calibrator_w14", "uplift": 0.024},
        {"variant": "c087_expanded_crop_positive_w14", "uplift": 0.11},
        {"variant": "c087_expanded_crop_positive_w14", "uplift": 0.10},
    )

    summary = summarize_c091_shape_rows(rows)

    assert summary["best_c091_variant"] == "c091_feature_calibrator_w14"
    assert summary["best_c089_variant"] == "c089_shape_w14"
    assert summary["best_qwen_baseline_variant"] == "c087_expanded_crop_positive_w14"
    assert summary["decision"] == "c091_matches_c089_not_qwen_baseline"


def test_write_c091_report_links_visual_audit(tmp_path: Path) -> None:
    rollup = {
        "decision": "c091_matches_c089_not_qwen_baseline",
        "best_c091_variant": "c091_feature_calibrator_w14",
        "best_c089_variant": "c089_shape_w14",
        "best_qwen_baseline_variant": "c087_expanded_crop_positive_w14",
        "variant_summaries": {
            "c091_feature_calibrator_w14": {
                "mean_uplift": 0.024,
                "improved_rate": 0.72,
                "cases": 11,
            }
        },
    }
    audit = {"blank_count": 2}

    write_c091_report(tmp_path, rollup, audit)

    report = (tmp_path / "report.md").read_text(encoding="utf-8")
    assert "visual_audit.md" in report
    assert "c091_matches_c089_not_qwen_baseline" in report
